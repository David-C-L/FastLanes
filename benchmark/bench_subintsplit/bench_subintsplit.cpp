// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// benchmark/bench_subintsplit/bench_subintsplit.cpp
// ────────────────────────────────────────────────────────
//
// Measures SubIntSplit against the other integer encodings -- and against FastLanes' own wizard choice with and
// without SubIntSplit in the pool -- on encode time, compressed size, and all three read paths.
//
// A note on comparability, because the read paths are not equally fair:
//
//   bulk    comparable across every encoding: they all decode a vector at a time.
//   gather  comparable through the *decode-then-index* path, which every encoding can do: decode the containing
//           vector, then index the requested rows out of it. That is also what Nimble's bulkScan does. SubIntSplit's
//           own `GatherPointwise` is a capability the others do not have and is reported separately.
//   point   likewise reported as decode-then-index for every encoding, so the column stays comparable; SubIntSplit's
//           position-arithmetic `PointAccess` is a separate, non-comparable metric.
//
// Every timed metric is the MINIMUM over repetitions, the robust estimator for a microbenchmark whose noise is
// one-sided. All random probes are pre-generated outside the timed regions from one fixed seed and reused for every
// encoding, so the columns compare encodings rather than random draws.
//
// Dataset root: $FLS_SUBINTSPLIT_DATA_DIR, falling back to the in-repo data/generated/subintsplit.
// Output: benchmark/result/subintsplit/subintsplit.csv
//
#include "data/fastlanes_data.hpp"
#include "fastlanes.hpp"
#include "fls/connection.hpp"
#include "fls/expression/decoding_operator.hpp"
#include "fls/expression/dict_expression.hpp"
#include "fls/expression/rle_expression.hpp"
#include "fls/expression/rpn.hpp"
#include "fls/expression/slpatch_operator.hpp"
#include "fls/expression/subintsplit_operator.hpp"
#include "fls/expression/transpose_operator.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/table_descriptor.hpp"
#include "fls/reader/rowgroup_reader.hpp"
#include "fls/table/rowgroup.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>

using namespace fastlanes; // NOLINT

namespace {

using clock_t_ = std::chrono::steady_clock;
using millis_t = std::chrono::duration<double, std::milli>;
using micros_t = std::chrono::duration<double, std::micro>;

//  Repetition counts. Every timed metric reports the minimum over its repetitions.
constexpr n_t BULK_REPS    = 5;
constexpr n_t POINT_REPS   = 5;
constexpr n_t POINT_PROBES = 20000;
constexpr n_t GATHER_REPS  = 5;
constexpr n_t GATHER_BATCH = 512;
constexpr n_t SEED         = 11;

const vector<n_t> GATHER_WIDTHS {1, 4, 16, 64, 256, CFG::VEC_SZ};

// Keeps every measured loop observable so nothing is optimised away.
uint64_t g_sink {0};

/*--------------------------------------------------------------------------------------------------------------------*\
 * Row specification: what to measure, and how the writing Connection is configured for it.
\*--------------------------------------------------------------------------------------------------------------------*/
enum class Mode : uint8_t {
	Forced,     // force_schema_pool({token}) -- one candidate, no wizard search
	Wizard,     // plain default Connection -- FastLanes' own choice
	WizardNoSis // default Connection minus both SubIntSplit tokens
};

struct RowSpec {
	string        group;
	string        label;
	Mode          mode;
	OperatorToken token {OperatorToken::INVALID}; // only meaningful for Mode::Forced
};

struct DatasetSpec {
	string   name;
	string   dir;
	DataType type;
	n_t      n_rows {0};
	n_t      raw_bytes {0};
};

/*--------------------------------------------------------------------------------------------------------------------*\
 * CSV rows
\*--------------------------------------------------------------------------------------------------------------------*/
struct Row {
	string dataset;
	string group;
	string encoding;
	string metric;
	double value {0.0};
	string unit;
	n_t    detail {0};
	string note;
};

vector<Row> g_rows;

// The note field is the only free-form column, and the consumer parses this file with a plain CSV reader; a comma in
// it would silently shift every following column. Semicolons separate items, '=' and '|' structure them.
void check_no_comma(const string& text, const string& field) {
	if (text.find(',') != string::npos) {
		throw std::runtime_error("bench_subintsplit: '" + field + "' must not contain a comma: " + text);
	}
}

void record(const string& dataset,
            const string& group,
            const string& encoding,
            const string& metric,
            const double  value,
            const string& unit,
            const n_t     detail = 0,
            const string& note   = "") {
	check_no_comma(note, "note");
	check_no_comma(encoding, "encoding");
	check_no_comma(metric, "metric");
	check_no_comma(unit, "unit");

	g_rows.push_back(Row {dataset, group, encoding, metric, value, unit, detail, note});

	std::cout << "   " << std::setw(20) << std::left << encoding << std::setw(26) << metric << std::setw(14)
	          << std::right << std::fixed << std::setprecision(3) << value << " " << std::setw(12) << std::left << unit;
	if (detail != 0) {
		std::cout << " (n=" << detail << ")";
	}
	if (!note.empty()) {
		std::cout << " [" << note << "]";
	}
	std::cout << std::endl;
}

string join(const vector<string>& parts, const string& sep) {
	string out;
	for (n_t i {0}; i < parts.size(); ++i) {
		if (i != 0) {
			out += sep;
		}
		out += parts[i];
	}
	return out;
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * Dataset shape
\*--------------------------------------------------------------------------------------------------------------------*/
n_t element_size(const DataType type) {
	switch (type) {
	case DataType::INT8:
	case DataType::UINT8:
		return 1;
	case DataType::INT16:
	case DataType::UINT16:
		return 2;
	case DataType::INT32:
	case DataType::UINT32:
		return 4;
	case DataType::INT64:
	case DataType::UINT64:
		return 8;
	default:
		throw std::runtime_error("bench_subintsplit: unsupported logical dataset type");
	}
}

// The raw size must come from the actual row count: hardcoding it silently corrupts every compression ratio as soon as
// the datasets are regenerated at a different size.
n_t count_rows(const path& csv_path) {
	std::ifstream in {csv_path, std::ios::binary};
	if (!in) {
		throw std::runtime_error("bench_subintsplit: cannot open " + csv_path.string());
	}

	vector<char> buffer(1u << 20);
	n_t          rows {0};
	char         last {'\n'};
	while (in.read(buffer.data(), static_cast<std::streamsize>(buffer.size())) || in.gcount() > 0) {
		const auto n = static_cast<n_t>(in.gcount());
		for (n_t i {0}; i < n; ++i) {
			if (buffer[i] == '\n') {
				++rows;
			}
			last = buffer[i];
		}
	}
	if (last != '\n') { // a final line without a trailing newline still holds a value
		++rows;
	}
	return rows;
}

// TableReader exposes no rowgroup count, so the table descriptor is read the same way TableReader itself reads it.
n_t count_rowgroups(const path& fls_path) {
	FileHeader file_header {};
	FileFooter file_footer {};
	FileHeader::Load(file_header, fls_path);
	FileFooter::Load(file_footer, fls_path);

	const auto handle =
	    file_header.settings.inline_footer
	        ? make_table_descriptor(fls_path, file_footer.table_descriptor_offset, file_footer.table_descriptor_size)
	        : make_table_descriptor(fls_path.parent_path() / "table_descriptor.fbb");

	const auto* rowgroups = handle->Get()->m_rowgroup_descriptors();
	return rowgroups == nullptr ? 0 : static_cast<n_t>(rowgroups->size());
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * Probe plan
 *
 * Every random draw is made here, before any timer starts, and the same plan is reused for every encoding of a
 * dataset. Drawing inside a timed loop charges the RNG to the measurement -- and charges it in proportion to the
 * gather width, which is exactly the axis being swept.
\*--------------------------------------------------------------------------------------------------------------------*/
struct Layout {
	n_t         n_rowgroups {0};
	vector<n_t> n_vec; // per rowgroup

	bool operator==(const Layout& other) const {
		return n_rowgroups == other.n_rowgroups && n_vec == other.n_vec;
	}
};

struct ProbePlan {
	Layout layout;
	// Point probes: (rowgroup, vector, row) triples spread over the whole file so the working set is the real one.
	vector<n_t>   point_rg;
	vector<n_t>   point_vec;
	vector<idx_t> point_row;
	// Gather probes: one (rowgroup, vector) per gather plus a block of CFG::VEC_SZ candidate rows, of which the first
	// `n` are used at width `n`.
	vector<n_t>   gather_rg;
	vector<n_t>   gather_vec;
	vector<idx_t> gather_row;
};

ProbePlan make_plan(const Layout& layout) {
	std::mt19937_64 rng {SEED};
	ProbePlan       plan;
	plan.layout = layout;

	const auto draw_position = [&](n_t& rg, n_t& vec) {
		rg  = rng() % layout.n_rowgroups;
		vec = rng() % layout.n_vec[rg];
	};

	plan.point_rg.resize(POINT_PROBES);
	plan.point_vec.resize(POINT_PROBES);
	plan.point_row.resize(POINT_PROBES);
	for (n_t i {0}; i < POINT_PROBES; ++i) {
		draw_position(plan.point_rg[i], plan.point_vec[i]);
		plan.point_row[i] = static_cast<idx_t>(rng() % CFG::VEC_SZ);
	}

	plan.gather_rg.resize(GATHER_BATCH);
	plan.gather_vec.resize(GATHER_BATCH);
	plan.gather_row.resize(GATHER_BATCH * CFG::VEC_SZ);
	for (n_t g {0}; g < GATHER_BATCH; ++g) {
		draw_position(plan.gather_rg[g], plan.gather_vec[g]);
		for (n_t i {0}; i < CFG::VEC_SZ; ++i) {
			plan.gather_row[(g * CFG::VEC_SZ) + i] = static_cast<idx_t>(rng() % CFG::VEC_SZ);
		}
	}

	return plan;
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * Generic decode-then-index harness
 *
 * Mirrors src/encoder/materializer.cpp's material_visitor. Operators whose decoded values already sit in logical order
 * are read in place; everything else goes through Materialize.
 *
 * dec_transpose_opr (delta) MUST go through Materialize: `transposed_data` is in FastLanes transposed order, so
 * indexing it directly returns wrong values and an unrealistically fast number. The untranspose stays inside the timed
 * region because a real point read on a delta column genuinely pays it.
 *
 * Materialize implementations disagree about vec_idx -- some append, some resize(size + VEC_SZ) and write at
 * vec_idx * VEC_SZ. One rule satisfies both: reserve the whole rowgroup once, outside every timed region, then
 * resize(vec_idx * VEC_SZ) per probe, which is O(1) and never reallocates.
\*--------------------------------------------------------------------------------------------------------------------*/
template <typename PT>
struct VecAccess {
	const PT* base {nullptr};
	PT        constant {};
	bool      is_constant {false};

	[[nodiscard]] PT at(const idx_t row) const {
		return is_constant ? constant : base[row];
	}
};

// A dictionary whose keys are exactly the column's physical type; every other dict instantiation must not match.
template <typename T, typename PT>
struct is_dict_of : std::false_type {};
template <typename INDEX_PT, typename PT>
struct is_dict_of<sp<dec_dict_opr<PT, INDEX_PT>>, PT> : std::true_type {};
template <typename T, typename PT>
constexpr bool is_dict_of_v = is_dict_of<T, PT>::value;

template <typename PT>
struct IndexVisitor {
	using UT = make_unsigned_t<PT>;

	n_t            vec_idx;
	TypedCol<PT>*  scratch;
	VecAccess<PT>* out;
	bool*          unsupported;

	template <typename T>
	void operator()(const T& opr) const {
		using opr_t = std::decay_t<T>;

		if constexpr (std::is_same_v<opr_t, sp<dec_uncompressed_opr<PT>>>) {
			out->base = opr->Data();
		} else if constexpr (std::is_same_v<opr_t, sp<dec_unffor_opr<PT>>>) {
			out->base = opr->Data();
		} else if constexpr (std::is_same_v<opr_t, sp<dec_unffor_opr<UT>>>) {
			// unffor is instantiated on the unsigned type; the bit pattern is the value.
			out->base = reinterpret_cast<const PT*>(opr->Data());
		} else if constexpr (std::is_same_v<opr_t, sp<dec_slpatch_opr<PT>>>) {
			out->base = opr->data;
		} else if constexpr (std::is_same_v<opr_t, sp<dec_subintsplit_opr<PT>>>) {
			out->base = opr->data;
		} else if constexpr (std::is_same_v<opr_t, sp<dec_constant_opr<PT>>>) {
			out->is_constant = true;
			out->constant    = opr->value;
		} else if constexpr (std::is_same_v<opr_t, sp<PhysicalExpr>>) {
			visit_dec(*this, opr->operators[opr->operators.size() - 1]);
		} else if constexpr (requires { opr->Materialize(vec_idx, *scratch); }) {
			materialize_into();
			opr->Materialize(vec_idx, *scratch);
			out->base = scratch->data.data() + (vec_idx * CFG::VEC_SZ);
		} else if constexpr (requires { opr->Decode(vec_idx, scratch->data); }) {
			materialize_into();
			opr->Decode(vec_idx, scratch->data);
			out->base = scratch->data.data() + (vec_idx * CFG::VEC_SZ);
		} else if constexpr (is_dict_of_v<opr_t, PT>) {
			materialize_into();
			scratch->data.resize((vec_idx + 1) * CFG::VEC_SZ);
			const auto* keys  = opr->Keys();
			const auto* index = opr->Index();
			PT*         dst   = scratch->data.data() + (vec_idx * CFG::VEC_SZ);
			for (n_t i {0}; i < CFG::VEC_SZ; ++i) {
				dst[i] = keys[index[i]];
			}
			out->base = dst;
		} else {
			*unsupported = true;
		}
	}

	void materialize_into() const {
		scratch->data.resize(vec_idx * CFG::VEC_SZ); // O(1): capacity was reserved outside every timed region
	}
};

template <typename PT>
class Harness {
public:
	Harness(vector<up<RowgroupReader>>& readers, const Layout& layout)
	    : m_readers(readers) {
		const n_t max_n_vec = *std::max_element(layout.n_vec.begin(), layout.n_vec.end());
		m_scratch.data.reserve(max_n_vec * CFG::VEC_SZ);
	}

	// Decode the vector containing the probe and hand back a logically-ordered view of it.
	VecAccess<PT> decode(const n_t rowgroup_idx, const n_t vec_idx) {
		auto& expressions = m_readers[rowgroup_idx]->get_chunk(vec_idx);
		auto& expression  = *expressions[0];
		expression.PointTo(vec_idx); // exactly what Materializer::Materialize does before it reads

		VecAccess<PT> access;
		visit_dec(IndexVisitor<PT> {vec_idx, &m_scratch, &access, &m_unsupported},
		          expression.operators[expression.operators.size() - 1]);
		return access;
	}

	PT value_at(const n_t rowgroup_idx, const n_t vec_idx, const idx_t row) {
		return decode(rowgroup_idx, vec_idx).at(row);
	}

	[[nodiscard]] bool unsupported() const {
		return m_unsupported;
	}

private:
	vector<up<RowgroupReader>>& m_readers;
	TypedCol<PT>                m_scratch;
	bool                        m_unsupported {false};
};

/*--------------------------------------------------------------------------------------------------------------------*\
 * Timing helpers
\*--------------------------------------------------------------------------------------------------------------------*/
template <typename F>
double min_over(const n_t reps, const n_t divisor, F&& body) {
	double best = std::numeric_limits<double>::infinity();
	body(); // untimed warm-up: the first pass pays page faults and cold caches for everybody
	for (n_t rep {0}; rep < reps; ++rep) {
		const auto start = clock_t_::now();
		body();
		const double elapsed = micros_t {clock_t_::now() - start}.count();
		best                 = std::min(best, elapsed / static_cast<double>(divisor));
	}
	return best;
}

string reps_note(const n_t reps, const string& extra = "") {
	string note = "min_of=" + std::to_string(reps);
	if (!extra.empty()) {
		note += ";" + extra;
	}
	return note;
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * Writing
\*--------------------------------------------------------------------------------------------------------------------*/
path write_fls(const DatasetSpec& spec, const RowSpec& row_spec, const path& out_dir, double& encode_ms) {
	std::filesystem::remove_all(out_dir);
	std::filesystem::create_directories(out_dir);

	Connection conn;
	conn.reset();
	switch (row_spec.mode) {
	case Mode::Forced:
		conn.force_schema_pool({row_spec.token});
		break;
	case Mode::Wizard:
		break; // the default candidate pool, i.e. what FastLanes would do on its own
	case Mode::WizardNoSis:
		// Cast() can narrow the column, so both widths have to go or the wizard just picks the other one.
		conn.disable_encoding(OperatorToken::EXP_SUBINTSPLIT_I64);
		conn.disable_encoding(OperatorToken::EXP_SUBINTSPLIT_I32);
		break;
	}

	conn.read_csv(path {spec.dir}); // parsing is excluded from the encode timing below

	const auto start = clock_t_::now();
	conn.to_fls(out_dir / "data.fls");
	encode_ms = millis_t {clock_t_::now() - start}.count();

	return out_dir / "data.fls";
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * SubIntSplit operator lookup
\*--------------------------------------------------------------------------------------------------------------------*/
template <typename PT>
sp<dec_subintsplit_opr<PT>> find_operator(RowgroupReader& rowgroup_reader) {
	sp<dec_subintsplit_opr<PT>> found;
	for (auto& expression : rowgroup_reader.m_expressions) {
		for (auto& physical_operator : expression->operators) {
			visit_dec(
			    [&](const auto& candidate) {
				    using candidate_t = std::decay_t<decltype(candidate)>;
				    if constexpr (std::is_same_v<candidate_t, sp<dec_subintsplit_opr<PT>>>) {
					    found = candidate;
				    }
			    },
			    physical_operator);
		}
	}
	return found;
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * Per-encoding measurement
\*--------------------------------------------------------------------------------------------------------------------*/
template <typename PT>
void cross_check(const DatasetSpec&          spec,
                 const RowSpec&              row_spec,
                 vector<up<RowgroupReader>>& readers,
                 Harness<PT>&                harness,
                 const ProbePlan&            plan) {
	// One full materialize() per checked rowgroup, compared against the harness on several random triples. Cheap, and
	// it protects every number in the table -- above all delta, where reading the operator's buffer directly would be
	// wrong and fast.
	const n_t n_checked = std::min<n_t>(2, plan.layout.n_rowgroups);
	n_t       n_compared {0};

	for (n_t rg {0}; rg < n_checked; ++rg) {
		const auto  rowgroup = readers[rg]->materialize();
		const auto& column   = std::get<up<TypedCol<PT>>>(rowgroup->internal_rowgroup[0]);

		for (n_t i {0}; i < POINT_PROBES && n_compared < 64; ++i) {
			if (plan.point_rg[i] != rg) {
				continue;
			}
			const n_t   vec_idx  = plan.point_vec[i];
			const idx_t row      = plan.point_row[i];
			const PT    expected = column->data[(vec_idx * CFG::VEC_SZ) + row];
			const PT    got      = harness.value_at(rg, vec_idx, row);
			if (expected != got) {
				throw std::runtime_error("bench_subintsplit: decode-then-index harness disagrees with materialize() "
				                         "for " +
				                         spec.name + "/" + row_spec.label + " at rowgroup " + std::to_string(rg) +
				                         " vector " + std::to_string(vec_idx) + " row " + std::to_string(row) +
				                         ": expected " + std::to_string(expected) + " got " + std::to_string(got));
			}
			++n_compared;
		}
	}

	if (harness.unsupported()) {
		throw std::runtime_error("bench_subintsplit: no decode-then-index path for the operator chosen for " +
		                         spec.name + "/" + row_spec.label);
	}
	if (n_compared == 0) {
		throw std::runtime_error("bench_subintsplit: cross-check compared nothing for " + spec.name);
	}
	std::cout << "   (harness cross-checked against materialize() on " << n_compared << " probes)" << std::endl;
}

template <typename PT>
void bench_sis_paths(const DatasetSpec&          spec,
                     const RowSpec&              row_spec,
                     vector<up<RowgroupReader>>& readers,
                     const ProbePlan&            plan) {
	if constexpr (std::is_same_v<PT, i64_pt> || std::is_same_v<PT, i32_pt>) {
		vector<sp<dec_subintsplit_opr<PT>>> oprs(plan.layout.n_rowgroups);
		for (n_t rg {0}; rg < plan.layout.n_rowgroups; ++rg) {
			oprs[rg] = find_operator<PT>(*readers[rg]);
			if (!oprs[rg]) {
				return; // not a SubIntSplit column (or not in every rowgroup): no native paths to report
			}
		}

		// Layout: the section boundaries, plus the per-section bit width as a MEDIAN over vectors. Vector 0 alone is
		// not representative of a rowgroup.
		//
		// The DP selector runs once per column PER ROWGROUP, so different rowgroups of the same column can pick
		// different section counts and boundaries -- the synthetic datasets happen to converge on one plan everywhere,
		// but real data does not. Reading rowgroup rg's bw_segment_views[s] for s beyond ITS OWN section count is
		// out-of-bounds, so bw_medians is only computed up to the minimum section count seen across all rowgroups, and
		// any disagreement (in count or in boundaries) is surfaced via the same ";varies_by_rowgroup" convention
		// schema_note uses, rather than silently reporting rowgroup 0's plan as if it were universal.
		n_t  min_sections = oprs[0]->bit_starts.size();
		bool varies       = false;
		for (n_t rg {1}; rg < plan.layout.n_rowgroups; ++rg) {
			min_sections = std::min(min_sections, oprs[rg]->bit_starts.size());
			if (oprs[rg]->bit_starts != oprs[0]->bit_starts) {
				varies = true;
			}
		}
		const n_t      n_sections = oprs[0]->bit_starts.size();
		vector<string> starts;
		for (n_t s {0}; s < n_sections; ++s) {
			starts.push_back(std::to_string(static_cast<n_t>(oprs[0]->bit_starts[s])));
		}
		vector<string> bw_medians;
		for (n_t s {0}; s < min_sections; ++s) {
			vector<n_t> widths;
			for (n_t rg {0}; rg < plan.layout.n_rowgroups; ++rg) {
				for (n_t vec_idx {0}; vec_idx < plan.layout.n_vec[rg]; ++vec_idx) {
					oprs[rg]->PointTo(vec_idx);
					widths.push_back(*reinterpret_cast<const bw_t*>(oprs[rg]->bw_segment_views[s].data));
				}
			}
			std::sort(widths.begin(), widths.end());
			bw_medians.push_back(std::to_string(widths[widths.size() / 2]));
		}
		record(spec.name,
		       row_spec.group,
		       row_spec.label,
		       "sis_layout",
		       static_cast<double>(n_sections),
		       "sections",
		       n_sections,
		       "starts=" + join(starts, ";") + "|bw_med=" + join(bw_medians, ";") +
		           (varies ? "|varies_by_rowgroup" : ""));

		// Native point access: position arithmetic, no vector decode.
		const double point_us = min_over(POINT_REPS, POINT_PROBES, [&] {
			for (n_t i {0}; i < POINT_PROBES; ++i) {
				g_sink ^=
				    static_cast<uint64_t>(oprs[plan.point_rg[i]]->PointAccess(plan.point_vec[i], plan.point_row[i]));
			}
		});
		record(spec.name,
		       row_spec.group,
		       row_spec.label,
		       "point_access",
		       point_us,
		       "us/probe",
		       0,
		       reps_note(POINT_REPS, "probes=" + std::to_string(POINT_PROBES)));

		// Gather sweep, both native modes.
		vector<PT> out(CFG::VEC_SZ);
		for (const n_t n : GATHER_WIDTHS) {
			for (const bool pointwise : {true, false}) {
				const double us = min_over(GATHER_REPS, GATHER_BATCH, [&] {
					for (n_t g {0}; g < GATHER_BATCH; ++g) {
						const idx_t* rows = plan.gather_row.data() + (g * CFG::VEC_SZ);
						if (pointwise) {
							oprs[plan.gather_rg[g]]->GatherPointwise(plan.gather_vec[g], rows, n, out.data());
						} else {
							oprs[plan.gather_rg[g]]->GatherDecoded(plan.gather_vec[g], rows, n, out.data());
						}
						g_sink ^= static_cast<uint64_t>(out[0]);
					}
				});
				record(spec.name,
				       row_spec.group,
				       row_spec.label,
				       pointwise ? "gather_pointwise" : "gather_decoded",
				       us,
				       "us/gather",
				       n,
				       reps_note(GATHER_REPS, "batch=" + std::to_string(GATHER_BATCH)));
			}
		}
	} else {
		(void)spec;
		(void)row_spec;
		(void)readers;
		(void)plan;
	}
}

template <typename PT>
void bench_read_paths(const DatasetSpec&          spec,
                      const RowSpec&              row_spec,
                      vector<up<RowgroupReader>>& readers,
                      const ProbePlan&            plan) {
	Harness<PT> harness {readers, plan.layout};
	cross_check<PT>(spec, row_spec, readers, harness, plan);

	// Bulk decode: every rowgroup, every vector.
	const double bulk_ms = min_over(BULK_REPS, plan.layout.n_rowgroups * 1000, [&] {
		for (n_t rg {0}; rg < plan.layout.n_rowgroups; ++rg) {
			for (n_t vec_idx {0}; vec_idx < plan.layout.n_vec[rg]; ++vec_idx) {
				readers[rg]->get_chunk(vec_idx);
			}
		}
	});
	record(spec.name,
	       row_spec.group,
	       row_spec.label,
	       "bulk_decode",
	       bulk_ms,
	       "ms/rowgroup",
	       0,
	       reps_note(BULK_REPS, "rowgroups=" + std::to_string(plan.layout.n_rowgroups)));

	// Point: decode the containing vector, then index. The path every encoding has.
	const double point_us = min_over(POINT_REPS, POINT_PROBES, [&] {
		for (n_t i {0}; i < POINT_PROBES; ++i) {
			g_sink ^= static_cast<uint64_t>(harness.value_at(plan.point_rg[i], plan.point_vec[i], plan.point_row[i]));
		}
	});
	record(spec.name,
	       row_spec.group,
	       row_spec.label,
	       "point_decode_then_index",
	       point_us,
	       "us/probe",
	       0,
	       reps_note(POINT_REPS, "probes=" + std::to_string(POINT_PROBES)));

	// Gather: decode the containing vector once, then index the requested rows out of it.
	for (const n_t n : GATHER_WIDTHS) {
		const double us = min_over(GATHER_REPS, GATHER_BATCH, [&] {
			for (n_t g {0}; g < GATHER_BATCH; ++g) {
				const auto   access = harness.decode(plan.gather_rg[g], plan.gather_vec[g]);
				const idx_t* rows   = plan.gather_row.data() + (g * CFG::VEC_SZ);
				for (n_t i {0}; i < n; ++i) {
					g_sink ^= static_cast<uint64_t>(access.at(rows[i]));
				}
			}
		});
		record(spec.name,
		       row_spec.group,
		       row_spec.label,
		       "gather_decode_then_index",
		       us,
		       "us/gather",
		       n,
		       reps_note(GATHER_REPS, "batch=" + std::to_string(GATHER_BATCH)));
	}

	bench_sis_paths<PT>(spec, row_spec, readers, plan);
}

// The physical type is only known once the file is written: a wizard row may narrow it through Cast().
template <typename F>
void dispatch_on_type(const DataType data_type, F&& body) {
	switch (data_type) {
	case DataType::INT64:
		body(i64_pt {});
		return;
	case DataType::INT32:
		body(i32_pt {});
		return;
	case DataType::INT16:
		body(i16_pt {});
		return;
	case DataType::INT8:
		body(i08_pt {});
		return;
	default:
		throw std::runtime_error(string {"bench_subintsplit: unhandled physical type "} + EnumNameDataType(data_type));
	}
}

// The encoding actually used, read off the descriptor of the file just written. CompressionRatioBenchmarker::get_schema
// would re-Write with a default Connection and therefore report the wizard's choice for every row.
string schema_note(const vector<up<RowgroupReader>>& readers) {
	const auto tokens_of = [](const RowgroupReader& reader) {
		const auto*    column_descriptor = reader.get_descriptor().m_column_descriptors()->Get(0);
		vector<string> names;
		const auto*    rpn = column_descriptor->encoding_rpn();
		if (rpn != nullptr && rpn->operator_tokens() != nullptr) {
			for (const auto token : *rpn->operator_tokens()) {
				names.push_back(token_to_string(token));
			}
		}
		return join(names, ";");
	};

	const string first = tokens_of(*readers[0]);
	for (n_t rg {1}; rg < readers.size(); ++rg) {
		if (tokens_of(*readers[rg]) != first) {
			return first + ";varies_by_rowgroup";
		}
	}
	return first;
}

void run_row(const DatasetSpec& spec, const RowSpec& row_spec, const path& work_dir, std::optional<ProbePlan>& plan) {
	double     encode_ms {0.0};
	const path fls_path = write_fls(spec, row_spec, work_dir, encode_ms);
	const auto size     = static_cast<double>(std::filesystem::file_size(fls_path));

	record(spec.name, row_spec.group, row_spec.label, "encode_time", encode_ms, "ms", 0, "min_of=1");
	record(spec.name, row_spec.group, row_spec.label, "compressed_size", size, "bytes");
	record(spec.name,
	       row_spec.group,
	       row_spec.label,
	       "compression_ratio",
	       static_cast<double>(spec.raw_bytes) / size,
	       "x",
	       0,
	       "raw_bytes=" + std::to_string(spec.raw_bytes) + ";rows=" + std::to_string(spec.n_rows));

	Connection conn;
	const auto reader = conn.reset().read_fls(fls_path);

	Layout                     layout;
	vector<up<RowgroupReader>> readers;
	layout.n_rowgroups = count_rowgroups(fls_path);
	for (n_t rg {0}; rg < layout.n_rowgroups; ++rg) {
		readers.push_back(reader->get_rowgroup_reader(rg));
		layout.n_vec.push_back(readers.back()->get_descriptor().m_n_vec());
	}

	// One plan per dataset, shared by every encoding, so the columns differ only in the encoding.
	if (!plan.has_value() || !(plan->layout == layout)) {
		plan = make_plan(layout);
	}

	const auto* column_descriptor = readers[0]->get_descriptor().m_column_descriptors()->Get(0);
	const auto  data_type         = column_descriptor->data_type();

	record(spec.name, row_spec.group, row_spec.label, "schema", 0.0, "token", 0, schema_note(readers));
	record(spec.name, row_spec.group, row_spec.label, "data_type", 0.0, "type", 0, EnumNameDataType(data_type));

	dispatch_on_type(data_type, [&](auto tag) {
		using PT = decltype(tag);
		bench_read_paths<PT>(spec, row_spec, readers, *plan);
	});
}

void run_dataset(const DatasetSpec& spec, const vector<RowSpec>& row_specs) {
	std::cout << "\n=== " << spec.name << "  (" << spec.n_rows << " rows, raw " << spec.raw_bytes << " B)" << std::endl;

	const path               work_dir = std::filesystem::temp_directory_path() / ("bench_subintsplit_" + spec.name);
	std::optional<ProbePlan> plan;

	for (const auto& row_spec : row_specs) {
		std::cout << " -- " << row_spec.label << std::endl;
		run_row(spec, row_spec, work_dir, plan);
	}

	std::filesystem::remove_all(work_dir);
}

vector<RowSpec> make_row_specs(const DataType type) {
	const bool is_64 = type == DataType::INT64;
	return {
	    {"baseline",
	     "uncompressed",
	     Mode::Forced,
	     is_64 ? OperatorToken::EXP_UNCOMPRESSED_I64 : OperatorToken::EXP_UNCOMPRESSED_I32},
	    {"codec", "ffor", Mode::Forced, is_64 ? OperatorToken::EXP_FFOR_I64 : OperatorToken::EXP_FFOR_I32},
	    {"codec", "delta", Mode::Forced, is_64 ? OperatorToken::EXP_DELTA_I64 : OperatorToken::EXP_DELTA_I32},
	    {"codec",
	     "ffor_slpatch",
	     Mode::Forced,
	     is_64 ? OperatorToken::EXP_FFOR_SLPATCH_I64 : OperatorToken::EXP_FFOR_SLPATCH_I32},
	    {"codec",
	     "subintsplit",
	     Mode::Forced,
	     is_64 ? OperatorToken::EXP_SUBINTSPLIT_I64 : OperatorToken::EXP_SUBINTSPLIT_I32},
	    {"wizard", "wizard_with_sis", Mode::Wizard},
	    {"wizard", "wizard_without_sis", Mode::WizardNoSis},
	};
}

} // namespace

int main() {
	const char*  env_root = std::getenv("FLS_SUBINTSPLIT_DATA_DIR");
	const string data_root =
	    env_root != nullptr ? string {env_root} : string {FLS_CMAKE_SOURCE_DIR} + "/data/generated/subintsplit";
	std::cout << "dataset root: " << data_root << std::endl;

	vector<DatasetSpec> datasets = {
	    {"snowflake_i64", data_root + "/snowflake_i64", DataType::INT64},
	    {"tpch_partkey_i32", data_root + "/tpch_partkey_i32", DataType::INT32},
	    {"ipv4_i32", data_root + "/ipv4_i32", DataType::INT32},
	    // Real Twitter snowflake IDs, not simulated. Optional: only present when
	    // extract_real_snowflake.py has been run, since it needs the EncodingsPlayground parquet.
	    {"snowflake_i64_real", data_root + "/snowflake_i64_real", DataType::INT64},
	};

	for (auto& spec : datasets) {
		if (!std::filesystem::exists(path {spec.dir} / "generated.csv")) {
			std::cout << "-- skipping " << spec.name << " (no dataset at " << spec.dir << ")" << std::endl;
			continue;
		}
		spec.n_rows    = count_rows(path {spec.dir} / "generated.csv");
		spec.raw_bytes = spec.n_rows * element_size(spec.type);
		run_dataset(spec, make_row_specs(spec.type));
	}

	const path result_path =
	    path {string {FLS_CMAKE_SOURCE_DIR}} / "benchmark" / "result" / "subintsplit" / "subintsplit.csv";
	std::filesystem::create_directories(result_path.parent_path());
	std::ofstream csv {result_path};
	csv << "dataset,group,encoding,metric,value,unit,detail,note\n";
	for (const auto& row : g_rows) {
		csv << row.dataset << "," << row.group << "," << row.encoding << "," << row.metric << "," << std::fixed
		    << std::setprecision(6) << row.value << "," << row.unit << "," << row.detail << "," << row.note << "\n";
	}
	std::cout << "\n-- results written to " << result_path << std::endl;

	if (g_sink == std::numeric_limits<uint64_t>::max()) {
		std::cout << "" << std::endl; // keeps every measured loop observable
	}

	return EXIT_SUCCESS;
}
