// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// benchmark/bench_subintsplit/BenchSubIntSplitCommon.hpp
// ────────────────────────────────────────────────────────
//
// Shared measurement machinery for the SubIntSplit benchmark drivers: everything from bench_subintsplit.cpp except
// make_row_specs() and main(), so a sibling driver can supply its own codec list / row specs and main() while reusing
// this harness unchanged.
//
// NOTE: the contents below sit in an anonymous namespace, matching how they lived in the original .cpp (internal to
// one benchmark binary). If this header is ever #include'd from more than one .cpp linked into the SAME binary, each
// TU gets its own copy of these symbols -- harmless for correctness (anonymous-namespace symbols have internal
// linkage) but wasteful, and g_rows/g_sink would NOT be shared across TUs in that case. Fine for two separate
// executables, which is the current plan.
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
#ifndef FLS_BENCHMARK_BENCH_SUBINTSPLIT_COMMON_HPP
#define FLS_BENCHMARK_BENCH_SUBINTSPLIT_COMMON_HPP

// Set by CMake so the metadata sidecar can say how the binary was built; a
// timing from a Debug build is not a timing.
#ifndef FLS_BENCH_BUILD_TYPE
#define FLS_BENCH_BUILD_TYPE "unknown"
#endif

#include "data/fastlanes_data.hpp"
#include "fastlanes.hpp"
#include "fls/connection.hpp"
#include "fls/expression/decoding_operator.hpp"
#include "fls/expression/dict_expression.hpp"
#include "fls/expression/frequency_operator.hpp"
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
#include <ctime>
#include <thread>
#include <stdexcept>

using namespace fastlanes; // NOLINT

namespace {

using clock_t_ = std::chrono::steady_clock;
using millis_t = std::chrono::duration<double, std::milli>;
using micros_t = std::chrono::duration<double, std::micro>;

//  Repetition counts. Every timed metric reports the MEDIAN over its repetitions.
//
//  This used to be the minimum, which is the better estimator for an isolated
//  microbenchmark whose noise is one-sided. It is the wrong one here: these
//  numbers are read next to BtrBlocks' and Nimble's, and both of those reduce by
//  median. A minimum on this side against a median on theirs would show
//  FastLanes faster by exactly the width of its own noise, which is a fact about
//  the estimator rather than about the format.
constexpr n_t BULK_REPS    = 5;
constexpr n_t POINT_REPS   = 5;
// Aligned across all three harnesses at 256, downward from this harness's
// 20,000 and Nimble's 128.
//
// Alignment means the same count everywhere, and the defensible common count is
// a small one: for any codec that decodes a vector or a block per access the
// per-probe cost is constant, so a small sample estimates the same ns/probe as
// a large one. Measured on the BtrBlocks harness at 64 / 256 / 20,000 probes,
// ns/probe agrees within 1.17x for every codec and within 1.10x at 256 against
// 20,000, while the point phase falls from 34% of a run to 1.4%. 64 is biased
// high (up to 1.17x) because first-touch effects do not amortise over so few
// probes; 256 removes most of that for nothing.
constexpr n_t POINT_PROBES = 256;
constexpr n_t GATHER_REPS  = 5;
constexpr n_t GATHER_BATCH = 512;
constexpr n_t RANGE_REPS   = 5;
constexpr n_t SEED         = 11;

const vector<n_t> GATHER_WIDTHS {1, 4, 16, 64, 256, CFG::VEC_SZ};

// Range reads: B contiguous elements starting at a uniformly drawn offset, at
// log-spaced B. This is the axis the cross-format comparison had no evidence on
// at all, and the one the SubIntSplit claim rests on: a scheme with an
// addressable interior answers a small range by touching only the vectors the
// range covers, while a scheme without one pays for everything in front of it.
//
// RANGE_OFFSETS separate offsets per width, drawn once and reused by every
// encoding, so the columns compare encodings rather than random draws.
// Offsets per range width, parallel to RANGE_WIDTHS.
//
// A constant count is the wrong shape. The spread across offsets is not
// constant in B: it is ~1.03x at B=1, peaks around 7.4x at B=512, and falls
// back to ~1.6x by B=32768. At the small end almost every offset costs the same
// (one vector, one decode) so extra offsets buy nothing; at the large end each
// offset costs B element-reads, so extra offsets are the most expensive samples
// in the sweep and the spread does not justify them. Sampling is concentrated
// in the middle, where the answer actually varies.
//
// Against a flat 32 this is 2.6M element-reads per repeat instead of 9.6M --
// 3.7x less work -- while sampling B=512 harder than the flat schedule did.
const vector<n_t> RANGE_WIDTHS {1, 8, 64, 512, 4096, 32768, 262144};
const vector<n_t> RANGE_OFFSETS_PER_WIDTH {8, 16, 24, 48, 24, 12, 8};
// The widest schedule entry, for sizing the offset table.
constexpr n_t RANGE_OFFSETS_MAX = 48;

// Keeps every measured loop observable so nothing is optimised away.
uint64_t g_sink {0};

/*--------------------------------------------------------------------------------------------------------------------*\
 * Row specification: what to measure, and how the writing Connection is configured for it.
\*--------------------------------------------------------------------------------------------------------------------*/
enum class Mode : uint8_t {
	Forced,            // force_schema_pool({token}) -- one candidate, no wizard search
	Wizard,            // plain default Connection -- FastLanes' own choice
	WizardNoSis,       // default Connection minus both SubIntSplit tokens
	WizardLimited,     // default Connection minus RLE_slpatch/Delta/CrossRLE (both widths)
	WizardLimitedNoSis // WizardLimited minus both SubIntSplit tokens
};

struct RowSpec {
	string                group;
	string                label;
	Mode                  mode;
	OperatorToken         token {OperatorToken::INVALID}; // only meaningful for Mode::Forced; ignored if tokens is set
	vector<OperatorToken> tokens;                         // only meaningful for Mode::Forced: a multi-candidate pool,
	                                                       // e.g. offering every Dictionary width variant and letting
	                                                       // the wizard's evaluate_expressions() pick the cheapest one.
	                                                       // Empty means "use token" (the single-candidate case).
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
	// Range reads: RANGE_OFFSETS global start offsets per entry of RANGE_WIDTHS,
	// laid out width-major. A width wider than the file contributes no offsets
	// and is skipped by the driver.
	vector<n_t> range_start;
	n_t         total_rows {0};
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

	// Range starts are drawn over global row offsets, so a range is free to span
	// a vector boundary and a rowgroup boundary -- which is what a real range
	// read does, and where the cost of an addressable interior actually shows.
	plan.total_rows = 0;
	for (n_t rg {0}; rg < layout.n_rowgroups; ++rg) {
		plan.total_rows += layout.n_vec[rg] * CFG::VEC_SZ;
	}
	// Laid out width-major with a fixed stride of RANGE_OFFSETS_MAX, so a width
	// that uses fewer offsets simply leaves the tail of its row unused.
	plan.range_start.resize(RANGE_WIDTHS.size() * RANGE_OFFSETS_MAX, 0);
	for (n_t w {0}; w < RANGE_WIDTHS.size(); ++w) {
		if (RANGE_WIDTHS[w] > plan.total_rows) {
			continue;
		}
		const n_t last_start = plan.total_rows - RANGE_WIDTHS[w];
		for (n_t o {0}; o < RANGE_OFFSETS_PER_WIDTH[w]; ++o) {
			plan.range_start[(w * RANGE_OFFSETS_MAX) + o] = last_start == 0 ? 0 : rng() % (last_start + 1);
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
[[maybe_unused]] double min_over(const n_t reps, const n_t divisor, F&& body) {
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

// The reduction every timed metric in this harness uses. See the note on the
// repetition counts above for why it is not the minimum.
template <typename F>
double median_over(const n_t reps, const n_t divisor, F&& body) {
	vector<double> samples;
	samples.reserve(reps);
	body(); // untimed warm-up: the first pass pays page faults and cold caches for everybody
	for (n_t rep {0}; rep < reps; ++rep) {
		const auto start = clock_t_::now();
		body();
		samples.push_back(micros_t {clock_t_::now() - start}.count() / static_cast<double>(divisor));
	}
	std::sort(samples.begin(), samples.end());
	const n_t mid = samples.size() / 2;
	return (samples.size() % 2 == 0) ? 0.5 * (samples[mid - 1] + samples[mid]) : samples[mid];
}

string reps_note(const n_t reps, const string& extra = "") {
	string note = "median_of=" + std::to_string(reps);
	if (!extra.empty()) {
		note += ";" + extra;
	}
	return note;
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * Dataset discovery
 *
 * The dataset list used to be a literal in each driver's main(), which meant measuring a new column was a code change
 * in two files. It is now a manifest: $FLS_SUBINTSPLIT_MANIFEST names a CSV of `name,dir,type` rows, and the built-in
 * list is the fallback when no manifest is set. Adding XMark, OSM or a Public BI column is then a manifest line plus a
 * dataset directory, both produced by scripts/make_cross_format_datasets.py.
\*--------------------------------------------------------------------------------------------------------------------*/
std::optional<DataType> parse_data_type(const string& text) {
	if (text == "INT64" || text == "i64") {
		return DataType::INT64;
	}
	if (text == "INT32" || text == "i32") {
		return DataType::INT32;
	}
	return std::nullopt;
}

vector<DatasetSpec> load_manifest(const path& manifest_path, const vector<DatasetSpec>& fallback) {
	if (manifest_path.empty()) {
		return fallback;
	}
	std::ifstream in {manifest_path};
	if (!in.good()) {
		throw std::runtime_error("cannot read dataset manifest: " + manifest_path.string());
	}
	vector<DatasetSpec> specs;
	string              line;
	n_t                 line_no {0};
	while (std::getline(in, line)) {
		++line_no;
		if (line.empty() || line[0] == '#') {
			continue;
		}
		const auto first = line.find(',');
		if (first == string::npos) {
			throw std::runtime_error("manifest line " + std::to_string(line_no) + " is not name,dir,type");
		}
		const auto second = line.find(',', first + 1);
		if (second == string::npos) {
			throw std::runtime_error("manifest line " + std::to_string(line_no) + " is not name,dir,type");
		}
		const string name = line.substr(0, first);
		if (name == "name") {
			continue; // header row
		}
		const string dir  = line.substr(first + 1, second - first - 1);
		const string type = line.substr(second + 1);
		const auto   parsed = parse_data_type(type);
		if (!parsed.has_value()) {
			throw std::runtime_error("manifest line " + std::to_string(line_no) + " has unknown type: " + type);
		}
		specs.push_back(DatasetSpec {name, dir, *parsed});
	}
	return specs;
}

vector<DatasetSpec> resolve_datasets(const vector<DatasetSpec>& fallback) {
	const char* env = std::getenv("FLS_SUBINTSPLIT_MANIFEST");
	return load_manifest(env == nullptr ? path {} : path {env}, fallback);
}

// $FLS_SUBINTSPLIT_SHARED_CODECS=1 holds both wizard arms (wizard_with_sis, wizard_without_sis) to the codec set shared
// by the cross-format comparison with Nimble and BtrBlocks: Uncompressed, Constant, Dictionary, RLE, FFOR_SLPATCH,
// Frequency and FFOR (plus SubIntSplit in wizard_with_sis). That is WizardLimited's set (see write_fls), so the same
// tokens go: RLE_slpatch, Delta and CrossRLE, here at every integer width Cast() can narrow a column to. SubIntSplit's
// own sections already choose from exactly this set (subintsplit_section_selector.hpp), so nothing changes there.
// Unset or any other value keeps the default, full wizard pool.
bool shared_codecs_enabled() {
	const char* env = std::getenv("FLS_SUBINTSPLIT_SHARED_CODECS");
	return env != nullptr && string {env} == "1";
}

void disable_non_shared_codecs(Connection& conn) {
	for (const OperatorToken token : {OperatorToken::EXP_RLE_I64_SLPATCH_U16,
	                                  OperatorToken::EXP_RLE_I32_SLPATCH_U16,
	                                  OperatorToken::EXP_RLE_I16_SLPATCH_U16,
	                                  OperatorToken::EXP_RLE_I08_SLPATCH_U16,
	                                  OperatorToken::EXP_DELTA_I64,
	                                  OperatorToken::EXP_DELTA_I32,
	                                  OperatorToken::EXP_DELTA_I16,
	                                  OperatorToken::EXP_DELTA_I08,
	                                  OperatorToken::EXP_DELTA_U08,
	                                  OperatorToken::EXP_CROSS_RLE_I64,
	                                  OperatorToken::EXP_CROSS_RLE_I32,
	                                  OperatorToken::EXP_CROSS_RLE_I16,
	                                  OperatorToken::EXP_CROSS_RLE_I08,
	                                  OperatorToken::EXP_CROSS_RLE_U08}) {
		conn.disable_encoding(token);
	}
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * Run metadata
 *
 * No result CSV in this repository recorded what machine produced it, which makes every timing here uncomparable to
 * any timing produced anywhere else -- including to the same benchmark run on the same machine a month later under a
 * different governor. Written as a sidecar rather than a column so the result schema is unchanged.
\*--------------------------------------------------------------------------------------------------------------------*/
string read_first_match(const path& file, const string& prefix) {
	std::ifstream in {file};
	string        line;
	while (std::getline(in, line)) {
		if (line.rfind(prefix, 0) == 0) {
			const auto colon = line.find(':');
			if (colon == string::npos) {
				return {};
			}
			auto value = line.substr(colon + 1);
			const auto first = value.find_first_not_of(" \t");
			return first == string::npos ? string {} : value.substr(first);
		}
	}
	return {};
}

void write_run_metadata(const path& out_path, const string& driver) {
	std::filesystem::create_directories(out_path.parent_path());
	std::ofstream out {out_path};

	const auto now   = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	char       stamp[32] {};
	std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%S", std::gmtime(&now));

	out << "key,value\n";
	out << "driver," << driver << "\n";
	out << "timestamp_utc," << stamp << "\n";
	out << "cpu_model," << read_first_match("/proc/cpuinfo", "model name") << "\n";
	out << "hardware_threads," << std::thread::hardware_concurrency() << "\n";
	{
		std::ifstream governor {"/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"};
		string        value;
		std::getline(governor, value);
		out << "scaling_governor," << (value.empty() ? "unknown" : value) << "\n";
	}
#if defined(__clang__)
	out << "compiler,clang " << __clang_major__ << "." << __clang_minor__ << "\n";
#elif defined(__GNUC__)
	out << "compiler,gcc " << __GNUC__ << "." << __GNUC_MINOR__ << "\n";
#else
	out << "compiler,unknown\n";
#endif
	out << "build_type," << FLS_BENCH_BUILD_TYPE << "\n";
	out << "reduction,median_of_5\n";
	out << "point_probes," << POINT_PROBES << "\n";
	out << "range_offsets_per_width,";
	for (n_t i {0}; i < RANGE_OFFSETS_PER_WIDTH.size(); ++i) {
		out << (i ? " " : "") << RANGE_OFFSETS_PER_WIDTH[i];
	}
	out << "\n";
	out << "seed," << SEED << "\n";
	out << "wizard_codec_set," << (shared_codecs_enabled() ? "shared" : "full") << "\n";
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
		conn.force_schema_pool(row_spec.tokens.empty() ? vector<OperatorToken> {row_spec.token} : row_spec.tokens);
		break;
	case Mode::Wizard:
		// the default candidate pool, i.e. what FastLanes would do on its own
		if (shared_codecs_enabled()) {
			disable_non_shared_codecs(conn);
		}
		break;
	case Mode::WizardNoSis:
		// Cast() can narrow the column, so both widths have to go or the wizard just picks the other one.
		conn.disable_encoding(OperatorToken::EXP_SUBINTSPLIT_I64);
		conn.disable_encoding(OperatorToken::EXP_SUBINTSPLIT_I32);
		if (shared_codecs_enabled()) {
			disable_non_shared_codecs(conn);
		}
		break;
	case Mode::WizardLimited:
		// Limited codec-set comparison: keep only {RLE, Dictionary, FFOR, FFOR_slpatch, Uncompressed, Frequency,
		// SubIntSplit} in the wizard's default pool by disabling the rest -- RLE_slpatch, Delta, CrossRLE. Cast() can
		// narrow the column, so both widths of each disabled token have to go, exactly as WizardNoSis does above.
		conn.disable_encoding(OperatorToken::EXP_RLE_I64_SLPATCH_U16);
		conn.disable_encoding(OperatorToken::EXP_RLE_I32_SLPATCH_U16);
		conn.disable_encoding(OperatorToken::EXP_DELTA_I64);
		conn.disable_encoding(OperatorToken::EXP_DELTA_I32);
		conn.disable_encoding(OperatorToken::EXP_CROSS_RLE_I64);
		conn.disable_encoding(OperatorToken::EXP_CROSS_RLE_I32);
		break;
	case Mode::WizardLimitedNoSis:
		// Same exclusions as WizardLimited, plus both SubIntSplit tokens.
		conn.disable_encoding(OperatorToken::EXP_RLE_I64_SLPATCH_U16);
		conn.disable_encoding(OperatorToken::EXP_RLE_I32_SLPATCH_U16);
		conn.disable_encoding(OperatorToken::EXP_DELTA_I64);
		conn.disable_encoding(OperatorToken::EXP_DELTA_I32);
		conn.disable_encoding(OperatorToken::EXP_CROSS_RLE_I64);
		conn.disable_encoding(OperatorToken::EXP_CROSS_RLE_I32);
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
// Returns whether every compared probe matched. A mismatch is reported and recorded (metric "validated" = 0) rather
// than thrown, so one wrong arm cannot hide every other arm's numbers; a missing read path or an empty comparison
// still throws, since then there is nothing to validate.
template <typename PT>
bool cross_check(const DatasetSpec&          spec,
                 const RowSpec&              row_spec,
                 vector<up<RowgroupReader>>& readers,
                 Harness<PT>&                harness,
                 const ProbePlan&            plan) {
	// One full materialize() per checked rowgroup, compared against the harness on several random triples. Cheap, and
	// it protects every number in the table -- above all delta, where reading the operator's buffer directly would be
	// wrong and fast.
	const n_t n_checked = std::min<n_t>(2, plan.layout.n_rowgroups);
	n_t       n_compared {0};
	n_t       n_mismatched {0};

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
				std::cout << "   !! VALIDATION FAILED: decode-then-index harness disagrees with materialize() for "
				          << spec.name << "/" << row_spec.label << " at rowgroup " << rg << " vector " << vec_idx
				          << " row " << row << ": expected " << static_cast<int64_t>(expected) << " got "
				          << static_cast<int64_t>(got) << std::endl;
				++n_mismatched;
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
	std::cout << "   (harness cross-checked against materialize() on " << n_compared << " probes, " << n_mismatched
	          << " mismatched)" << std::endl;
	return n_mismatched == 0;
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
			if (oprs[rg]->bit_starts != oprs[0]->bit_starts || oprs[rg]->section_tokens != oprs[0]->section_tokens) {
				varies = true;
			}
		}
		const n_t      n_sections = oprs[0]->bit_starts.size();
		vector<string> starts;
		for (n_t s {0}; s < n_sections; ++s) {
			starts.push_back(std::to_string(static_cast<n_t>(oprs[0]->bit_starts[s])));
		}
		// A section's bit width is only meaningful (and only reachable via bw_segment_views, which is
		// only populated for plain-FFOR sections now that a section can be any of {Uncompressed,
		// Constant, RLE, Dictionary, FFOR, FFOR_SLPATCH, FrequencyPartition}) when every rowgroup agrees
		// it's plain FFOR; otherwise report the token instead of dereferencing a null segment view.
		// Rowgroup 0's per-section codecs; other rowgroups may differ, flagged by varies_by_rowgroup like the starts.
		vector<string> codecs;
		for (n_t s {0}; s < n_sections; ++s) {
			codecs.push_back(token_to_string(oprs[0]->section_tokens[s]));
		}
		vector<string> bw_medians;
		for (n_t s {0}; s < min_sections; ++s) {
			const bool all_plain_ffor = std::all_of(
			    oprs.begin(), oprs.end(), [&](const sp<dec_subintsplit_opr<PT>>& opr) {
				    return opr->section_is_plain_ffor[s];
			    });
			if (!all_plain_ffor) {
				bw_medians.push_back("non_ffor");
				continue;
			}
			vector<n_t> widths;
			for (n_t rg {0}; rg < plan.layout.n_rowgroups; ++rg) {
				for (n_t vec_idx {0}; vec_idx < plan.layout.n_vec[rg]; ++vec_idx) {
					oprs[rg]->PointTo(vec_idx);
					widths.push_back(*reinterpret_cast<const bw_t*>(oprs[rg]->bw_segment_views[s]->data));
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
		       "starts=" + join(starts, ";") + "|bw_med=" + join(bw_medians, ";") + "|codecs=" + join(codecs, ";") +
		           (varies ? "|varies_by_rowgroup" : ""));

		// Native point access: position arithmetic, no vector decode.
		const double point_us = median_over(POINT_REPS, POINT_PROBES, [&] {
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
				const double us = median_over(GATHER_REPS, GATHER_BATCH, [&] {
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

		// Range sweep on SubIntSplit's own path.
		//
		// bench_range_paths() serves a range through the decode-then-index path,
		// which is the path every encoding has and therefore the comparable one.
		// But SubIntSplit is the one encoding here that can answer a range by
		// position arithmetic per element, without decoding the vector at all,
		// and reporting only the shared path would show it at less than its best
		// on the axis the whole comparison exists to measure. Both are recorded;
		// neither replaces the other.
		vector<n_t> first_row(plan.layout.n_rowgroups + 1, 0);
		for (n_t rg {0}; rg < plan.layout.n_rowgroups; ++rg) {
			first_row[rg + 1] = first_row[rg] + (plan.layout.n_vec[rg] * CFG::VEC_SZ);
		}
		for (n_t w {0}; w < RANGE_WIDTHS.size(); ++w) {
			const n_t width = RANGE_WIDTHS[w];
			if (width > plan.total_rows) {
				continue;
			}
			const n_t    offsets      = RANGE_OFFSETS_PER_WIDTH[w];
			const n_t*   range_starts = plan.range_start.data() + (w * RANGE_OFFSETS_MAX);
			const double us = median_over(RANGE_REPS, offsets, [&] {
				for (n_t o {0}; o < offsets; ++o) {
					const n_t begin = range_starts[o];
					const n_t end   = begin + width;
					n_t       rowgroup {0};
					while (first_row[rowgroup + 1] <= begin) {
						++rowgroup;
					}
					for (n_t row {begin}; row < end; ++row) {
						if (row >= first_row[rowgroup + 1]) {
							++rowgroup;
						}
						const n_t local = row - first_row[rowgroup];
						g_sink ^= static_cast<uint64_t>(oprs[rowgroup]->PointAccess(
						    local / CFG::VEC_SZ, static_cast<idx_t>(local % CFG::VEC_SZ)));
					}
				}
			});
			const string note = reps_note(RANGE_REPS, "offsets=" + std::to_string(offsets));
			record(spec.name, row_spec.group, row_spec.label, "range_read_pointwise", us, "us/range", width, note);
			record(spec.name,
			       row_spec.group,
			       row_spec.label,
			       "range_read_pointwise_per_element",
			       (us * 1000.0) / static_cast<double>(width),
			       "ns/element",
			       width,
			       note);
		}
	} else {
		(void)spec;
		(void)row_spec;
		(void)readers;
		(void)plan;
	}
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * Range reads
 *
 * Read B contiguous elements starting at a uniformly drawn global offset, for a log-spaced sweep of B. The range is
 * served by decoding only the vectors it actually covers -- located by arithmetic on the offset, never by scanning --
 * so a narrow range touches one vector and a wide one touches many.
 *
 * This is deliberately NOT "decode the column, then slice". That version would produce a number, and the number would
 * describe the bulk decode path with an extra memcpy, not the range path. Every encoding measured here has an
 * addressable interior at vector granularity, so every encoding gets a real answer; the differences between them are
 * differences in what one vector costs and in how many vectors the width forces.
 *
 * Reported both as us per range (what a query pays) and ns per element (what makes widths comparable to each other).
\*--------------------------------------------------------------------------------------------------------------------*/
template <typename PT>
void bench_range_paths(const DatasetSpec&          spec,
                       const RowSpec&              row_spec,
                       vector<up<RowgroupReader>>& readers,
                       const ProbePlan&            plan) {
	Harness<PT> harness {readers, plan.layout};

	// Global row index -> (rowgroup, vector, row), built once outside every timed region.
	vector<n_t> first_row(plan.layout.n_rowgroups + 1, 0);
	for (n_t rg {0}; rg < plan.layout.n_rowgroups; ++rg) {
		first_row[rg + 1] = first_row[rg] + (plan.layout.n_vec[rg] * CFG::VEC_SZ);
	}

	for (n_t w {0}; w < RANGE_WIDTHS.size(); ++w) {
		const n_t width = RANGE_WIDTHS[w];
		if (width > plan.total_rows) {
			continue;
		}
		const n_t  offsets = RANGE_OFFSETS_PER_WIDTH[w];
		const n_t* starts  = plan.range_start.data() + (w * RANGE_OFFSETS_MAX);

		const double us = median_over(RANGE_REPS, offsets, [&] {
			for (n_t o {0}; o < offsets; ++o) {
				const n_t begin = starts[o];
				const n_t end   = begin + width;

				n_t rowgroup {0};
				while (first_row[rowgroup + 1] <= begin) {
					++rowgroup;
				}
				n_t row = begin;
				while (row < end) {
					const n_t vec_idx  = (row - first_row[rowgroup]) / CFG::VEC_SZ;
					const n_t vec_base = first_row[rowgroup] + (vec_idx * CFG::VEC_SZ);
					const n_t stop     = std::min(end, vec_base + CFG::VEC_SZ);
					const auto access  = harness.decode(rowgroup, vec_idx);
					for (n_t i {row}; i < stop; ++i) {
						g_sink ^= static_cast<uint64_t>(access.at(static_cast<idx_t>(i - vec_base)));
					}
					row = stop;
					if (row >= first_row[rowgroup + 1]) {
						++rowgroup;
					}
				}
			}
		});

		const string note = reps_note(RANGE_REPS, "offsets=" + std::to_string(offsets));
		record(spec.name, row_spec.group, row_spec.label, "range_read", us, "us/range", width, note);
		record(spec.name,
		       row_spec.group,
		       row_spec.label,
		       "range_read_per_element",
		       (us * 1000.0) / static_cast<double>(width),
		       "ns/element",
		       width,
		       note);
	}
}

template <typename PT>
void bench_read_paths(const DatasetSpec&          spec,
                      const RowSpec&              row_spec,
                      vector<up<RowgroupReader>>& readers,
                      const ProbePlan&            plan) {
	Harness<PT> harness {readers, plan.layout};
	const bool  validated = cross_check<PT>(spec, row_spec, readers, harness, plan);
	record(spec.name, row_spec.group, row_spec.label, "validated", validated ? 1.0 : 0.0, "bool");

	// Bulk decode: every rowgroup, every vector.
	const double bulk_ms = median_over(BULK_REPS, plan.layout.n_rowgroups * 1000, [&] {
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
	const double point_us = median_over(POINT_REPS, POINT_PROBES, [&] {
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
		const double us = median_over(GATHER_REPS, GATHER_BATCH, [&] {
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

	bench_range_paths<PT>(spec, row_spec, readers, plan);
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

	// One unrepeated encode. Left unrepeated on purpose -- write_fls() writes a
	// file to disk and repeating it would measure the filesystem as much as the
	// encoder -- but labelled honestly, because an unrepeated number carries the
	// full spread of the machine. The same arm's encode time has been seen to
	// differ by 42% between two runs on identical data.
	record(spec.name, row_spec.group, row_spec.label, "encode_time", encode_ms, "ms", 0, "single_run=1;unrepeated");
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

} // namespace

#endif // FLS_BENCHMARK_BENCH_SUBINTSPLIT_COMMON_HPP
