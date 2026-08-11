// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// benchmark/bench_subintsplit/bench_subintsplit.cpp
// ────────────────────────────────────────────────────────
//
// Measures SubIntSplit against the other integer encodings on encode time, compressed size, and all three read paths.
//
// A note on comparability, because the three read paths are not equally fair:
//
//   bulk    comparable across every encoding: they all decode a vector at a time.
//   gather  comparable only via the "decoded" mode, which decodes the vector and gathers out of it -- what every other
//           encoding is limited to, and what Nimble's bulkScan does. The "pointwise" mode is a capability the others
//           do not have.
//   point   NOT comparable as a like-for-like speed test. Only SubIntSplit can reach a value by position arithmetic;
//           every other encoding must decode the whole 1024-value vector and index into it. The baselines below are
//           reported so the difference reads as a capability gap rather than an implementation win, and so the honest
//           comparison -- SubIntSplit against a single-section FFOR column read the same way -- is visible too.
//
// Output: benchmark/result/subintsplit/*.csv
//
#include "data/fastlanes_data.hpp"
#include "fastlanes.hpp"
#include "fls/connection.hpp"
#include "fls/expression/subintsplit_operator.hpp"
#include "fls/reader/rowgroup_reader.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

using namespace fastlanes; // NOLINT

namespace {

using clock_t_    = std::chrono::high_resolution_clock;
using millis_t    = std::chrono::duration<double, std::milli>;
using micros_t    = std::chrono::duration<double, std::micro>;
constexpr n_t REP = 20;

struct DatasetSpec {
	string        name;
	string        dir;
	DataType      type;
	n_t           raw_bytes;
	OperatorToken subintsplit_token;
};

struct Row {
	string dataset;
	string encoding;
	string metric;
	double value {0.0};
	string unit;
	n_t    detail {0};
};

vector<Row> g_rows;

void record(const string& dataset,
            const string& encoding,
            const string& metric,
            double        value,
            const string& unit,
            const n_t     detail = 0) {
	g_rows.push_back(Row {dataset, encoding, metric, value, unit, detail});
	std::cout << "   " << std::setw(18) << std::left << encoding << std::setw(22) << metric << std::setw(14)
	          << std::right << std::fixed << std::setprecision(3) << value << " " << unit;
	if (detail != 0) {
		std::cout << "   (n=" << detail << ")";
	}
	std::cout << std::endl;
}

path write_fls(const DatasetSpec& spec, const OperatorToken token, const path& out_dir, double& encode_ms) {
	std::filesystem::remove_all(out_dir);
	std::filesystem::create_directories(out_dir);

	Connection conn;
	conn.reset().force_schema_pool({token});
	conn.read_csv(path {spec.dir}); // parsing is excluded from the encode timing below

	const auto start = clock_t_::now();
	conn.to_fls(out_dir / "data.fls");
	encode_ms = millis_t {clock_t_::now() - start}.count();

	return out_dir / "data.fls";
}

double bench_bulk(const path& fls_path) {
	Connection conn;
	auto       reader          = conn.reset().read_fls(fls_path);
	auto       rowgroup_reader = reader->get_rowgroup_reader(0);
	const n_t  n_vec           = rowgroup_reader->get_descriptor().m_n_vec();

	const auto start = clock_t_::now();
	for (n_t rep {0}; rep < REP; ++rep) {
		for (n_t vec_idx {0}; vec_idx < n_vec; ++vec_idx) {
			rowgroup_reader->get_chunk(vec_idx);
		}
	}
	return millis_t {clock_t_::now() - start}.count() / static_cast<double>(REP);
}

// Point access as every encoding other than SubIntSplit must do it: decode the containing vector, then index.
double bench_point_via_bulk(const path& fls_path, const n_t n_probes) {
	Connection conn;
	auto       reader          = conn.reset().read_fls(fls_path);
	auto       rowgroup_reader = reader->get_rowgroup_reader(0);
	const n_t  n_vec           = rowgroup_reader->get_descriptor().m_n_vec();

	std::mt19937_64 rng {11};
	const auto      start = clock_t_::now();
	for (n_t probe {0}; probe < n_probes; ++probe) {
		rowgroup_reader->get_chunk(rng() % n_vec);
	}
	return micros_t {clock_t_::now() - start}.count() / static_cast<double>(n_probes);
}

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

template <typename PT>
void bench_subintsplit_paths(const DatasetSpec& spec, const path& fls_path) {
	Connection conn;
	auto       reader          = conn.reset().read_fls(fls_path);
	auto       rowgroup_reader = reader->get_rowgroup_reader(0);
	auto       opr             = find_operator<PT>(*rowgroup_reader);
	if (!opr) {
		std::cout << "   (no SubIntSplit operator found - skipping access paths)" << std::endl;
		return;
	}

	record(spec.name, "subintsplit", "sections", static_cast<double>(opr->bit_starts.size()), "count");

	const n_t       n_vec = rowgroup_reader->get_descriptor().m_n_vec();
	std::mt19937_64 rng {11};

	// Point access.
	constexpr n_t N_PROBES = 200000;
	vector<n_t>   probe_vec(N_PROBES);
	vector<n_t>   probe_idx(N_PROBES);
	for (n_t i {0}; i < N_PROBES; ++i) {
		probe_vec[i] = rng() % n_vec;
		probe_idx[i] = rng() % CFG::VEC_SZ;
	}
	PT         sink {0};
	const auto point_start = clock_t_::now();
	for (n_t i {0}; i < N_PROBES; ++i) {
		sink ^= opr->PointAccess(probe_vec[i], probe_idx[i]);
	}
	const double point_us = micros_t {clock_t_::now() - point_start}.count() / static_cast<double>(N_PROBES);
	record(spec.name, "subintsplit", "point_access", point_us, "us/probe");
	if (sink == std::numeric_limits<PT>::min()) {
		std::cout << "" << std::endl; // keep the accumulator observable so the loop is not optimised away
	}

	// Gather, swept over selectivity, both modes, so the crossover is measured rather than assumed.
	vector<idx_t> rows;
	vector<PT>    out;
	for (const n_t n : {n_t {1}, n_t {4}, n_t {16}, n_t {64}, n_t {256}, n_t {CFG::VEC_SZ}}) {
		rows.resize(n);
		out.resize(n);
		constexpr n_t GATHER_REPS = 2000;

		for (const bool pointwise : {true, false}) {
			const auto start = clock_t_::now();
			for (n_t rep {0}; rep < GATHER_REPS; ++rep) {
				const n_t vec_idx = rng() % n_vec;
				for (n_t i {0}; i < n; ++i) {
					rows[i] = static_cast<idx_t>(rng() % CFG::VEC_SZ);
				}
				if (pointwise) {
					opr->GatherPointwise(vec_idx, rows.data(), n, out.data());
				} else {
					opr->GatherDecoded(vec_idx, rows.data(), n, out.data());
				}
			}
			const double us = micros_t {clock_t_::now() - start}.count() / static_cast<double>(GATHER_REPS);
			record(spec.name, "subintsplit", pointwise ? "gather_pointwise" : "gather_decoded", us, "us/gather", n);
		}
	}
}

void run_dataset(const DatasetSpec& spec, const vector<std::pair<string, OperatorToken>>& candidates) {
	std::cout << "\n=== " << spec.name << "  (raw " << spec.raw_bytes << " B)" << std::endl;

	const path work_dir = std::filesystem::temp_directory_path() / ("bench_subintsplit_" + spec.name);

	for (const auto& [label, token] : candidates) {
		double     encode_ms {0.0};
		const path fls_path = write_fls(spec, token, work_dir, encode_ms);
		const auto size     = static_cast<double>(std::filesystem::file_size(fls_path));

		record(spec.name, label, "encode_time", encode_ms, "ms");
		record(spec.name, label, "compressed_size", size, "bytes");
		record(spec.name, label, "compression_ratio", static_cast<double>(spec.raw_bytes) / size, "x");
		record(spec.name, label, "bulk_decode", bench_bulk(fls_path), "ms/rowgroup");
		record(spec.name, label, "point_via_bulk", bench_point_via_bulk(fls_path, 20000), "us/probe");

		if (token == spec.subintsplit_token) {
			if (spec.type == DataType::INT64) {
				bench_subintsplit_paths<i64_pt>(spec, fls_path);
			} else {
				bench_subintsplit_paths<i32_pt>(spec, fls_path);
			}
		}
	}

	std::filesystem::remove_all(work_dir);
}

} // namespace

int main() {
	const string root = string {FLS_CMAKE_SOURCE_DIR};

	const vector<std::pair<string, OperatorToken>> i64_candidates = {
	    {"uncompressed", OperatorToken::EXP_UNCOMPRESSED_I64},
	    {"ffor", OperatorToken::EXP_FFOR_I64},
	    {"delta", OperatorToken::EXP_DELTA_I64},
	    {"ffor_slpatch", OperatorToken::EXP_FFOR_SLPATCH_I64},
	    {"subintsplit", OperatorToken::EXP_SUBINTSPLIT_I64},
	};
	const vector<std::pair<string, OperatorToken>> i32_candidates = {
	    {"uncompressed", OperatorToken::EXP_UNCOMPRESSED_I32},
	    {"ffor", OperatorToken::EXP_FFOR_I32},
	    {"delta", OperatorToken::EXP_DELTA_I32},
	    {"ffor_slpatch", OperatorToken::EXP_FFOR_SLPATCH_I32},
	    {"subintsplit", OperatorToken::EXP_SUBINTSPLIT_I32},
	};

	const vector<DatasetSpec> datasets = {
	    {"snowflake_i64",
	     root + "/data/generated/subintsplit/snowflake_i64",
	     DataType::INT64,
	     65536 * 8,
	     OperatorToken::EXP_SUBINTSPLIT_I64},
	    {"tpch_partkey_i32",
	     root + "/data/generated/subintsplit/tpch_partkey_i32",
	     DataType::INT32,
	     65536 * 4,
	     OperatorToken::EXP_SUBINTSPLIT_I32},
	    {"ipv4_i32",
	     root + "/data/generated/subintsplit/ipv4_i32",
	     DataType::INT32,
	     65536 * 4,
	     OperatorToken::EXP_SUBINTSPLIT_I32},
	};

	for (const auto& spec : datasets) {
		run_dataset(spec, spec.type == DataType::INT64 ? i64_candidates : i32_candidates);
	}

	const path result_path = path {root} / "benchmark" / "result" / "subintsplit" / "subintsplit.csv";
	std::filesystem::create_directories(result_path.parent_path());
	std::ofstream csv {result_path};
	csv << "dataset,encoding,metric,value,unit,detail\n";
	for (const auto& row : g_rows) {
		csv << row.dataset << "," << row.encoding << "," << row.metric << "," << std::fixed << std::setprecision(6)
		    << row.value << "," << row.unit << "," << row.detail << "\n";
	}
	std::cout << "\n-- results written to " << result_path << std::endl;

	return EXIT_SUCCESS;
}
