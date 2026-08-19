// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// benchmark/bench_subintsplit_limited/bench_subintsplit_limited.cpp
// ────────────────────────────────────────────────────────
//
// Measures SubIntSplit against a LIMITED codec set -- {uncompressed, RLE, Dictionary, FFOR, Frequency} -- rather than
// the full wizard pool bench_subintsplit.cpp compares against. This isolates SubIntSplit's value against the codecs a
// simpler / more conservative encoder would actually ship, and against FastLanes' own wizard restricted to the same
// limited pool, with and without SubIntSplit in it.
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
// Output: benchmark/result/subintsplit/subintsplit_limited.csv
//
#include "../bench_subintsplit/BenchSubIntSplitCommon.hpp"

namespace {

vector<RowSpec> make_row_specs(const DataType type) {
	// Constant is excluded: force_schema_pool cannot select EXP_CONSTANT_*, which bypasses the wizard's
	// poolable-candidate path entirely (see docs/subintsplit.md).
	const bool is_64 = type == DataType::INT64;
	return {
	    {"baseline",
	     "uncompressed",
	     Mode::Forced,
	     is_64 ? OperatorToken::EXP_UNCOMPRESSED_I64 : OperatorToken::EXP_UNCOMPRESSED_I32},
	    {"codec", "rle", Mode::Forced, is_64 ? OperatorToken::EXP_RLE_I64_U16 : OperatorToken::EXP_RLE_I32_U16},
	    // Dictionary's index width depends on the column's cardinality, not its own value width, and the wizard picks
	    // it dynamically (src/wizard/wizard.cpp's get_dict_encoding_pool, keyed on the estimated index type). Forcing
	    // one width would be a guess, so all three FFOR-packed width variants are offered as the candidate pool and
	    // evaluate_expressions() (src/wizard/wizard.cpp) picks the cheapest -- the honest "best Dictionary variant"
	    // comparison. force_schema_pool() replaces the pool wholesale, so passing several tokens here is a genuine
	    // multi-candidate forced search, not a single forced token.
	    {"codec",
	     "dict",
	     Mode::Forced,
	     OperatorToken::INVALID,
	     is_64 ? vector<OperatorToken> {OperatorToken::EXP_DICT_I64_FFOR_U32,
	                                     OperatorToken::EXP_DICT_I64_FFOR_U16,
	                                     OperatorToken::EXP_DICT_I64_FFOR_U08}
	           : vector<OperatorToken> {OperatorToken::EXP_DICT_I32_FFOR_U32,
	                                     OperatorToken::EXP_DICT_I32_FFOR_U16,
	                                     OperatorToken::EXP_DICT_I32_FFOR_U08}},
	    {"codec", "ffor", Mode::Forced, is_64 ? OperatorToken::EXP_FFOR_I64 : OperatorToken::EXP_FFOR_I32},
	    {"codec",
	     "frequency",
	     Mode::Forced,
	     is_64 ? OperatorToken::EXP_FREQUENCY_I64 : OperatorToken::EXP_FREQUENCY_I32},
	    {"codec",
	     "subintsplit",
	     Mode::Forced,
	     is_64 ? OperatorToken::EXP_SUBINTSPLIT_I64 : OperatorToken::EXP_SUBINTSPLIT_I32},
	    {"wizard", "wizard_limited_with_sis", Mode::WizardLimited},
	    {"wizard", "wizard_limited_without_sis", Mode::WizardLimitedNoSis},
	};
}

} // namespace

int main() {
	const char*  env_root = std::getenv("FLS_SUBINTSPLIT_DATA_DIR");
	const string data_root =
	    env_root != nullptr ? string {env_root} : string {FLS_CMAKE_SOURCE_DIR} + "/data/generated/subintsplit";
	std::cout << "dataset root: " << data_root << std::endl;

	vector<DatasetSpec> datasets = {
	    // Real Twitter snowflake IDs, not simulated. This is the headline dataset. Optional: only
	    // present when extract_real_snowflake.py has been run, since it needs the
	    // EncodingsPlayground parquet.
	    {"snowflake_i64_real", data_root + "/snowflake_i64_real", DataType::INT64},
	    // Synthetic snowflake IDs (same field layout, simulated). Secondary reference dataset,
	    // kept for comparison against the real one above.
	    {"snowflake_i64", data_root + "/snowflake_i64", DataType::INT64},
	    {"tpch_partkey_i32", data_root + "/tpch_partkey_i32", DataType::INT32},
	    {"ipv4_i32", data_root + "/ipv4_i32", DataType::INT32},
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
	    path {string {FLS_CMAKE_SOURCE_DIR}} / "benchmark" / "result" / "subintsplit" / "subintsplit_limited.csv";
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
