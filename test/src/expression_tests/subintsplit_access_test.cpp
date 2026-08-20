// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// test/src/expression_tests/subintsplit_access_test.cpp
// ────────────────────────────────────────────────────────
#include "data/fastlanes_data.hpp"
#include "fastlanes.hpp"
#include "fls/connection.hpp"
#include "fls/expression/subintsplit_operator.hpp"
#include "fls/reader/rowgroup_reader.hpp"
#include "gtest/gtest.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>

#if defined(_WIN32)
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

namespace fastlanes {

/*--------------------------------------------------------------------------------------------------------------------*\
 * Point and gather access must agree, value for value, with a full Decode of the same vector. Decode is already covered
 * by the round-trip test, so it is the trustworthy reference here.
\*--------------------------------------------------------------------------------------------------------------------*/
class SubIntSplitAccessTester : public ::testing::Test {
public:
	path fls_dir;

	void SetUp() override {
		fls_dir = path {FLS_CMAKE_SOURCE_DIR} / "data" / "fls" / ("access_" + std::to_string(getpid()));
		std::filesystem::create_directories(fls_dir);
	}
	void TearDown() override {
		std::filesystem::remove_all(fls_dir);
	}

	template <typename PT>
	void check_access(const string_view dataset, const OperatorToken token) {
		Connection write_conn;
		write_conn.reset().force_schema_pool({token});
		write_conn.read_csv(path {string(dataset)});
		write_conn.to_fls(fls_dir / "data.fls");

		Connection read_conn;
		auto       reader          = read_conn.reset().read_fls(fls_dir / "data.fls");
		auto       rowgroup_reader = reader->get_rowgroup_reader(0);

		// Forcing the schema pool means the column must have been encoded with SubIntSplit; if this operator is not
		// here, the test would otherwise pass without exercising anything.
		sp<dec_subintsplit_opr<PT>> opr;
		for (auto& expression : rowgroup_reader->m_expressions) {
			for (auto& physical_operator : expression->operators) {
				visit_dec(
				    [&](const auto& candidate) {
					    using candidate_t = std::decay_t<decltype(candidate)>;
					    if constexpr (std::is_same_v<candidate_t, sp<dec_subintsplit_opr<PT>>>) {
						    opr = candidate;
					    }
				    },
				    physical_operator);
			}
		}
		ASSERT_NE(opr, nullptr) << "column was not encoded with SubIntSplit";
		ASSERT_FALSE(opr->bit_starts.empty());

		const n_t       n_vec = rowgroup_reader->get_descriptor().m_n_vec();
		std::mt19937_64 rng {7};
		vector<idx_t>   rows;
		vector<PT>      gathered;

		for (n_t vec_idx {0}; vec_idx < n_vec; ++vec_idx) {
			opr->Decode(vec_idx);
			const vector<PT> reference {opr->data, opr->data + CFG::VEC_SZ};

			for (n_t probe {0}; probe < 64; ++probe) {
				const n_t idx = rng() % CFG::VEC_SZ;
				ASSERT_EQ(opr->PointAccess(vec_idx, idx), reference[idx])
				    << "point access disagrees at vector " << vec_idx << " index " << idx;
			}

			// Straddles the default gather_decode_threshold in both directions, so Gather() is exercised on both
			// sides of its crossover as well as each mode directly.
			for (const n_t n : {n_t {1}, n_t {7}, n_t {63}, n_t {200}, n_t {CFG::VEC_SZ}}) {
				rows.clear();
				for (n_t i {0}; i < n; ++i) {
					rows.push_back(static_cast<idx_t>(rng() % CFG::VEC_SZ));
				}

				gathered.assign(n, PT {0});
				opr->GatherPointwise(vec_idx, rows.data(), n, gathered.data());
				for (n_t i {0}; i < n; ++i) {
					ASSERT_EQ(gathered[i], reference[rows[i]]) << "GatherPointwise disagrees, n = " << n;
				}

				gathered.assign(n, PT {0});
				opr->GatherDecoded(vec_idx, rows.data(), n, gathered.data());
				for (n_t i {0}; i < n; ++i) {
					ASSERT_EQ(gathered[i], reference[rows[i]]) << "GatherDecoded disagrees, n = " << n;
				}

				gathered.assign(n, PT {0});
				opr->Gather(vec_idx, rows.data(), n, gathered.data());
				for (n_t i {0}; i < n; ++i) {
					ASSERT_EQ(gathered[i], reference[rows[i]]) << "Gather disagrees, n = " << n;
				}
			}
		}
	}
};

// Splits into three sections, so the multi-section OR-accumulation is what gets exercised.
TEST_F(SubIntSplitAccessTester, ACCESS_PATHS_I64) {
	check_access<i64_pt>(GENERATED::SUBINTSPLIT_SNOWFLAKE_I64, OperatorToken::EXP_SUBINTSPLIT_I64);
}

// Signed 32-bit with negative values, and a single-section plan: the degenerate shape, where SubIntSplit collapses to
// plain FFOR and the access paths must still be exact.
TEST_F(SubIntSplitAccessTester, ACCESS_PATHS_I32_NEGATIVE_VALUES) {
	check_access<i32_pt>(GENERATED::SUBINTSPLIT_IPV4_I32, OperatorToken::EXP_SUBINTSPLIT_I32);
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * Per-section codec selection: the tests above both exercise real datasets where every section happens to land on
 * FFOR, same as before per-section selection existed -- they prove the plumbing didn't regress, not that a non-FFOR
 * section round-trips correctly. This builds a dataset engineered to force one, and checks whatever it actually
 * gets (rather than asserting one specific codec) -- which one wins depends on the DP's own split choice, not just
 * this dataset's shape; see write_repeated_run_section_dataset's comment for how that played out here.
\*--------------------------------------------------------------------------------------------------------------------*/

// A snowflake-shaped id where the top field only changes every 64 rows, i.e. genuinely repeats -- not merely
// low-cardinality the way the existing snowflake fixture's cycling machine-id field is (which the DP folds into
// the low section for free, since a constant offset drops out of bit_width(max - min) entirely; confirmed by
// inspecting the actual chosen plan while writing this test). Real, same-value *runs* are what RLE needs and
// FFOR can't exploit, so the DP -- which still only ever measures FFOR width when deciding where to split -- ends
// up isolating this field into its own section anyway (a lower bit_width for the top section on its own than
// folded into the bottom one), and once isolated select_section_encoding picks EXP_RLE_I64_U16 for it on real
// measured cost, not FFOR.
void write_repeated_run_section_dataset(const path& dataset_dir, const n_t n_rows) {
	std::filesystem::create_directories(dataset_dir);

	std::ofstream schema {dataset_dir / "schema.json"};
	schema << R"({"columns": [{"name": "COLUMN_0", "type": "FLS_I64"}]})";
	schema.close();

	std::ofstream                           csv {dataset_dir / "generated.csv"};
	std::mt19937_64                         rng {123};
	std::uniform_int_distribution<uint64_t> seq_d {0, 4095}; // low 22 bits available, varies every row
	uint64_t                                timestamp {1};   // top bits: repeats for 64 rows, then advances
	for (n_t i {0}; i < n_rows; ++i) {
		if (i % 64 == 0) {
			++timestamp;
		}
		const uint64_t value = (timestamp << 22) | seq_d(rng);
		csv << static_cast<int64_t>(value) << "\n";
	}
}

TEST_F(SubIntSplitAccessTester, ACCESS_PATHS_I64_NON_FFOR_SECTION) {
	const path dataset_dir = fls_dir / "csv_src";
	write_repeated_run_section_dataset(dataset_dir, 8192);

	check_access<i64_pt>(dataset_dir.string(), OperatorToken::EXP_SUBINTSPLIT_I64);

	// check_access already proved point/gather agree with Decode() on this file; separately confirm the file it
	// wrote actually contains a non-FFOR section, or this test would pass without exercising anything new.
	Connection read_conn;
	auto       reader          = read_conn.reset().read_fls(fls_dir / "data.fls");
	auto       rowgroup_reader = reader->get_rowgroup_reader(0);

	sp<dec_subintsplit_opr<i64_pt>> opr;
	for (auto& expression : rowgroup_reader->m_expressions) {
		for (auto& physical_operator : expression->operators) {
			visit_dec(
			    [&](const auto& candidate) {
				    using candidate_t = std::decay_t<decltype(candidate)>;
				    if constexpr (std::is_same_v<candidate_t, sp<dec_subintsplit_opr<i64_pt>>>) {
					    opr = candidate;
				    }
			    },
			    physical_operator);
		}
	}
	ASSERT_NE(opr, nullptr);

	std::string plan_description;
	for (n_t s {0}; s < opr->bit_starts.size(); ++s) {
		plan_description += "[bit_start=" + std::to_string(opr->bit_starts[s]) +
		                    " token=" + std::to_string(static_cast<int>(opr->section_tokens[s])) + "] ";
	}

	const bool has_non_ffor_section =
	    std::any_of(opr->section_tokens.begin(), opr->section_tokens.end(), [](const OperatorToken token) {
		    return token != OperatorToken::EXP_FFOR_I64;
	    });
	EXPECT_TRUE(has_non_ffor_section) << "expected at least one non-FFOR section (e.g. EXP_CONSTANT_I64 for the "
	                                     "constant middle bit range), got all FFOR. Plan: "
	                                  << plan_description;
}

} // namespace fastlanes
