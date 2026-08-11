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
#include <filesystem>
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

} // namespace fastlanes
