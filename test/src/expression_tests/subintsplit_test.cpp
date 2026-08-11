// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// test/src/expression_tests/subintsplit_test.cpp
// ────────────────────────────────────────────────────────
#include "fls/footer/column_descriptor_generated.h"
#include "fls/footer/operator_token_generated.h"
#include "fls/reader/rowgroup_reader.hpp"
#include "fls_tester.hpp"

namespace fastlanes {

// Passing the expression list forces the schema pool, so the wizard cannot quietly pick a different encoding and
// leave this test green without SubIntSplit ever having run.
TEST_F(FastLanesReaderTester, TEST_SUBINTSPLIT_I64) {
	TestCorrectness(GENERATED::SUBINTSPLIT_I64_EXPR, {OperatorToken::EXP_SUBINTSPLIT_I64});
}

// One rowgroup per vector: exercises the block-based header being written once per column against many small
// rowgroups, and the decoder recovering the section count from each of them.
TEST_F(FastLanesReaderTester, TEST_SUBINTSPLIT_I64_SMALL_ROWGROUPS) {
	TestCorrectness(GENERATED::SUBINTSPLIT_I64_EXPR, {OperatorToken::EXP_SUBINTSPLIT_I64}, 1);
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * Connection::disable_encoding removes a candidate from the pool the wizard searches, which is what lets a benchmark
 * compare FastLanes' own choice with and without SubIntSplit available while nothing else differs. The two runs below
 * are identical apart from that call.
\*--------------------------------------------------------------------------------------------------------------------*/
class SubIntSplitAblationTester : public ::testing::Test {
public:
	path fls_dir;

	void SetUp() override {
		fls_dir = path {FLS_CMAKE_SOURCE_DIR} / "data" / "fls" / ("ablation_" + std::to_string(getpid()));
		std::filesystem::create_directories(fls_dir);
	}
	void TearDown() override {
		std::filesystem::remove_all(fls_dir);
	}

	// Encodes `dataset` with a plain default connection (no forced pool), optionally disabling SubIntSplit first, and
	// returns the operator token the wizard chose for the first column. Also asserts the file round-trips.
	OperatorToken encode_and_read_back_first_token(const string_view dataset,
	                                               const bool        disable_subintsplit,
	                                               const string&     file_name) {
		const path fls_file_path = fls_dir / file_name;

		Connection write_conn;
		write_conn.reset();
		if (disable_subintsplit) {
			// Rowgroup::Cast() may narrow the column, so both widths have to go.
			write_conn.disable_encoding(OperatorToken::EXP_SUBINTSPLIT_I64)
			    .disable_encoding(OperatorToken::EXP_SUBINTSPLIT_I32);
		}
		write_conn.read_csv(path {string(dataset)});
		const auto& original_table = write_conn.get_table();
		write_conn.to_fls(fls_file_path);

		Connection read_conn;
		auto       reader          = read_conn.reset().read_fls(fls_file_path);
		auto       rowgroup_reader = reader->get_rowgroup_reader(0);

		const auto* column_descriptors = rowgroup_reader->get_descriptor().m_column_descriptors();
		EXPECT_NE(column_descriptors, nullptr);
		EXPECT_GT(column_descriptors->size(), 0);

		const auto* column_descriptor = column_descriptors->Get(0);
		const auto* operator_tokens   = column_descriptor->encoding_rpn()->operator_tokens();
		EXPECT_NE(operator_tokens, nullptr);
		EXPECT_GT(operator_tokens->size(), 0);

		const auto chosen_token = static_cast<OperatorToken>(operator_tokens->Get(0));

		// The file has to stay readable whichever encoding was chosen.
		const auto decoded_table = reader->materialize();
		const auto result        = (original_table == *decoded_table);
		EXPECT_TRUE(result.is_equal) << "Rowgroups differ. First not matching column index: "
		                             << result.first_failed_column_idx << " description: " << result.description;

		return chosen_token;
	}
};

TEST_F(SubIntSplitAblationTester, DISABLE_ENCODING_CHANGES_WIZARD_CHOICE) {
	const auto with_sis = encode_and_read_back_first_token(GENERATED::SUBINTSPLIT_SNOWFLAKE_I64, false, "with_sis.fls");
	ASSERT_EQ(with_sis, OperatorToken::EXP_SUBINTSPLIT_I64)
	    << "wizard did not pick SubIntSplit by default, it picked " << EnumNameOperatorToken(with_sis);

	const auto without_sis =
	    encode_and_read_back_first_token(GENERATED::SUBINTSPLIT_SNOWFLAKE_I64, true, "without_sis.fls");
	ASSERT_NE(without_sis, OperatorToken::EXP_SUBINTSPLIT_I64);
	ASSERT_NE(without_sis, OperatorToken::EXP_SUBINTSPLIT_I32);
	ASSERT_NE(without_sis, with_sis);
}

// Disabling is idempotent, and enable_all_encodings puts the candidate back.
TEST_F(SubIntSplitAblationTester, DISABLE_ENCODING_IS_IDEMPOTENT_AND_REVERSIBLE) {
	Connection con;
	ASSERT_FALSE(con.is_encoding_disabled(OperatorToken::EXP_SUBINTSPLIT_I64));
	ASSERT_TRUE(con.get_disabled_encodings().empty());

	con.disable_encoding(OperatorToken::EXP_SUBINTSPLIT_I64).disable_encoding(OperatorToken::EXP_SUBINTSPLIT_I64);
	ASSERT_TRUE(con.is_encoding_disabled(OperatorToken::EXP_SUBINTSPLIT_I64));
	ASSERT_EQ(con.get_disabled_encodings().size(), 1);

	con.enable_all_encodings();
	ASSERT_FALSE(con.is_encoding_disabled(OperatorToken::EXP_SUBINTSPLIT_I64));
	ASSERT_TRUE(con.get_disabled_encodings().empty());
}

} // namespace fastlanes
