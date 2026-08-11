// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// test/src/expression_tests/subintsplit_test.cpp
// ────────────────────────────────────────────────────────
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

} // namespace fastlanes
