// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// test/src/unit_tests/subintsplit_section_selector_test.cpp
// ────────────────────────────────────────────────────────
#include "fls/expression/subintsplit_section_selector.hpp"
#include "fls/footer/operator_token_generated.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {

using fastlanes::i32_pt;
using fastlanes::i64_pt;
using fastlanes::n_t;
using fastlanes::OperatorToken;
using fastlanes::subintsplit::select_section_encoding;

constexpr n_t VEC_SZ = 1024;

// A genuinely constant section: same value in every row of every vector.
TEST(SubIntSplitSectionSelector, PicksConstant) {
	const n_t          n_vec = 4;
	std::vector<i64_pt> values(n_vec * VEC_SZ, 42);

	const OperatorToken chosen = select_section_encoding<i64_pt>(values, n_vec);
	EXPECT_EQ(chosen, OperatorToken::EXP_CONSTANT_I64);
}

TEST(SubIntSplitSectionSelector, PicksConstantI32) {
	const n_t          n_vec = 2;
	std::vector<i32_pt> values(n_vec * VEC_SZ, -7);

	const OperatorToken chosen = select_section_encoding<i32_pt>(values, n_vec);
	EXPECT_EQ(chosen, OperatorToken::EXP_CONSTANT_I32);
}

// Long runs within each vector: RLE should beat FFOR/Dictionary/Frequency here.
TEST(SubIntSplitSectionSelector, PicksRLEOnLongRuns) {
	const n_t           n_vec = 4;
	std::vector<i64_pt> values;
	values.reserve(n_vec * VEC_SZ);
	for (n_t v {0}; v < n_vec; ++v) {
		// Runs of 128 identical values sweeping through a small range -- RLE-friendly, and not
		// low-enough cardinality per-vector to look attractive to Dictionary/Frequency the way a
		// true few-values-repeated-everywhere column would.
		for (n_t run {0}; run < VEC_SZ / 128; ++run) {
			for (n_t i {0}; i < 128; ++i) {
				values.push_back(static_cast<i64_pt>(run));
			}
		}
	}

	const OperatorToken chosen = select_section_encoding<i64_pt>(values, n_vec);
	EXPECT_EQ(chosen, OperatorToken::EXP_RLE_I64_U16);
}

// A handful of distinct values, each repeated across the whole column, in an order that defeats
// RLE (no long same-value runs) but is very cheap for Dictionary.
TEST(SubIntSplitSectionSelector, PicksDictionaryOnLowCardinalityShuffled) {
	const n_t           n_vec = 8;
	std::vector<i64_pt> values;
	values.reserve(n_vec * VEC_SZ);
	std::mt19937_64                      rng {7};
	constexpr int64_t                    distinct[16] = {10, 20, 30, 40, 50, 60, 70, 80,
	                                                     90, 100, 110, 120, 130, 140, 150, 160};
	std::uniform_int_distribution<int>   pick {0, 15};
	for (n_t i {0}; i < n_vec * VEC_SZ; ++i) {
		values.push_back(distinct[pick(rng)]);
	}

	const OperatorToken chosen = select_section_encoding<i64_pt>(values, n_vec);
	EXPECT_TRUE(chosen == OperatorToken::EXP_DICT_I64_FFOR_U08 || chosen == OperatorToken::EXP_DICT_I64_FFOR_U16 ||
	            chosen == OperatorToken::EXP_DICT_I64_FFOR_U32)
	    << "expected some Dictionary width, got token " << static_cast<int>(chosen);
}

// One dominant value with rare exceptions scattered in: Frequency's exception-list design should
// win over Dictionary/RLE/FFOR here.
TEST(SubIntSplitSectionSelector, PicksFrequencyOnMostlyOneValue) {
	const n_t           n_vec = 8;
	std::vector<i64_pt> values(n_vec * VEC_SZ, 5);
	std::mt19937_64                    rng {3};
	std::uniform_int_distribution<n_t> pos {0, n_vec * VEC_SZ - 1};
	std::uniform_int_distribution<int> exc {1000, 2000};
	// ~1% exceptions per vector, scattered.
	for (n_t i {0}; i < (n_vec * VEC_SZ) / 100; ++i) {
		values[pos(rng)] = exc(rng);
	}

	const OperatorToken chosen = select_section_encoding<i64_pt>(values, n_vec);
	EXPECT_EQ(chosen, OperatorToken::EXP_FREQUENCY_I64);
}

// High-entropy, wide-range data with no exploitable structure: plain FFOR should win (cheapest
// per-value overhead once nothing else applies).
TEST(SubIntSplitSectionSelector, PicksFFOROnHighEntropyData) {
	const n_t           n_vec = 8;
	std::vector<i64_pt> values;
	values.reserve(n_vec * VEC_SZ);
	std::mt19937_64                        rng {99};
	std::uniform_int_distribution<int64_t> dist {0, (1LL << 40) - 1};
	for (n_t i {0}; i < n_vec * VEC_SZ; ++i) {
		values.push_back(dist(rng));
	}

	const OperatorToken chosen = select_section_encoding<i64_pt>(values, n_vec);
	EXPECT_EQ(chosen, OperatorToken::EXP_FFOR_I64);
}

} // namespace
