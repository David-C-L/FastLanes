// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// test/src/unit_tests/subintsplit_selector_test.cpp
// ────────────────────────────────────────────────────────
#include "fls/expression/subintsplit_selector.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {

using fastlanes::n_t;
using fastlanes::span;
using fastlanes::subintsplit::select_splits;
using fastlanes::subintsplit::SubIntSplitPlan;
using fastlanes::subintsplit::SubIntSplitSelectorConfig;

constexpr n_t VEC_SZ = 1024;

template <typename PT>
SubIntSplitPlan run(const std::vector<PT>& values, const n_t n_vectors_total, const SubIntSplitSelectorConfig& cfg) {
	return select_splits<PT>(span<const PT> {values.data(), values.size()}, n_vectors_total, cfg);
}

template <typename PT>
SubIntSplitPlan run(const std::vector<PT>& values, const n_t n_vectors_total) {
	return run<PT>(values, n_vectors_total, SubIntSplitSelectorConfig {});
}

// Estimated size of the do-nothing plan: a single section covering the whole physical type.
template <typename PT>
double single_section_bits(const std::vector<PT>& values, const n_t n_vectors_total) {
	SubIntSplitSelectorConfig cfg;
	cfg.max_sections = 1;
	return run<PT>(values, n_vectors_total, cfg).estimated_bits;
}

// Synthetic Twitter-snowflake ids: 41-bit timestamp | 10-bit machine id | 12-bit sequence.
// The timestamp advances slowly, the machine id comes from a small set, the sequence churns over its full range.
std::vector<uint64_t> make_snowflake(const size_t n) {
	std::mt19937_64                         rng {42};
	std::uniform_int_distribution<uint64_t> seq_d {0, 4095};
	constexpr uint64_t                      machines[4] = {3, 7, 9, 11};

	std::vector<uint64_t> out;
	out.reserve(n);
	uint64_t timestamp = 1'700'000'000'000ULL & ((1ULL << 41) - 1);
	for (size_t i {0}; i < n; ++i) {
		if ((i % 64) == 0) {
			++timestamp;
		}
		out.push_back((timestamp << 22) | (machines[i % 4] << 12) | seq_d(rng));
	}
	return out;
}

// Set of interior boundaries of a plan, i.e. the bit positions where a new section starts.
std::vector<int> boundaries_of(const SubIntSplitPlan& plan) {
	std::vector<int> out;
	for (size_t i {1}; i < plan.segments.size(); ++i) {
		out.push_back(static_cast<int>(plan.segments[i].bit_start));
	}
	return out;
}

bool has_boundary_near(const SubIntSplitPlan& plan, const int expected, const int tolerance) {
	for (const int b : boundaries_of(plan)) {
		if (std::abs(b - expected) <= tolerance) {
			return true;
		}
	}
	return false;
}

void expect_covers_all_bits(const SubIntSplitPlan& plan, const int k_bits) {
	ASSERT_FALSE(plan.segments.empty());
	EXPECT_EQ(plan.segments.front().bit_start, 0);
	EXPECT_EQ(plan.segments.back().bit_end, k_bits - 1);
	for (size_t i {1}; i < plan.segments.size(); ++i) {
		EXPECT_EQ(plan.segments[i].bit_start, plan.segments[i - 1].bit_end + 1) << "sections must be contiguous";
	}
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * 1. snowflake ids: the field boundaries must be discovered, and splitting must pay off.
\*--------------------------------------------------------------------------------------------------------------------*/
TEST(SubIntSplitSelector, SnowflakeFindsFieldBoundaries) {
	const std::vector<uint64_t> data = make_snowflake(VEC_SZ * 16);

	const SubIntSplitPlan plan = run<uint64_t>(data, 16);
	expect_covers_all_bits(plan, 64);

	EXPECT_GE(plan.segments.size(), 3U) << "timestamp | machine | sequence should give three sections";
	EXPECT_TRUE(has_boundary_near(plan, 12, 2)) << "expected a boundary at/near the sequence field edge";
	EXPECT_TRUE(has_boundary_near(plan, 22, 2)) << "expected a boundary at/near the timestamp field edge";

	// Note the margin is bounded by what FFOR already achieves on the unsplit value: FFOR subtracts a per-vector base,
	// so the single section already costs only 22 + bit_width(timestamp delta) bits rather than the full 64.
	const double single               = single_section_bits<uint64_t>(data, 16);
	const double saved_bits_per_value = (single - plan.estimated_bits) / static_cast<double>(VEC_SZ * 16);
	EXPECT_GT(single, 1.2 * plan.estimated_bits);
	EXPECT_GT(saved_bits_per_value, 5.0);
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * 2. uniform random: no structure to exploit, so any split is pure overhead.
\*--------------------------------------------------------------------------------------------------------------------*/
TEST(SubIntSplitSelector, UniformRandomDoesNotOverSplit) {
	std::mt19937_64       rng {7};
	std::vector<uint64_t> data(VEC_SZ * 8);
	for (auto& value : data) {
		value = rng();
	}

	const SubIntSplitPlan plan = run<uint64_t>(data, 8);
	expect_covers_all_bits(plan, 64);
	EXPECT_EQ(plan.segments.size(), 1U) << "uniform random data must stay in a single section";
	EXPECT_NEAR(plan.estimated_bits / static_cast<double>(VEC_SZ * 8), 64.0, 0.5);
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * 3. constant column: bit width 0 everywhere, so the plan should be one nearly free section.
\*--------------------------------------------------------------------------------------------------------------------*/
TEST(SubIntSplitSelector, ConstantColumnIsCheapAndUnsplit) {
	const std::vector<uint64_t> data(VEC_SZ * 4, 0xDEADBEEFCAFEULL);

	const SubIntSplitPlan plan = run<uint64_t>(data, 4);
	expect_covers_all_bits(plan, 64);
	EXPECT_EQ(plan.segments.size(), 1U);
	EXPECT_LT(plan.estimated_bits / static_cast<double>(VEC_SZ * 4), 0.2) << "only per-vector bases remain";
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * 4. narrow low range in a wide column: the constant high bits must cost nothing.
 *
 * Deviation from the task brief worth spelling out: the brief expected the selector to *split off* the constant high
 * bits. Under the FastLanes cost oracle it correctly does not, because FFOR subtracts a per-vector base and therefore
 * already packs 0..255 in a 64-bit column at 8 bits/value in one section. Splitting would add a second section's
 * segments and per-vector base for exactly zero saving. What the test asserts is the property that actually matters:
 * the high bits are free.
\*--------------------------------------------------------------------------------------------------------------------*/
TEST(SubIntSplitSelector, NarrowLowRangeCostsOnlyItsOwnBits) {
	std::mt19937_64       rng {11};
	std::vector<uint64_t> data(VEC_SZ * 4);
	for (auto& value : data) {
		value = rng() & 0xFF;
	}

	const SubIntSplitPlan plan = run<uint64_t>(data, 4);
	expect_covers_all_bits(plan, 64);
	EXPECT_LT(plan.estimated_bits / static_cast<double>(VEC_SZ * 4), 9.0) << "the 56 constant high bits are free";
	EXPECT_LE(plan.segments.size(), 2U) << "no reason to fragment a column FFOR already handles";
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * 5. determinism.
\*--------------------------------------------------------------------------------------------------------------------*/
TEST(SubIntSplitSelector, IsDeterministic) {
	const std::vector<uint64_t> data = make_snowflake(VEC_SZ * 4);

	const SubIntSplitPlan a = run<uint64_t>(data, 4);
	const SubIntSplitPlan b = run<uint64_t>(data, 4);

	ASSERT_EQ(a.segments.size(), b.segments.size());
	EXPECT_DOUBLE_EQ(a.estimated_bits, b.estimated_bits);
	for (size_t i {0}; i < a.segments.size(); ++i) {
		EXPECT_EQ(a.segments[i].bit_start, b.segments[i].bit_start);
		EXPECT_EQ(a.segments[i].bit_end, b.segments[i].bit_end);
	}
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * 6. boundary conditions.
\*--------------------------------------------------------------------------------------------------------------------*/
TEST(SubIntSplitSelector, EmptyInputYieldsDegenerateFullWidthPlan) {
	const std::vector<uint64_t> data;

	const SubIntSplitPlan plan = run<uint64_t>(data, 0);
	ASSERT_EQ(plan.segments.size(), 1U);
	EXPECT_EQ(plan.segments[0].bit_start, 0);
	EXPECT_EQ(plan.segments[0].bit_end, 63);
	EXPECT_DOUBLE_EQ(plan.estimated_bits, 0.0);
}

TEST(SubIntSplitSelector, FewerValuesThanOneVector) {
	const std::vector<uint64_t> data = make_snowflake(300);

	const SubIntSplitPlan plan = run<uint64_t>(data, 1);
	expect_covers_all_bits(plan, 64);
	EXPECT_GE(plan.segments.size(), 2U) << "structure is visible even in a partial vector";
}

TEST(SubIntSplitSelector, ExactlyOneVector) {
	const std::vector<uint64_t> data = make_snowflake(VEC_SZ);

	const SubIntSplitPlan plan = run<uint64_t>(data, 1);
	expect_covers_all_bits(plan, 64);
	EXPECT_TRUE(has_boundary_near(plan, 22, 2));
}

TEST(SubIntSplitSelector, MaxSectionsIsRespected) {
	const std::vector<uint64_t> data = make_snowflake(VEC_SZ * 8);

	for (n_t k {1}; k <= 5; ++k) {
		SubIntSplitSelectorConfig cfg;
		cfg.max_sections  = k;
		cfg.split_penalty = 0.0; // remove the decode-speed bias so only max_sections limits the section count

		const SubIntSplitPlan plan = run<uint64_t>(data, 8, cfg);
		EXPECT_LE(plan.segments.size(), k) << "max_sections = " << k;
		expect_covers_all_bits(plan, 64);
	}
}

TEST(SubIntSplitSelector, MinSegmentWidthIsRespected) {
	const std::vector<uint64_t> data = make_snowflake(VEC_SZ * 4);

	SubIntSplitSelectorConfig cfg;
	cfg.min_segment_width = 16;

	const SubIntSplitPlan plan = run<uint64_t>(data, 4, cfg);
	expect_covers_all_bits(plan, 64);
	for (const auto& segment : plan.segments) {
		EXPECT_GE(segment.width(), 16);
	}
}

TEST(SubIntSplitSelector, SplitPenaltyBiasesTowardsFewerSections) {
	const std::vector<uint64_t> data = make_snowflake(VEC_SZ * 8);

	SubIntSplitSelectorConfig cheap;
	cheap.split_penalty = 0.0;
	SubIntSplitSelectorConfig expensive;
	expensive.split_penalty = 1'000'000.0;

	EXPECT_GE(run<uint64_t>(data, 8, cheap).segments.size(), run<uint64_t>(data, 8, expensive).segments.size());
	EXPECT_EQ(run<uint64_t>(data, 8, expensive).segments.size(), 1U);
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * 7. 32-bit and signed physical types.
\*--------------------------------------------------------------------------------------------------------------------*/
TEST(SubIntSplitSelector, Supports32BitTypes) {
	std::mt19937          rng {3};
	std::vector<uint32_t> data(VEC_SZ * 4);
	for (size_t i {0}; i < data.size(); ++i) {
		data[i] = (static_cast<uint32_t>(i / 512) << 16) | (rng() & 0xFFU); // two independent fields, boundary at 16
	}

	const SubIntSplitPlan plan = run<uint32_t>(data, 4);
	expect_covers_all_bits(plan, 32);
	EXPECT_TRUE(has_boundary_near(plan, 16, 2));
}

TEST(SubIntSplitSelector, SupportsSignedTypes) {
	std::vector<int64_t> data(VEC_SZ * 2);
	for (size_t i {0}; i < data.size(); ++i) {
		data[i] = -1'000'000 - static_cast<int64_t>(i);
	}

	const SubIntSplitPlan plan = run<int64_t>(data, 2);
	expect_covers_all_bits(plan, 64);
	// The unsigned reinterpretation makes the sign bit just the top bit; the per-vector FFOR base absorbs it.
	EXPECT_LT(plan.estimated_bits / static_cast<double>(VEC_SZ * 2), 13.0);
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * 8. the span-of-vectors overload agrees with the flat overload.
\*--------------------------------------------------------------------------------------------------------------------*/
TEST(SubIntSplitSelector, SpanOfVectorsOverloadMatchesFlatOverload) {
	const std::vector<uint64_t> data = make_snowflake(VEC_SZ * 4);

	std::vector<span<const uint64_t>> vectors;
	for (n_t offset {0}; offset < data.size(); offset += VEC_SZ) {
		vectors.emplace_back(data.data() + offset, VEC_SZ);
	}

	const SubIntSplitPlan flat  = run<uint64_t>(data, 4);
	const SubIntSplitPlan spans = select_splits<uint64_t>(span<const span<const uint64_t>> {vectors}, 4);

	ASSERT_EQ(flat.segments.size(), spans.segments.size());
	for (size_t i {0}; i < flat.segments.size(); ++i) {
		EXPECT_EQ(flat.segments[i].bit_start, spans.segments[i].bit_start);
		EXPECT_EQ(flat.segments[i].bit_end, spans.segments[i].bit_end);
	}
	EXPECT_DOUBLE_EQ(flat.estimated_bits, spans.estimated_bits);
}

} // namespace
