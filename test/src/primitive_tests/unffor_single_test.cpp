// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// test/src/primitive_tests/unffor_single_test.cpp
// ────────────────────────────────────────────────────────
// Google Test suite for fastlanes::unffor_single.
//
// unffor_single must be bit-for-bit identical to the generated whole-vector `unffor` kernel at a single index.
// The reference is therefore the kernel itself: pack 1024 pseudo-random values with `ffor`, unpack them with
// `unffor`, and compare every one of the 1024 outputs against unffor_single. Exhaustive over PT in
// {u8, u16, u32, u64} and over every bit width in [0, 8 * sizeof(PT)], with both a zero and a non-zero base.

#include "fls/ffor.hpp"
#include "fls/primitive/unffor_single.hpp"
#include "fls/unffor.hpp"
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace fastlanes {

namespace {
constexpr n_t N_VALUES = 1024;
constexpr n_t SEED     = 0xF1A57; // fixed seed: the test must be reproducible

template <typename PT>
constexpr PT bw_mask(const bw_t bw) {
	constexpr bw_t TW = static_cast<bw_t>(8 * sizeof(PT));
	if (bw == 0) {
		return static_cast<PT>(0);
	}
	if (bw == TW) {
		return static_cast<PT>(~static_cast<PT>(0));
	}
	return static_cast<PT>((static_cast<PT>(1) << bw) - 1);
}
} // namespace

// -----------------------------------------------------------------------------
// Typed-test fixture
// -----------------------------------------------------------------------------

template <typename PT>
class UnfforSingleTest : public ::testing::Test {
protected:
	// Packs `in` with `ffor`, unpacks with `unffor`, and asserts unffor_single agrees at every index.
	static void check_all_indices(const bw_t bw, const PT base) {
		std::mt19937_64 rng(SEED + sizeof(PT) + bw);
		const PT        mask = bw_mask<PT>(bw);

		std::vector<PT> in(N_VALUES);
		for (n_t i = 0; i < N_VALUES; ++i) {
			in[i] = static_cast<PT>((static_cast<PT>(rng()) & mask) + base);
		}

		// a few slack words so that a hypothetical over-read is caught by the sanitizers rather than by luck
		std::vector<PT> packed(N_VALUES + 64, static_cast<PT>(0));
		std::vector<PT> out(N_VALUES, static_cast<PT>(0));

		ffor::ffor(in.data(), packed.data(), bw, &base);
		unffor::unffor(packed.data(), out.data(), bw, &base);

		for (n_t i = 0; i < N_VALUES; ++i) {
			ASSERT_EQ(out[i], in[i]) << "ffor/unffor round-trip failed, bw = " << static_cast<n_t>(bw)
			                         << ", idx = " << i;
			ASSERT_EQ(unffor_single<PT>(packed.data(), bw, base, i), out[i])
			    << "unffor_single mismatch, bw = " << static_cast<n_t>(bw) << ", idx = " << i;
		}
	}
};

using TestTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
TYPED_TEST_SUITE(UnfforSingleTest, TestTypes);

// -----------------------------------------------------------------------------
// 1. Every bit width, zero base  -----------------------------------------------
// -----------------------------------------------------------------------------

TYPED_TEST(UnfforSingleTest, MatchesUnfforForEveryBitWidthZeroBase) {
	using PT          = TypeParam;
	constexpr bw_t TW = static_cast<bw_t>(8 * sizeof(PT));

	for (bw_t bw = 0; bw <= TW; ++bw) {
		TestFixture::check_all_indices(bw, static_cast<PT>(0));
	}
}

// -----------------------------------------------------------------------------
// 2. Every bit width, non-zero base  -------------------------------------------
// -----------------------------------------------------------------------------

TYPED_TEST(UnfforSingleTest, MatchesUnfforForEveryBitWidthNonZeroBase) {
	using PT          = TypeParam;
	constexpr bw_t TW = static_cast<bw_t>(8 * sizeof(PT));
	const PT       base {static_cast<PT>(0x5A5A5A5A5A5A5A5AULL)};

	for (bw_t bw = 0; bw <= TW; ++bw) {
		TestFixture::check_all_indices(bw, base);
	}
}

// -----------------------------------------------------------------------------
// 3. bw == 0 returns the base without touching the buffer  ---------------------
// -----------------------------------------------------------------------------

TYPED_TEST(UnfforSingleTest, ZeroBitWidthReturnsBase) {
	using PT = TypeParam;

	const PT base {static_cast<PT>(37)};
	for (n_t i = 0; i < N_VALUES; ++i) {
		ASSERT_EQ(unffor_single<PT>(nullptr, 0, base, i), base) << "idx = " << i;
	}
}

} // namespace fastlanes
