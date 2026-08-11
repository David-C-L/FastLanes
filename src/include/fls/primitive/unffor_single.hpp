// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/include/fls/primitive/unffor_single.hpp
// ────────────────────────────────────────────────────────
#ifndef FLS_PRIMITIVE_UNFFOR_SINGLE_HPP
#define FLS_PRIMITIVE_UNFFOR_SINGLE_HPP

#include "fls/common/alias.hpp"
#include <type_traits>

namespace fastlanes {

/*--------------------------------------------------------------------------------------------------------------------\
 * unffor_single : O(1) point access into a FFOR/bit-packed FastLanes vector.
 *
 * The regular `unffor` kernels always materialize a whole 1024-value vector. This primitive extracts exactly one
 * value, reading at most two packed words, and is bit-for-bit identical to `unffor(...)[idx]`.
 *
 * LAYOUT (derived from the generated kernels in src/alp/src/fastlanes_gen_unffor.cpp and
 * src/alp/src/fastlanes_gen_ffor.cpp — those files are the ground truth; re-verify against them if the
 * code generator ever changes).
 *
 * A vector holds 1024 values of element type PT. It is split into LANES = 128 / sizeof(PT) lanes
 * (128, 64, 32, 16 lanes for 8/16/32/64-bit PT) of 1024 / LANES slots each. Every generated kernel is a
 * `for (int i = 0; i < LANES; ++i)` loop, and inside it:
 *
 *   - the unpacked output of slot s is written to  out[i + LANES * s]                (natural index order,
 *     hence  lane = idx % LANES  and  slot = idx / LANES),
 *   - the packed words it reads are               in [i + LANES * w]                 (i.e. packed word w of every
 *     lane is stored contiguously; the lane stride is 1 and the word stride is LANES).
 *
 * Within a lane the slots are laid out as a plain little-endian bit stream of `bw`-bit fields, so slot s starts at
 * bit position s * bw, i.e. in word (s * bw) / TW at shift (s * bw) % TW, where TW = 8 * sizeof(PT). When a field
 * straddles a word boundary the kernels take the low (TW - shift) bits from word w and the remaining
 * (shift + bw - TW) bits from word w + 1, shifted up by (TW - shift) — exactly what is reproduced below.
 *
 * Verified against the generated kernels for every PT in {u8, u16, u32, u64} and every bw in [0, 8 * sizeof(PT)].
 *
 * Notes:
 *   - bw == 0 stores nothing at all; the kernel writes `base` to every output slot, so `in` is never dereferenced.
 *   - bw == TW is the no-straddle full-width case; the mask must not be computed as (1 << bw) - 1 (UB), and the
 *     straddle branch must not be taken (shift is always 0, and `x << (TW - shift)` would be a shift by TW).
 *   - the base is added with wrapping unsigned arithmetic in PT, matching `tmp += base` in the kernels.
 *   - PT must be an unsigned integer type; signed vectors are handled by the callers reinterpreting them as the
 *     unsigned type of the same width, which is what the generated kernels do internally too.
\--------------------------------------------------------------------------------------------------------------------*/
template <typename PT>
constexpr PT unffor_single(const PT* in, const bw_t bw, const PT base, const n_t idx) {
	static_assert(std::is_unsigned_v<PT>, "unffor_single requires an unsigned element type");

	constexpr n_t LANES = 128 / sizeof(PT); // 128, 64, 32, 16
	constexpr n_t TW    = 8 * sizeof(PT);   // bits per packed word

	if (bw == 0) {
		return base;
	}

	const n_t lane   = idx % LANES;
	const n_t slot   = idx / LANES;
	const n_t bitpos = slot * static_cast<n_t>(bw);
	const n_t word   = bitpos / TW;
	const n_t shift  = bitpos % TW;

	// mask of `bw` low bits, safe for bw == TW
	const PT mask = static_cast<PT>(bw == TW ? static_cast<PT>(~static_cast<PT>(0))
	                                         : static_cast<PT>((static_cast<PT>(1) << bw) - 1));

	PT value = static_cast<PT>(in[word * LANES + lane] >> shift);
	if (shift + static_cast<n_t>(bw) > TW) {
		// shift > 0 is guaranteed here (shift == 0 implies bw > TW, impossible), so TW - shift is in [1, TW - 1]
		value = static_cast<PT>(value | static_cast<PT>(in[(word + 1) * LANES + lane] << (TW - shift)));
	}

	return static_cast<PT>(static_cast<PT>(value & mask) + base);
}

} // namespace fastlanes

#endif // FLS_PRIMITIVE_UNFFOR_SINGLE_HPP
