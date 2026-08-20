// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/include/fls/expression/subintsplit_selector.hpp
// ────────────────────────────────────────────────────────
#ifndef FLS_EXPRESSION_SUBINTSPLIT_SELECTOR_HPP
#define FLS_EXPRESSION_SUBINTSPLIT_SELECTOR_HPP

#include "fls/common/alias.hpp"
#include "fls/std/span.hpp"
#include "fls/std/vector.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <type_traits>

/*----------------------------------------------------------------------------------------------------------------------
 * SubIntSplit bit-range split selector.
 *
 * SubIntSplitEncoding cuts every integer of a column into contiguous bit ranges ("sections") and bit-packs each
 * section on its own. This pays off when the bits of a value carry distinct semantic fields: a Twitter snowflake id
 * is a 41-bit timestamp | 10-bit machine id | 12-bit sequence, and each field has a much narrower per-vector range
 * than the concatenated value does. This header decides *where* to cut.
 *
 * Cost oracle
 * -----------
 * Unlike the Nimble reference selector this file is ported from, the cost of a candidate range is not modelled with
 * closed-form formulas for a zoo of encodings. In FastLanes every section is FFOR-packed per 1024-value vector with
 * its own per-vector base and bit width, so the cost is directly *measurable* on sampled vectors:
 *
 *     for a candidate range [l, r], w = r - l + 1
 *     for every sampled vector v:
 *         section values = (value >> l) & ((1 << w) - 1)
 *         bw_v           = bit_width(max(section values) - min(section values))   // FFOR base = per-vector min
 *     cost_bits(l, r) = (n_vectors_total / |V|) * SUM_v [ VEC_SZ * bw_v + 8 * (sizeof(PT) + 1) ]
 *                       + section_overhead_bits
 *
 * The 8 * (sizeof(PT) + 1) term is the per-vector FFOR base plus the per-vector bit-width byte.
 *
 * A consequence worth internalising: because FFOR already subtracts a per-vector base, *constant high bits are
 * already free* in a single wide section. Splitting only wins when two fields vary independently, so that the
 * concatenated value spans a far wider range than the sum of the field ranges.
 *
 * split_penalty
 * -------------
 * In the Nimble reference the split penalty was a fudge factor. Here it has a concrete meaning: every extra section
 * costs 3 additional segments in the rowgroup *and* one more pass over the output buffer during bulk decode. It is
 * therefore the knob that trades compression ratio against decode speed - raise it to bias towards fewer, faster
 * sections. It is expressed in bits per 1024-value vector of the *full* column so that a single default stays
 * meaningful across column sizes; the DP scales it by n_vectors_total internally.
 *
 * max_sections
 * ------------
 * Enforced *inside the DP state* (dp is indexed by [section count][bit position]) rather than by rejecting finished
 * plans. Post-hoc rejection would need a fallback whose optimality is not guaranteed, whereas the extra DP dimension
 * yields the true optimum among all plans with at most max_sections sections at negligible cost: the state space is
 * only max_sections * 64.
 *
 * Dependencies are deliberately kept minimal (aliases, span, vector) so the selector can be unit-tested standalone
 * and reused by the wizard without dragging in rowgroup/segment headers.
\*--------------------------------------------------------------------------------------------------------------------*/

namespace fastlanes {
/*--------------------------------------------------------------------------------------------------------------------*/
namespace subintsplit {
/*--------------------------------------------------------------------------------------------------------------------*\
 * SubIntSplitSegment : one contiguous, inclusive bit range of the physical type.
\*--------------------------------------------------------------------------------------------------------------------*/
struct SubIntSplitSegment {
	bw_t bit_start {0};
	bw_t bit_end {0};

	[[nodiscard]] bw_t width() const noexcept {
		return static_cast<bw_t>(bit_end - bit_start + 1);
	}
};

/*--------------------------------------------------------------------------------------------------------------------*\
 * SubIntSplitPlan : the chosen partition of [0, kBits) plus its estimated size for the whole column.
\*--------------------------------------------------------------------------------------------------------------------*/
struct SubIntSplitPlan {
	vector<SubIntSplitSegment> segments;
	double                     estimated_bits {0.0};

	[[nodiscard]] n_t n_sections() const noexcept {
		return static_cast<n_t>(segments.size());
	}
};

/*--------------------------------------------------------------------------------------------------------------------*\
 * SubIntSplitSelectorConfig
\*--------------------------------------------------------------------------------------------------------------------*/
struct SubIntSplitSelectorConfig {
	// Narrowest section the selector is allowed to emit. Sections narrower than this are never worth their segment
	// overhead in practice, and forbidding them shrinks the candidate space.
	bw_t min_segment_width {1};

	// Largest number of sections a plan may contain. Enforced as a DP dimension, see the file comment.
	n_t max_sections {4};

	// Bits charged per split boundary beyond the first, *per 1024-value vector of the full column*. Trades
	// compression against decode speed, see the file comment. The DP multiplies this by n_vectors_total.
	double split_penalty {32.0};

	// Fixed bits charged once per section, covering the 3 extra rowgroup segments a section costs.
	double section_overhead_bits {3.0 * 8.0 * 8.0};

	// Upper bound on how many of the supplied vectors are actually measured. The candidate sweep is O(64 * 64 * 1024)
	// per vector, so this is the knob that bounds selector runtime; the sample is spread evenly over the supplied
	// vectors and the resulting cost is scaled back up to n_vectors_total.
	n_t max_sampled_vectors {8};
};

[[nodiscard]] inline SubIntSplitSelectorConfig default_selector_config() noexcept {
	return SubIntSplitSelectorConfig {};
}

/*--------------------------------------------------------------------------------------------------------------------*/
namespace detail {
/*--------------------------------------------------------------------------------------------------------------------*/
// Mirrors CFG::VEC_SZ. Replicated instead of including fls/cfg/cfg.hpp, which drags in the flatbuffers-generated
// data-type headers and would make this selector impossible to unit-test standalone.
constexpr n_t SUBINTSPLIT_VEC_SZ = 1024;

// Per-vector FFOR overhead in bits: the base value plus the bit-width byte.
template <typename PT>
constexpr double per_vector_overhead_bits() noexcept {
	return 8.0 * static_cast<double>(sizeof(PT) + 1);
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * BitRangeExtractor
 *
 * Incrementally materialises bits [bit_start .. bit_end] of the bound vector, extending one bit at a time so that the
 * O(64 * 64) candidate sweep costs one pass over the vector per candidate instead of one pass per bit per candidate.
 * That reuse is the whole reason this class exists - re-extracting every (l, r) from scratch would be 64x the work.
 *
 * The min/max fold is fused into the same pass, so a range extension touches the data exactly once, and the extractor
 * is deliberately bound to a *single* 1024-value vector at a time: at 8 KiB the materialised buffer stays L1-resident
 * across the entire 2080-candidate sweep. Sweeping all sampled vectors at once instead was measured ~10x slower.
\*--------------------------------------------------------------------------------------------------------------------*/
template <typename UT>
class BitRangeExtractor {
public:
	explicit BitRangeExtractor(const n_t capacity)
	    : m_values(capacity, UT {0})
	    , m_data(nullptr)
	    , m_n(0)
	    , m_bit_start(0)
	    , m_bit_end(0) {
	}

	// Bind the vector the following reset/extend calls operate on.
	void bind(const UT* data_p, const n_t n) noexcept {
		m_data = data_p;
		m_n    = n;
	}

	// Start a fresh range at [bit_start, bit_start]; returns the FFOR bit width of the bound vector for that range.
	bw_t reset(const bw_t bit_start) {
		m_bit_start = bit_start;
		m_bit_end   = bit_start;
		// __restrict is load-bearing: without it the compiler must assume the source and the materialised buffer may
		// alias, which serialises the load/store chain and blocks vectorisation of this hot sweep.
		const UT* __restrict src_p = m_data;
		UT* __restrict vals_p      = m_values.data();
		for (n_t i {0}; i < m_n; ++i) {
			vals_p[i] = (src_p[i] >> bit_start) & UT {1};
		}
		return fold();
	}

	// Pull the next higher bit into the range, i.e. [m_bit_start, m_bit_end + 1]; returns the new FFOR bit width.
	bw_t extend_one_bit() {
		++m_bit_end;
		const UT   shift           = static_cast<UT>(m_bit_end - m_bit_start);
		const bw_t bit_end         = m_bit_end;
		const UT* __restrict src_p = m_data;
		UT* __restrict vals_p      = m_values.data();
		for (n_t i {0}; i < m_n; ++i) {
			vals_p[i] = static_cast<UT>(vals_p[i] | (((src_p[i] >> bit_end) & UT {1}) << shift));
		}
		return fold();
	}

	[[nodiscard]] const vector<UT>& values() const noexcept {
		return m_values;
	}

private:
	// FastLanes FFOR uses the per-vector minimum as base, so the packed width is bit_width(max - min).
	// Four independent accumulator pairs: a single min/max pair is a loop-carried dependency that keeps the reduction
	// scalar and roughly halves throughput. Kept in its own pass rather than fused into the extension loop, which
	// measured ~2x faster because the extension loop then vectorises cleanly.
	[[nodiscard]] bw_t fold() const noexcept {
		if (m_n == 0) {
			return 0;
		}
		constexpr UT ut_max         = std::numeric_limits<UT>::max();
		const UT* __restrict vals_p = m_values.data();
		std::array<UT, 4> min_v {ut_max, ut_max, ut_max, ut_max};
		std::array<UT, 4> max_v {0, 0, 0, 0};
		const n_t         tail = m_n & ~n_t {3};
		for (n_t i {0}; i < tail; i += 4) {
			for (n_t k {0}; k < 4; ++k) {
				min_v[k] = std::min(min_v[k], vals_p[i + k]);
				max_v[k] = std::max(max_v[k], vals_p[i + k]);
			}
		}
		for (n_t i {tail}; i < m_n; ++i) {
			min_v[0] = std::min(min_v[0], vals_p[i]);
			max_v[0] = std::max(max_v[0], vals_p[i]);
		}
		const UT lo = std::min(std::min(min_v[0], min_v[1]), std::min(min_v[2], min_v[3]));
		const UT hi = std::max(std::max(max_v[0], max_v[1]), std::max(max_v[2], max_v[3]));
		return static_cast<bw_t>(std::bit_width(static_cast<uint64_t>(hi - lo)));
	}

private:
	vector<UT> m_values;
	const UT*  m_data;
	n_t        m_n;
	bw_t       m_bit_start;
	bw_t       m_bit_end;
};

/*--------------------------------------------------------------------------------------------------------------------*\
 * select_splits_impl : the DP itself, on already-flattened unsigned samples.
\*--------------------------------------------------------------------------------------------------------------------*/
template <typename PT, typename UT>
[[nodiscard]] SubIntSplitPlan select_splits_impl(const vector<UT>&                samples,
                                                 const vector<n_t>&               vector_offsets,
                                                 const n_t                        n_vectors_total,
                                                 const SubIntSplitSelectorConfig& config) {
	constexpr bw_t k_bits = static_cast<bw_t>(sizeof(PT) * 8);

	const n_t n_sampled_vectors = vector_offsets.empty() ? 0 : vector_offsets.size() - 1;

	// Degenerate input: no samples to measure. Hand back a usable full-width single-section plan with an unknown
	// (zero) estimate so the caller never has to special-case an empty segment list.
	if (samples.empty() || n_sampled_vectors == 0) {
		SubIntSplitPlan plan;
		plan.segments.push_back(SubIntSplitSegment {0, static_cast<bw_t>(k_bits - 1)});
		plan.estimated_bits = 0.0;
		return plan;
	}

	const bw_t min_width    = std::max<bw_t>(config.min_segment_width, 1);
	const n_t  max_sections = std::max<n_t>(config.max_sections, 1);

	// The sweep costs ~6 ms per sampled 64-bit vector, so cap how many vectors actually get measured. Vectors are
	// picked evenly spaced (never at random) to keep the selector deterministic and to spread the sample across the
	// rowgroup rather than biasing it towards the head.
	const n_t   n_used = std::min<n_t>(n_sampled_vectors, std::max<n_t>(config.max_sampled_vectors, 1));
	vector<n_t> used_vectors;
	used_vectors.reserve(n_used);
	for (n_t i {0}; i < n_used; ++i) {
		used_vectors.push_back(i * n_sampled_vectors / n_used);
	}

	const double scale =
	    static_cast<double>(std::max<n_t>(n_vectors_total, n_sampled_vectors)) / static_cast<double>(n_used);
	const double vector_overhead = per_vector_overhead_bits<PT>() * static_cast<double>(n_used);
	const double split_penalty   = config.split_penalty * static_cast<double>(std::max<n_t>(n_vectors_total, 1));

	/* Candidate sweep: sum_bw[l * k_bits + r] = SUM over sampled vectors of the FFOR bit width of range [l, r]. -----*
	 * The vector loop is outermost so the extractor's materialised buffer stays L1-resident, see BitRangeExtractor.  */
	vector<n_t> sum_bw(static_cast<size_t>(k_bits) * k_bits, 0);

	n_t max_vector_len {0};
	for (const n_t vec_idx : used_vectors) {
		max_vector_len = std::max<n_t>(max_vector_len, vector_offsets[vec_idx + 1] - vector_offsets[vec_idx]);
	}
	BitRangeExtractor<UT> extractor {max_vector_len};

	for (const n_t vec_idx : used_vectors) {
		const n_t beg = vector_offsets[vec_idx];
		extractor.bind(samples.data() + beg, vector_offsets[vec_idx + 1] - beg);
		for (bw_t l {0}; l < k_bits; ++l) {
			sum_bw[static_cast<size_t>(l) * k_bits + l] += extractor.reset(l);
			for (bw_t r = static_cast<bw_t>(l + 1); r < k_bits; ++r) {
				sum_bw[static_cast<size_t>(l) * k_bits + r] += extractor.extend_one_bit();
			}
		}
	}

	/* Turn measured bit widths into estimated bits for the full column. --------------------------------------------*/
	vector<double> cost(static_cast<size_t>(k_bits) * k_bits, std::numeric_limits<double>::infinity());
	for (bw_t l {0}; l < k_bits; ++l) {
		for (bw_t r {l}; r < k_bits; ++r) {
			const size_t idx         = static_cast<size_t>(l) * k_bits + r;
			const double packed_bits = static_cast<double>(SUBINTSPLIT_VEC_SZ * sum_bw[idx]);
			cost[idx]                = scale * (packed_bits + vector_overhead) + config.section_overhead_bits;
		}
	}

	/* DP over bit positions, with the section count as an explicit state dimension. --------------------------------*
	 * dp[k][i] = minimum cost to cover bits [0, i) with exactly k sections.                                          */
	constexpr double inf = std::numeric_limits<double>::infinity();
	const size_t     row = static_cast<size_t>(k_bits) + 1;
	vector<double>   dp((max_sections + 1) * row, inf);
	vector<int32_t>  prev((max_sections + 1) * row, -1);
	dp[0] = 0.0; // dp[0][0]

	for (n_t k {1}; k <= max_sections; ++k) {
		for (bw_t i {min_width}; i <= k_bits; ++i) {
			double  best      = inf;
			int32_t best_prev = -1;
			for (bw_t j {0}; j + min_width <= i; ++j) {
				const double prev_cost = dp[(k - 1) * row + j];
				if (prev_cost == inf) {
					continue;
				}
				const double boundary  = (j == 0) ? 0.0 : split_penalty;
				const double candidate = prev_cost + cost[static_cast<size_t>(j) * k_bits + (i - 1)] + boundary;
				if (candidate < best) {
					best      = candidate;
					best_prev = static_cast<int32_t>(j);
				}
			}
			dp[k * row + i]   = best;
			prev[k * row + i] = best_prev;
		}
	}

	/* Pick the best section count, then backtrack. -----------------------------------------------------------------*/
	n_t    best_k    = 0;
	double best_cost = inf;
	for (n_t k {1}; k <= max_sections; ++k) {
		const double candidate = dp[k * row + k_bits];
		if (candidate < best_cost) {
			best_cost = candidate;
			best_k    = k;
		}
	}

	SubIntSplitPlan plan;
	if (best_k == 0) {
		// Unreachable for min_width <= k_bits, but keep a total function: fall back to one full-width section.
		plan.segments.push_back(SubIntSplitSegment {0, static_cast<bw_t>(k_bits - 1)});
		plan.estimated_bits = cost[static_cast<size_t>(0) * k_bits + (k_bits - 1)];
		return plan;
	}

	plan.estimated_bits = best_cost;
	n_t  k              = best_k;
	bw_t i              = k_bits;
	while (k > 0) {
		const int32_t j = prev[k * row + i];
		if (j < 0) {
			break;
		}
		plan.segments.push_back(SubIntSplitSegment {static_cast<bw_t>(j), static_cast<bw_t>(i - 1)});
		i = static_cast<bw_t>(j);
		--k;
	}
	std::reverse(plan.segments.begin(), plan.segments.end());
	return plan;
}

} // namespace detail
/*--------------------------------------------------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------------------------------------------------*\
 * select_splits
 *
 * Chooses the minimum-estimated-size partition of the sizeof(PT) * 8 bits of a column into at most
 * config.max_sections contiguous sections.
 *
 * `sampled_vectors`  : the sampled 1024-value vectors of the column. Each span may hold fewer than 1024 values (a
 *                      tail vector); the cost model still charges a full VEC_SZ worth of packed bits, matching the
 *                      fact that FastLanes pads a vector out to 1024.
 * `n_vectors_total`  : number of vectors in the *full* column, used to scale sample costs up to column costs. Values
 *                      below the sample count are clamped up to it.
 *
 * Signed PT is handled on its unsigned reinterpretation, so the sign bit is simply the top bit of the value; the
 * per-vector FFOR base makes that harmless for the common all-negative or all-positive vector.
 *
 * Deterministic: the sweep and the DP use strict `<` comparisons over a fixed iteration order, so identical input
 * always yields an identical plan.
\*--------------------------------------------------------------------------------------------------------------------*/
template <typename PT>
[[nodiscard]] SubIntSplitPlan select_splits(const span<const span<const PT>> sampled_vectors,
                                            const n_t                        n_vectors_total,
                                            const SubIntSplitSelectorConfig& config = default_selector_config()) {
	static_assert(std::is_integral_v<PT>, "SubIntSplit selector operates on integral physical types");
	static_assert(sizeof(PT) == 4 || sizeof(PT) == 8, "SubIntSplit selector supports 32-bit and 64-bit types");

	using UT = std::make_unsigned_t<PT>;

	n_t n_values {0};
	for (const auto& vec : sampled_vectors) {
		n_values += static_cast<n_t>(vec.size());
	}

	vector<UT>  samples;
	vector<n_t> vector_offsets;
	samples.reserve(n_values);
	vector_offsets.reserve(sampled_vectors.size() + 1);
	vector_offsets.push_back(0);
	for (const auto& vec : sampled_vectors) {
		if (vec.empty()) {
			continue; // an empty vector carries no information and would only add overhead to every candidate
		}
		for (const PT value : vec) {
			samples.push_back(static_cast<UT>(value));
		}
		vector_offsets.push_back(static_cast<n_t>(samples.size()));
	}
	if (vector_offsets.size() == 1) {
		vector_offsets.clear();
	}

	return detail::select_splits_impl<PT, UT>(samples, vector_offsets, n_vectors_total, config);
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * select_splits : convenience overload over one flat, contiguous run of values, cut into 1024-value vectors.
\*--------------------------------------------------------------------------------------------------------------------*/
template <typename PT>
[[nodiscard]] SubIntSplitPlan select_splits(const span<const PT>             values,
                                            const n_t                        n_vectors_total,
                                            const SubIntSplitSelectorConfig& config = default_selector_config()) {
	vector<span<const PT>> sampled_vectors;
	for (n_t offset {0}; offset < static_cast<n_t>(values.size()); offset += detail::SUBINTSPLIT_VEC_SZ) {
		const n_t len = std::min<n_t>(detail::SUBINTSPLIT_VEC_SZ, static_cast<n_t>(values.size()) - offset);
		sampled_vectors.push_back(values.subspan(offset, len));
	}
	return select_splits<PT>(span<const span<const PT>> {sampled_vectors}, n_vectors_total, config);
}

} // namespace subintsplit
} // namespace fastlanes

#endif // FLS_EXPRESSION_SUBINTSPLIT_SELECTOR_HPP
