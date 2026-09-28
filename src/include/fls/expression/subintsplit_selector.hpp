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
 * Every section is stored with one of the codecs subintsplit_section_selector.hpp may pick for it: Uncompressed,
 * Constant, Dictionary (FFOR-packed indices), RLE, FFOR, FFOR_SLPATCH (patched FFOR) and Frequency. The DP must see
 * the same menu when it places boundaries, otherwise it is blind to e.g. a low-cardinality wide field that Dictionary
 * stores in a couple of bits but FFOR prices at its full per-vector range. So the cost of a candidate range [l, r] is
 * the *minimum over the enabled section codecs* of a cheap size estimate computed on the sampled vectors, where
 *
 *     section values = (value >> l) & ((1 << w) - 1),    w = r - l + 1
 *
 * and each estimate mirrors the codec's real FastLanes layout, per-vector metadata included:
 *
 *     FFOR          VEC_SZ * bit_width(max - min)                 + base + bit-width byte              per vector
 *     FFOR_SLPATCH  min over k < 20 exceptions of
 *                   VEC_SZ * bit_width(best window of n - k values) + k * (value + u16 position)
 *                                                                  + u16 count + base + bit-width byte per vector
 *     RLE           runs * value + rsum bases + VEC_SZ * (runs > 1) + u16 FFOR base + bit-width byte   per vector
 *     Frequency     exceptions * (value + u16 position) + u16 count                                    per vector
 *                   + the most frequent value                                                          once
 *     Dictionary    VEC_SZ * bit_width(ndv - 1) + index base + bit-width byte                          per vector
 *                   + ndv * value                                                                      once
 *     Constant      value (only when the sample holds a single distinct value)                         once
 *     Uncompressed  VEC_SZ * value                                                                     per vector
 *
 * Per-vector terms are summed over the sampled vectors and scaled by n_vectors_total / |V|; once-per-section terms
 * (the dictionary, the frequent value, the constant) are not scaled. The dictionary size ndv is extrapolated from the
 * sample as ndv_sample + f1 * (scale - 1), f1 being the values seen exactly once: saturated low-cardinality fields
 * (f1 ~ 0) keep their sample ndv, while all-distinct fields scale linearly. This deliberately errs towards
 * over-estimating ndv, so Dictionary only steers a boundary when it clearly wins. The dictionary index bit width uses
 * the whole-section ndv, an upper bound on the per-vector index range. Every value is stored at sizeof(PT) - a section
 * keeps the column's physical type. section_overhead_bits is charged once per section on top, whatever the codec.
 *
 * Setting section_codecs to SECTION_CODEC_FFOR reproduces the original FFOR-width-only model exactly.
 *
 * A consequence worth internalising: because FFOR already subtracts a per-vector base, *constant high bits are
 * already free* in a single wide section. Splitting only wins when two fields vary independently, so that the
 * concatenated value spans a far wider range than the sum of the field costs.
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
 * Section codecs the split cost may price a range with (a bit set). SECTION_CODECS_ALL is exactly the menu
 * subintsplit_section_selector.hpp offers a section, see the file comment.
\*--------------------------------------------------------------------------------------------------------------------*/
using section_codec_set_t = uint32_t;

constexpr section_codec_set_t SECTION_CODEC_UNCOMPRESSED = 1U << 0U;
constexpr section_codec_set_t SECTION_CODEC_CONSTANT     = 1U << 1U;
constexpr section_codec_set_t SECTION_CODEC_DICTIONARY   = 1U << 2U;
constexpr section_codec_set_t SECTION_CODEC_RLE          = 1U << 3U;
constexpr section_codec_set_t SECTION_CODEC_FFOR         = 1U << 4U;
constexpr section_codec_set_t SECTION_CODEC_FFOR_SLPATCH = 1U << 5U;
constexpr section_codec_set_t SECTION_CODEC_FREQUENCY    = 1U << 6U;
constexpr section_codec_set_t SECTION_CODECS_ALL         = SECTION_CODEC_UNCOMPRESSED | SECTION_CODEC_CONSTANT |
                                                   SECTION_CODEC_DICTIONARY | SECTION_CODEC_RLE | SECTION_CODEC_FFOR |
                                                   SECTION_CODEC_FFOR_SLPATCH | SECTION_CODEC_FREQUENCY;

/*--------------------------------------------------------------------------------------------------------------------*\
 * SubIntSplitSelectorConfig
\*--------------------------------------------------------------------------------------------------------------------*/
struct SubIntSplitSelectorConfig {
	// Codecs whose estimated size the split cost takes the minimum over. SECTION_CODEC_FFOR alone is the original
	// FFOR-width-only model; an empty set falls back to it.
	section_codec_set_t section_codecs {SECTION_CODECS_ALL};

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
	// per vector (plus column-wide statistics over all measured vectors), so this is the knob that bounds selector
	// runtime; the sample is spread evenly over the supplied vectors and the resulting cost is scaled back up to
	// n_vectors_total.
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

// Layout constants of the section codecs, replicated for the same standalone-testability reason as VEC_SZ.
constexpr n_t    SLPATCH_EXCEPTION_LIMIT = 20;   // analyze_operator_impl.hpp LOCAL_EXC_LIMIT_C: strictly fewer allowed
constexpr double VEC_IDX_BITS            = 16.0; // vec_idx_t: exception positions and exception counts
constexpr double RSUM_BASES_BITS         = 8.0 * 128.0; // CFG::UNIFIED_TRANSPOSED::BASES_SIZE, RLE's run-index bases

/*--------------------------------------------------------------------------------------------------------------------*\
 * RangeStats : what the codec size estimates need to know about one candidate range on the sampled vectors.
\*--------------------------------------------------------------------------------------------------------------------*/
struct RangeStats {
	// Column-wide, over all sampled values.
	n_t n_distinct {0};  // distinct values
	n_t n_singleton {0}; // values seen exactly once (f1)
	// Summed over sampled vectors.
	n_t    sum_ffor_bw {0};        // FFOR bit width
	double sum_slpatch_bits {0.0}; // best patched-FFOR payload (packed bits + exceptions), metadata excluded
	n_t    sum_runs {0};           // RLE runs
	n_t    n_multi_run {0};        // vectors with more than one run (their run-index deltas need 1 bit)
	n_t    sum_freq_exc {0};       // values differing from the column's most frequent value
};

/*--------------------------------------------------------------------------------------------------------------------*\
 * BitRangeSweep
 *
 * Maintains, for the bound sampled vectors, the statistics of the range [bit_start .. bit_end] while it grows one bit
 * at a time, so the O(64 * 64) candidate sweep costs O(n) per candidate instead of a sort per candidate.
 *
 * The trick is to keep copies of the sampled raw values sorted by the current range value. Pulling the next *higher*
 * bit into the range makes that bit the most significant one, so the new sorted order is simply a stable partition of
 * the old one by the new bit - O(n), no re-sort, sequential and branch-free. Two such copies are kept:
 *
 *     m_column  all sampled values sorted together: distinct count, singleton count and most frequent value
 *               (Dictionary, Frequency, Constant)
 *     m_vectors every vector's values sorted within the vector: FFOR width (last - first) and patched-FFOR windows
 *
 * RLE's run boundaries in original order only ever gain members as bits are added, so they are OR-ed in per bit.
 *
 * Unlike an FFOR-only sweep this cannot visit one vector at a time, because Dictionary and Frequency depend on
 * column-wide statistics. The working set is a few copies of at most max_sampled_vectors * 1024 values, which stays
 * L2-resident at the default of 8 vectors.
\*--------------------------------------------------------------------------------------------------------------------*/
template <typename UT>
class BitRangeSweep {
public:
	// `data_p` holds the sampled vectors back to back; `offsets` has one entry per vector plus the end.
	BitRangeSweep(const UT* data_p, const vector<n_t>& offsets)
	    : m_data(data_p)
	    , m_n(offsets.back())
	    , m_offsets(offsets)
	    , m_column(offsets.back(), UT {0})
	    , m_vectors(offsets.back(), UT {0})
	    , m_zeros(offsets.back(), UT {0})
	    , m_ones(offsets.back(), UT {0})
	    , m_run_start(offsets.back(), 0)
	    , m_varying_bits(0)
	    , m_bit_start(0)
	    , m_bit_end(0) {
		for (n_t i {0}; i < m_n; ++i) {
			m_varying_bits = static_cast<UT>(m_varying_bits | (m_data[i] ^ m_data[0]));
		}
	}

	// Start a fresh range at [bit_start, bit_start].
	void reset(const bw_t bit_start) {
		m_bit_start = bit_start;
		m_bit_end   = bit_start;
		std::copy(m_data, m_data + m_n, m_column.begin());
		std::copy(m_data, m_data + m_n, m_vectors.begin());
		partition_all(bit_start);
		for (n_t i {1}; i < m_n; ++i) {
			m_run_start[i] = static_cast<uint8_t>(((m_data[i] ^ m_data[i - 1]) >> bit_start) & UT {1});
		}
		for (n_t v {0}; v + 1 < m_offsets.size(); ++v) {
			m_run_start[m_offsets[v]] = 1; // every vector is run-length encoded on its own
		}
	}

	// Pull the next higher bit into the range, i.e. [m_bit_start, m_bit_end + 1].
	void extend_one_bit() {
		++m_bit_end;
		const bw_t bit_end = m_bit_end;
		partition_all(bit_end);
		if (((m_varying_bits >> bit_end) & UT {1}) == 0) {
			return; // nor can it add a run boundary
		}
		const UT* __restrict src_p   = m_data;
		uint8_t* __restrict starts_p = m_run_start.data();
		for (n_t i {1}; i < m_n; ++i) {
			starts_p[i] = static_cast<uint8_t>(starts_p[i] | (((src_p[i] ^ src_p[i - 1]) >> bit_end) & UT {1}));
		}
	}

	// Statistics of the current range; `value_bits` is the stored width of one section value (8 * sizeof(PT)).
	[[nodiscard]] RangeStats measure(const double value_bits) const {
		const bw_t width = static_cast<bw_t>(m_bit_end - m_bit_start + 1);
		const UT   mask  = width >= sizeof(UT) * 8 ? ~UT {0} : static_cast<UT>((UT {1} << width) - 1);
		const bw_t shift = m_bit_start;
		const auto value = [&](const UT raw) {
			return static_cast<UT>((raw >> shift) & mask);
		};

		RangeStats stats;

		/* Column-wide: walk the groups of equal values in sorted order. Strict `>` keeps the smallest of equally
		 * frequent values, so the most frequent value is deterministic. ---------------------------------------------*/
		UT  top_value {0};
		n_t top_count {0};
		n_t group_begin {0};
		UT  group_value = value(m_column[0]);
		for (n_t i {1}; i <= m_n; ++i) {
			const UT current = i < m_n ? value(m_column[i]) : group_value;
			if (i < m_n && current == group_value) {
				continue;
			}
			const n_t count = i - group_begin;
			++stats.n_distinct;
			stats.n_singleton += count == 1 ? 1 : 0;
			if (count > top_count) {
				top_count = count;
				top_value = group_value;
			}
			group_begin = i;
			group_value = current;
		}

		/* Per vector, on the vector's own sorted values. ------------------------------------------------------------*/
		for (n_t v {0}; v + 1 < m_offsets.size(); ++v) {
			const n_t beg = m_offsets[v];
			const n_t n   = m_offsets[v + 1] - beg;
			const UT* s   = m_vectors.data() + beg;

			stats.sum_ffor_bw += static_cast<n_t>(std::bit_width(static_cast<uint64_t>(value(s[n - 1]) - value(s[0]))));

			// Patched FFOR: with k exceptions the packed values are the best n - k consecutive sorted values.
			double    best_slpatch = std::numeric_limits<double>::infinity();
			const n_t max_exc      = std::min<n_t>(SLPATCH_EXCEPTION_LIMIT - 1, n - 1);
			for (n_t k {0}; k <= max_exc; ++k) {
				UT window = std::numeric_limits<UT>::max();
				for (n_t i {0}; i <= k; ++i) {
					window = std::min<UT>(window, static_cast<UT>(value(s[i + n - 1 - k]) - value(s[i])));
				}
				const double bits =
				    static_cast<double>(SUBINTSPLIT_VEC_SZ * std::bit_width(static_cast<uint64_t>(window))) +
				    static_cast<double>(k) * (value_bits + VEC_IDX_BITS);
				best_slpatch = std::min(best_slpatch, bits);
			}
			stats.sum_slpatch_bits += best_slpatch;

			n_t runs {0};
			for (n_t i {beg}; i < beg + n; ++i) {
				runs += m_run_start[i];
			}
			stats.sum_runs += runs;
			stats.n_multi_run += runs > 1 ? 1 : 0;

			const UT* lo = std::partition_point(s, s + n, [&](const UT raw) { return value(raw) < top_value; });
			const UT* hi = std::partition_point(lo, s + n, [&](const UT raw) { return value(raw) == top_value; });
			stats.sum_freq_exc += n - static_cast<n_t>(hi - lo);
		}
		return stats;
	}

private:
	// Stable partition of values[beg, end) by bit `bit`: zeros first, ones after, each in their old order. Both
	// outputs are written unconditionally and only the cursors move, so the loop has no data-dependent branch.
	void partition(vector<UT>& values, const n_t beg, const n_t end, const bw_t bit) {
		UT* __restrict zeros_p = m_zeros.data();
		UT* __restrict ones_p  = m_ones.data();
		UT* __restrict vals_p  = values.data();
		n_t zeros {0};
		n_t ones {0};
		for (n_t i {beg}; i < end; ++i) {
			const UT  raw    = vals_p[i];
			const n_t is_one = static_cast<n_t>((raw >> bit) & UT {1});
			zeros_p[zeros]   = raw;
			ones_p[ones]     = raw;
			zeros += 1 - is_one;
			ones += is_one;
		}
		std::copy(zeros_p, zeros_p + zeros, vals_p + beg);
		std::copy(ones_p, ones_p + ones, vals_p + beg + zeros);
	}

	void partition_all(const bw_t bit) {
		if (((m_varying_bits >> bit) & UT {1}) == 0) {
			return; // a bit that never varies in the sample leaves every order unchanged
		}
		partition(m_column, 0, m_n, bit);
		for (n_t v {0}; v + 1 < m_offsets.size(); ++v) {
			partition(m_vectors, m_offsets[v], m_offsets[v + 1], bit);
		}
	}

private:
	const UT*       m_data;
	n_t             m_n;
	vector<n_t>     m_offsets;
	vector<UT>      m_column;  // all sampled raw values, sorted by the current range value
	vector<UT>      m_vectors; // each vector's raw values, sorted by the current range value within the vector
	vector<UT>      m_zeros;   // partition scratch
	vector<UT>      m_ones;    // partition scratch
	vector<uint8_t> m_run_start;
	UT              m_varying_bits; // bits that are not the same in every sampled value
	bw_t            m_bit_start;
	bw_t            m_bit_end;
};

/*--------------------------------------------------------------------------------------------------------------------*\
 * estimate_range_bits : the file comment's cost table, minimised over the enabled codecs, for the full column.
\*--------------------------------------------------------------------------------------------------------------------*/
template <typename PT>
[[nodiscard]] double estimate_range_bits(const RangeStats&         stats,
                                         const n_t                 n_used,
                                         const n_t                 n_values,
                                         const double              scale,
                                         const section_codec_set_t codecs) {
	constexpr double value_bits = 8.0 * static_cast<double>(sizeof(PT));
	constexpr double vec_sz     = static_cast<double>(SUBINTSPLIT_VEC_SZ);
	const double     n_vec      = static_cast<double>(n_used);
	const auto       enabled    = [&](const section_codec_set_t codec) {
        return (codecs & codec) != 0 || (codecs == 0 && codec == SECTION_CODEC_FFOR);
	};

	double best = std::numeric_limits<double>::infinity();
	if (enabled(SECTION_CODEC_FFOR)) {
		const double per_vector =
		    vec_sz * static_cast<double>(stats.sum_ffor_bw) + (per_vector_overhead_bits<PT>() * n_vec);
		best = std::min(best, scale * per_vector);
	}
	if (enabled(SECTION_CODEC_FFOR_SLPATCH)) {
		const double per_vector = stats.sum_slpatch_bits + ((per_vector_overhead_bits<PT>() + VEC_IDX_BITS) * n_vec);
		best                    = std::min(best, scale * per_vector);
	}
	if (enabled(SECTION_CODEC_RLE)) {
		// Run values at full width, run-index deltas (0/1 after rsum) FFOR-packed as u16 with their rsum bases.
		const double per_vector = (static_cast<double>(stats.sum_runs) * value_bits) +
		                          (vec_sz * static_cast<double>(stats.n_multi_run)) +
		                          ((RSUM_BASES_BITS + per_vector_overhead_bits<uint16_t>()) * n_vec);
		best = std::min(best, scale * per_vector);
	}
	if (enabled(SECTION_CODEC_FREQUENCY)) {
		const double per_vector =
		    (static_cast<double>(stats.sum_freq_exc) * (value_bits + VEC_IDX_BITS)) + (VEC_IDX_BITS * n_vec);
		best = std::min(best, (scale * per_vector) + value_bits);
	}
	if (enabled(SECTION_CODEC_DICTIONARY)) {
		const double n_total    = static_cast<double>(n_values) * scale;
		const double n_distinct = std::min(
		    n_total, static_cast<double>(stats.n_distinct) + (static_cast<double>(stats.n_singleton) * (scale - 1.0)));
		if (n_distinct <= 4294967296.0) {
			const double index_bytes = n_distinct <= 256.0 ? 1.0 : (n_distinct <= 65536.0 ? 2.0 : 4.0);
			const auto   index_bw    = std::bit_width(static_cast<uint64_t>(std::max(n_distinct, 1.0) - 1.0));
			const double per_vector  = (vec_sz * static_cast<double>(index_bw) + (8.0 * (index_bytes + 1.0))) * n_vec;
			best                     = std::min(best, (scale * per_vector) + (n_distinct * value_bits));
		}
	}
	if (enabled(SECTION_CODEC_CONSTANT) && stats.n_distinct == 1) {
		best = std::min(best, value_bits);
	}
	if (enabled(SECTION_CODEC_UNCOMPRESSED)) {
		best = std::min(best, scale * vec_sz * value_bits * n_vec);
	}
	return best;
}

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

	// The sweep costs a few ms per sampled 64-bit vector, so cap how many vectors actually get measured. Vectors are
	// picked evenly spaced (never at random) to keep the selector deterministic and to spread the sample across the
	// rowgroup rather than biasing it towards the head.
	const n_t   n_used = std::min<n_t>(n_sampled_vectors, std::max<n_t>(config.max_sampled_vectors, 1));
	vector<UT>  used_samples;
	vector<n_t> used_offsets;
	used_offsets.reserve(n_used + 1);
	used_offsets.push_back(0);
	for (n_t i {0}; i < n_used; ++i) {
		const n_t vec_idx = i * n_sampled_vectors / n_used;
		used_samples.insert(used_samples.end(),
		                    samples.begin() + static_cast<std::ptrdiff_t>(vector_offsets[vec_idx]),
		                    samples.begin() + static_cast<std::ptrdiff_t>(vector_offsets[vec_idx + 1]));
		used_offsets.push_back(used_samples.size());
	}

	const double scale =
	    static_cast<double>(std::max<n_t>(n_vectors_total, n_sampled_vectors)) / static_cast<double>(n_used);
	const double split_penalty = config.split_penalty * static_cast<double>(std::max<n_t>(n_vectors_total, 1));

	/* Candidate sweep: cost[l * k_bits + r] = estimated bits of range [l, r] for the full column. ------------------*/
	vector<double>    cost(static_cast<size_t>(k_bits) * k_bits, std::numeric_limits<double>::infinity());
	BitRangeSweep<UT> sweep {used_samples.data(), used_offsets};
	const auto        price = [&](const bw_t l, const bw_t r) {
        const RangeStats stats = sweep.measure(8.0 * static_cast<double>(sizeof(PT)));
        cost[static_cast<size_t>(l) * k_bits + r] =
            estimate_range_bits<PT>(stats, n_used, used_samples.size(), scale, config.section_codecs) +
            config.section_overhead_bits;
	};
	for (bw_t l {0}; l < k_bits; ++l) {
		sweep.reset(l);
		price(l, l);
		for (bw_t r = static_cast<bw_t>(l + 1); r < k_bits; ++r) {
			sweep.extend_one_bit();
			price(l, r);
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
