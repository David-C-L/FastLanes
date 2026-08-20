// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/include/fls/expression/subintsplit_operator.hpp
// ────────────────────────────────────────────────────────
#ifndef FLS_EXPRESSION_SUBINTSPLIT_OPERATOR_HPP
#define FLS_EXPRESSION_SUBINTSPLIT_OPERATOR_HPP

#include "fls/expression/subintsplit_selector.hpp"
#include "fls/footer/operator_token_generated.h"
#include "fls/reader/segment.hpp"
#include "fls/std/type_traits.hpp"
#include "fls/table/rowgroup.hpp"

/*----------------------------------------------------------------------------------------------------------------------
 * SubIntSplit encoding operators.
 *
 * Every value of the column is cut into K contiguous bit ranges ("sections"), chosen once per column by
 * subintsplit::select_splits. Each section is FFOR bit-packed on its own, per 1024-value vector, with its own base and
 * bit width - so a value whose bits carry independent semantic fields (a snowflake id is timestamp | machine | seq)
 * pays the narrow per-field width instead of the wide width of the concatenation.
 *
 * Storage layout, in rowgroup segment order:
 *
 *     section 0 : bitpacked | base | bitwidth     (per-vector, exactly the enc_ffor_opr triple)
 *     ...
 *     section K-1 : bitpacked | base | bitwidth
 *     header                                      (block-based, written once in Finalize)
 *
 * The header is `uint8_t n_sections` followed by n_sections 4-byte records: bit_start (1B), the section's chosen
 * OperatorToken (2B), and its operand count (1B). bit_end is implied: section s ends where section s+1 starts, and
 * the last section ends at 8 * sizeof(PT) - 1. Every section's token is currently EXP_FFOR_I64/I32 with operand
 * count 3 (the classic bitpacked|base|bitwidth triple) -- the format already carries a genuine per-section codec
 * choice, but Encode()/Decode() don't act on it yet. See subintsplit_section_selector.hpp for the (already wired
 * but not yet consulted) selector that will choose it in the next step.
 *
 * The header comes LAST on purpose. This is the only operator in the codebase with a *dynamic* segment count, so the
 * decoder cannot know how far to reach back until it has read K (and, once per-section codecs vary, each section's
 * own operand count); putting the header at `cur_operand - 0` lets it read the header first, then index the section
 * segments below it via a running prefix sum of persisted operand counts, then rewind cur_operand accordingly.
 * Making it block-based keeps PhysicalExpr::Size from charging the whole header against the sampled vectors during
 * wizard selection.
\*--------------------------------------------------------------------------------------------------------------------*/

namespace fastlanes {
/*--------------------------------------------------------------------------------------------------------------------*/
class Segment;
struct ColumnDescriptorT;
class PhysicalExpr;
class ColumnView;
struct InterpreterState;
/*--------------------------------------------------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------------------------------------------------*\
 * enc_subintsplit_opr
\*--------------------------------------------------------------------------------------------------------------------*/
template <typename PT>
struct enc_subintsplit_opr {
public:
	using UT = make_unsigned_t<PT>;

	explicit enc_subintsplit_opr(const PhysicalExpr& expr,
	                             const col_pt&       col,
	                             ColumnDescriptorT&  column_descriptor,
	                             InterpreterState&   state);

	void PointTo(n_t vec_idx);
	void Encode();
	void Finalize();
	void MoveSegments(vector<up<Segment>>& segments);

public:
	TypedColumnView<PT> col_viewer;
	// The chosen partition of the sizeof(PT) * 8 bits, LSB-first, always tiling the whole width.
	vector<subintsplit::SubIntSplitSegment> sections;
	// One codec token + operand count per section, persisted in the header. Currently always
	// EXP_FFOR_I64/I32 and 3 -- see the file header comment.
	vector<OperatorToken> section_tokens;
	vector<uint8_t>       section_operand_counts;
	// One triple per section; sized from sections.size() in the constructor.
	vector<up<Segment>> bitpacked_segments;
	vector<up<Segment>> base_segments;
	vector<up<Segment>> bitwidth_segments;
	// Block-based, written once by Finalize.
	up<Segment> header_segment;
	// Scratch: the extracted section values and their bit-packed image.
	alignas(64) UT section_arr[CFG::VEC_SZ];
	alignas(64) UT bitpacked_arr[CFG::VEC_SZ];
};

/*--------------------------------------------------------------------------------------------------------------------*\
 * dec_subintsplit_opr
\*--------------------------------------------------------------------------------------------------------------------*/
template <typename PT>
struct dec_subintsplit_opr {
public:
	using UT = make_unsigned_t<PT>;

	explicit dec_subintsplit_opr(PhysicalExpr& physical_expr, const ColumnView& column_view, InterpreterState& state);

public:
	void PointTo(n_t vec_idx);
	void Decode(n_t vec_idx);
	void Materialize(n_t vec_idx, TypedCol<PT>& typed_col);

	/*----------------------------------------------------------------------------------------------------------------
	 * Point and gather access.
	 *
	 * FastLanes has neither path for any other encoding: unffor is all-or-nothing over a vector, so reading one value
	 * means decoding the 1024 values around it. SubIntSplit can do better, because unffor_single reaches a single
	 * value by position arithmetic - one value costs K scattered word reads, one per section, instead of K decoded
	 * vectors.
	 *
	 * Note what this does and does not beat. Against decode-the-whole-vector it wins by a wide margin; against a
	 * single-section FFOR column read the same way it necessarily loses, because K reads cost more than one. Splitting
	 * buys compression on this path, not speed.
	 *---------------------------------------------------------------------------------------------------------------*/

	// One value from the vector the segment views currently point at.
	[[nodiscard]] PT ValueAt(n_t idx) const;
	// One value, pointing the segment views first.
	[[nodiscard]] PT PointAccess(n_t vec_idx, n_t idx);

	// Gather `n` rows of one vector. Pointwise reaches only the rows asked for; Decoded decodes the whole vector and
	// then gathers. Decoded is also the like-for-like comparison against Nimble's bulkScan, which likewise decodes a
	// whole span and gathers out of it - within a vector, "decode the span" and "decode the vector" are the same
	// thing here, because unffor cannot decode part of one.
	void GatherPointwise(n_t vec_idx, const idx_t* rows, n_t n, PT* out);
	void GatherDecoded(n_t vec_idx, const idx_t* rows, n_t n, PT* out);
	// Picks between the two on selectivity.
	void Gather(n_t vec_idx, const idx_t* rows, n_t n, PT* out);

public:
	// Row count at or above which Gather() decodes the vector instead of reaching per row.
	//
	// Measured crossover (benchmark/result/subintsplit): a 1-section plan crosses just past 64 rows, a 3-section plan
	// between 64 and 256, because each extra section adds a scattered read per row but only one more sequential unffor
	// to the decode. 128 sits inside both crossovers. It stays a member rather than a constant because no single value
	// is right for every section count, and the benchmark sweeps it.
	n_t gather_decode_threshold {128};
	// Section s covers bits [bit_starts[s], bit_starts[s + 1) - 1], the last one up to 8 * sizeof(PT) - 1.
	vector<bw_t>           bit_starts;
	// Read back from the header; currently always EXP_FFOR_I64/I32 and 3 (see the file header comment).
	vector<OperatorToken> section_tokens;
	vector<uint8_t>       section_operand_counts;
	vector<SegmentView> bitpacked_segment_views;
	vector<SegmentView> base_segment_views;
	vector<SegmentView> bw_segment_views;
	SegmentView         header_segment_view;
	alignas(64) UT unffored_arr[CFG::VEC_SZ];
	alignas(64) PT data[CFG::VEC_SZ];
};

} // namespace fastlanes

#endif // FLS_EXPRESSION_SUBINTSPLIT_OPERATOR_HPP
