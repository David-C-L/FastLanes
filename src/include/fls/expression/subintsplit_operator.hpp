// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/include/fls/expression/subintsplit_operator.hpp
// ────────────────────────────────────────────────────────
#ifndef FLS_EXPRESSION_SUBINTSPLIT_OPERATOR_HPP
#define FLS_EXPRESSION_SUBINTSPLIT_OPERATOR_HPP

#include "fls/expression/subintsplit_selector.hpp"
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
 * The header is `uint8_t n_sections` followed by n_sections bytes of bit_start. bit_end is implied: section s ends
 * where section s+1 starts, and the last section ends at 8 * sizeof(PT) - 1.
 *
 * The header comes LAST on purpose. This is the only operator in the codebase with a *dynamic* segment count, so the
 * decoder cannot know how far to reach back until it has read K; putting the header at `cur_operand - 0` lets it read
 * K first, then index the 3K section segments below it, then rewind cur_operand by 3K + 1. Making it block-based keeps
 * PhysicalExpr::Size from charging the whole header against the sampled vectors during wizard selection.
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

public:
	// Section s covers bits [bit_starts[s], bit_starts[s + 1) - 1], the last one up to 8 * sizeof(PT) - 1.
	vector<bw_t>        bit_starts;
	vector<SegmentView> bitpacked_segment_views;
	vector<SegmentView> base_segment_views;
	vector<SegmentView> bw_segment_views;
	SegmentView         header_segment_view;
	alignas(64) UT unffored_arr[CFG::VEC_SZ];
	alignas(64) PT data[CFG::VEC_SZ];
};

} // namespace fastlanes

#endif // FLS_EXPRESSION_SUBINTSPLIT_OPERATOR_HPP
