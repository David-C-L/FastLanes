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
 * subintsplit::select_splits (a cheap, FFOR-width-only cost oracle -- unchanged by any of this). Each section is
 * then independently handed to subintsplit::select_section_encoding (subintsplit_section_selector.hpp), which picks
 * its own real codec -- Uncompressed, Constant, RLE, Dictionary, FFOR, FFOR_SLPATCH, or FrequencyPartition -- from
 * real measured cost on the section's own extracted-and-shifted values. This mirrors Nimble's own two-phase design:
 * the split DP's guess at a cheap codec only steers where the splits go and is discarded, the real per-section pick
 * happens later, on real per-section data. A section whose values carry independent semantic fields but no further
 * exploitable structure (a snowflake id's timestamp field, say) typically still lands on FFOR; a section that is
 * constant, low-cardinality, or mostly-one-value across the whole column can do better.
 *
 * Storage layout, in rowgroup segment order:
 *
 *     section 0 : however many segments its chosen codec's own encoder produces (see MoveSegments)
 *     ...
 *     section K-1 : ditto
 *     header                                      (block-based, written once in Finalize)
 *
 * The header is `uint8_t n_sections` followed by n_sections 4-byte records: bit_start (1B), the section's chosen
 * OperatorToken (2B), and its operand count (1B, i.e. how many of the segments below belong to this section). Both
 * columns are derived from the section's actual encoder rather than assumed, so this works uniformly whether a
 * section resolved to a single-operand codec (FFOR, Frequency, Constant, Uncompressed) or a multi-operand one
 * (FFOR_SLPATCH: unffor + slpatch; RLE: unffor + rsum + rle_map; Dictionary: unffor + dict).
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
	// One codec token + operand count per section, persisted in the header (see
	// subintsplit_section_selector.hpp for how the token is chosen).
	vector<OperatorToken> section_tokens;
	vector<uint8_t>       section_operand_counts;
	// One real child PhysicalExpr per section, built once at construction over that section's full
	// extracted-and-shifted values and driven per vector via PointTo + ExprExecutor::execute, exactly
	// like any ordinary top-level column's encoder. Constant sections are the one exception: Constant
	// carries its value via ColumnDescriptorT::max rather than a Segment (enc_constant_opr is a bare
	// marker struct with nothing to Encode/MoveSegments), which has no per-section analogue here, so
	// a Constant section instead gets a hand-written single-value Segment in
	// section_constant_segments and a null entry here.
	vector<sp<PhysicalExpr>> section_exprs;
	vector<up<Segment>>      section_constant_segments;
	// Each section_exprs[s] holds a TypedColumnView into section_rowgroups[s]'s one column, so both
	// must outlive it -- kept alive here rather than as constructor locals.
	vector<rowgroup_pt>           section_rowgroups;
	vector<up<ColumnDescriptorT>> section_col_descriptors;
	// Block-based, written once by Finalize.
	up<Segment> header_segment;
	// Set by PointTo, read by Encode -- ExprExecutor::execute needs the vector index explicitly.
	n_t cur_vec_idx {0};
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
	 * means decoding the 1024 values around it. A plain-FFOR section can do better, because unffor_single reaches a
	 * single value by position arithmetic - one value costs K scattered word reads, one per section, instead of K
	 * decoded vectors.
	 *
	 * Now that a section can be any of {Uncompressed, Constant, RLE, Dictionary, FFOR, FFOR_SLPATCH, Frequency}, this
	 * O(1) path is only sound for a section whose child is plain FFOR: unffor_single reads raw bit-packed words
	 * directly, which is correct only when nothing (a patch list, a dictionary indirection, a run-length map) sits
	 * between the packed bits and the logical value. Every other section -- including FFOR_SLPATCH, whose patch list
	 * make a single unffor_single read wrong for a patched position -- falls back to decoding that one section's
	 * vector through its child PhysicalExpr and indexing into it, costing one full section decode instead of a
	 * scattered read, but only for the sections that actually need it.
	 *
	 * Note what the fast path does and does not beat. Against decode-the-whole-value it wins by a wide margin
	 * whenever most/all sections are plain FFOR; against a single-section FFOR column read the same way it
	 * necessarily loses, because K reads cost more than one. Splitting buys compression on this path, not speed.
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

private:
	// Decodes one section's vec_idx-th vector into `out` (CFG::VEC_SZ values). The one place every
	// non-plain-FFOR section's child codec is actually driven; see the .cpp for the per-codec pull.
	void DecodeSectionVector(n_t s, n_t vec_idx, PT* out);

public:
	// Row count at or above which Gather() decodes the vector instead of reaching per row.
	//
	// Measured crossover (benchmark/result/subintsplit): a 1-section plan crosses just past 64 rows, a 3-section plan
	// between 64 and 256, because each extra section adds a scattered read per row but only one more sequential unffor
	// to the decode. 128 sits inside both crossovers. It stays a member rather than a constant because no single value
	// is right for every section count, and the benchmark sweeps it.
	n_t gather_decode_threshold {128};
	// Section s covers bits [bit_starts[s], bit_starts[s + 1) - 1], the last one up to 8 * sizeof(PT) - 1.
	vector<bw_t> bit_starts;
	// Read back from the header (see subintsplit_section_selector.hpp for how the encoder chose these).
	vector<OperatorToken> section_tokens;
	vector<uint8_t>       section_operand_counts;
	// One reconstructed child decode chain per section (e.g. [unffor] for FFOR, [unffor, slpatch] for
	// FFOR_SLPATCH, [unffor, rsum, rle_map] for RLE, [unffor, dict] for Dictionary), built once at
	// construction from the persisted token. Bulk Decode/Materialize/GatherDecoded drive these
	// generically; only plain-FFOR sections additionally get the O(1) fast path below.
	vector<sp<PhysicalExpr>> section_exprs;
	// True only for sections whose child is plain FFOR -- see the point/gather comment above.
	vector<bool> section_is_plain_ffor;
	// Raw segment views for the O(1) fast path, populated only where section_is_plain_ffor[s] is true.
	// SegmentView has no default constructor, so these hold up<SegmentView> rather than SegmentView
	// directly -- only plain-FFOR sections populate the first three, only Constant sections the
	// fourth, and every vector needs a null-able slot per section to stay indexable by s.
	vector<up<SegmentView>> bitpacked_segment_views;
	vector<up<SegmentView>> base_segment_views;
	vector<up<SegmentView>> bw_segment_views;
	// Constant sections are also O(1) (the value never varies), read directly from their own single
	// -value segment rather than through a child PhysicalExpr -- see the encoder's matching comment.
	vector<up<SegmentView>> section_constant_segment_views;
	SegmentView             header_segment_view;
	alignas(64) UT unffored_arr[CFG::VEC_SZ];
	alignas(64) PT data[CFG::VEC_SZ];
	// Scratch used to pull one section's decoded vector out of its child PhysicalExpr (bulk Decode
	// and the decode-then-index fallback for point/gather on non-plain-FFOR sections).
	up<TypedCol<PT>> section_scratch;
	// Per non-plain-FFOR, non-Constant section: the current vector's decoded values, refreshed once
	// per PointTo(vec_idx) and then reused for every ValueAt/GatherPointwise row in that vector --
	// one section decode amortized over however many rows are actually read, instead of once per row.
	vector<vector<PT>> section_decoded_cache;
};

} // namespace fastlanes

#endif // FLS_EXPRESSION_SUBINTSPLIT_OPERATOR_HPP
