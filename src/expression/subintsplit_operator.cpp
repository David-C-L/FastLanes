// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/expression/subintsplit_operator.cpp
// ────────────────────────────────────────────────────────
#include "fls/expression/subintsplit_operator.hpp"
#include "fls/cfg/cfg.hpp"
#include "fls/common/alias.hpp"
#include "fls/common/assert.hpp"
#include "fls/expression/data_type.hpp"
#include "fls/expression/decoding_operator.hpp"
#include "fls/expression/dict_expression.hpp"
#include "fls/expression/expression_executor.hpp"
#include "fls/expression/frequency_operator.hpp"
#include "fls/expression/interpreter.hpp"
#include "fls/expression/physical_expression.hpp"
#include "fls/expression/rle_expression.hpp"
#include "fls/expression/rsum_operator.hpp"
#include "fls/expression/slpatch_operator.hpp"
#include "fls/expression/subintsplit_section_selector.hpp"
#include "fls/ffor.hpp"
#include "fls/primitive/bitpack/bitpack.hpp"
#include "fls/primitive/copy/fls_copy.hpp"
#include "fls/primitive/unffor_single.hpp"
#include "fls/reader/column_view.hpp"
#include "fls/reader/segment.hpp"
#include "fls/std/vector.hpp"
#include "fls/table/rowgroup.hpp"
#include "fls/unffor.hpp"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <utility>

namespace fastlanes {

namespace {
/*--------------------------------------------------------------------------------------------------------------------*\
 * Section bit mask. Written as a branch on the full width because (UT{1} << (8 * sizeof(UT))) is undefined behaviour.
\*--------------------------------------------------------------------------------------------------------------------*/
template <typename UT>
constexpr UT section_mask(const bw_t width) {
	constexpr bw_t TOTAL_BITS = static_cast<bw_t>(sizeof(UT) * 8);
	return width >= TOTAL_BITS ? static_cast<UT>(~static_cast<UT>(0))
	                           : static_cast<UT>((static_cast<UT>(1) << width) - static_cast<UT>(1));
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * Resolve one of this operator's operand slots to the segment index it names.
\*--------------------------------------------------------------------------------------------------------------------*/
uint32_t operand_segment_idx(const ColumnView& column_view, const n_t operand) {
	const auto* operand_tokens = column_view.column_descriptor.encoding_rpn()->operand_tokens();
	return static_cast<uint32_t>((*operand_tokens)[static_cast<uint32_t>(operand)]);
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * Builds one section's decode chain directly from its persisted token, bypassing interpreter_decoding.cpp's
 * make_dec_X_expr layer: every one of those functions does
 * `state.cur_operand = column_view.column_descriptor.encoding_rpn()->operand_tokens()->size() - 1`, i.e. assumes
 * it's decoding the column's own sole top-level operator, not a nested child at some computed offset. The concrete
 * operator constructors underneath that layer don't have this problem -- they read/decrement whatever
 * `state.cur_operand` the caller hands them -- so this calls those directly, seeded with this section's own last
 * operand index (see the .cpp header comment on operand indexing convention), exactly the same sequence
 * make_dec_rle_expr/make_dec_dict_ffor_expr/etc. use internally.
\*--------------------------------------------------------------------------------------------------------------------*/
template <typename PT>
sp<PhysicalExpr>
build_section_decode_expr(const OperatorToken token, const ColumnView& column_view, InterpreterState& state) {
	using UT  = make_unsigned_t<PT>;
	auto expr = make_shared<PhysicalExpr>();

	switch (token) {
	case OperatorToken::EXP_UNCOMPRESSED_I64:
	case OperatorToken::EXP_UNCOMPRESSED_I32: {
		// dec_uncompressed_opr takes an already-resolved segment index rather than InterpreterState.
		const auto segment_idx = operand_segment_idx(column_view, state.cur_operand);
		expr->operators.emplace_back(
		    dec_physical_operator {make_shared<dec_uncompressed_opr<PT>>(column_view, segment_idx)});
		state.cur_operand -= 1;
		break;
	}
	case OperatorToken::EXP_FFOR_I64:
	case OperatorToken::EXP_FFOR_I32: {
		expr->operators.emplace_back(dec_physical_operator {make_shared<dec_unffor_opr<UT>>(column_view, state)});
		break;
	}
	case OperatorToken::EXP_FFOR_SLPATCH_I64:
	case OperatorToken::EXP_FFOR_SLPATCH_I32: {
		expr->operators.emplace_back(dec_physical_operator {make_shared<dec_unffor_opr<UT>>(column_view, state)});
		expr->operators.emplace_back(
		    dec_physical_operator {make_shared<dec_slpatch_opr<PT>>(*expr, column_view, state)});
		break;
	}
	case OperatorToken::EXP_RLE_I64_U16:
	case OperatorToken::EXP_RLE_I32_U16: {
		expr->operators.emplace_back(dec_physical_operator {make_shared<dec_unffor_opr<u16_pt>>(column_view, state)});
		expr->operators.emplace_back(
		    dec_physical_operator {make_shared<dec_rsum_opr<u16_pt>>(*expr, column_view, state)});
		expr->operators.emplace_back(
		    dec_physical_operator {make_shared<dec_rle_map_opr<PT, u16_pt>>(*expr, column_view, state)});
		break;
	}
	case OperatorToken::EXP_DICT_I64_FFOR_U08:
	case OperatorToken::EXP_DICT_I32_FFOR_U08: {
		expr->operators.emplace_back(dec_physical_operator {make_shared<dec_unffor_opr<u08_pt>>(column_view, state)});
		expr->operators.emplace_back(
		    dec_physical_operator {make_shared<dec_dict_opr<PT, u08_pt>>(*expr, column_view, state)});
		break;
	}
	case OperatorToken::EXP_DICT_I64_FFOR_U16:
	case OperatorToken::EXP_DICT_I32_FFOR_U16: {
		expr->operators.emplace_back(dec_physical_operator {make_shared<dec_unffor_opr<u16_pt>>(column_view, state)});
		expr->operators.emplace_back(
		    dec_physical_operator {make_shared<dec_dict_opr<PT, u16_pt>>(*expr, column_view, state)});
		break;
	}
	case OperatorToken::EXP_DICT_I64_FFOR_U32:
	case OperatorToken::EXP_DICT_I32_FFOR_U32: {
		expr->operators.emplace_back(dec_physical_operator {make_shared<dec_unffor_opr<u32_pt>>(column_view, state)});
		expr->operators.emplace_back(
		    dec_physical_operator {make_shared<dec_dict_opr<PT, u32_pt>>(*expr, column_view, state)});
		break;
	}
	case OperatorToken::EXP_FREQUENCY_I64:
	case OperatorToken::EXP_FREQUENCY_I32: {
		expr->operators.emplace_back(
		    dec_physical_operator {make_shared<dec_frequency_opr<PT>>(*expr, column_view, state)});
		break;
	}
	default:
		FLS_UNREACHABLE()
	}

	ExprExecutor::CountOperator(*expr);
	return expr;
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * Pulls one already-smart_execute'd section's decoded vector out of its child chain's final operator, mirroring
 * materializer.cpp's material_visitor (which this can't reuse directly -- it's file-local there, and keyed by
 * column index into a real multi-column Rowgroup rather than a bare scratch buffer).
\*--------------------------------------------------------------------------------------------------------------------*/
template <typename PT>
struct section_pull_visitor {
	explicit section_pull_visitor(up<TypedCol<PT>>& a_typed_col)
	    : typed_col(a_typed_col) {
	}

	void operator()(const sp<dec_uncompressed_opr<PT>>& opr) const {
		for (n_t idx {0}; idx < CFG::VEC_SZ; ++idx) {
			typed_col->data.push_back(opr->Data()[idx]);
		}
	}
	void operator()(const sp<dec_slpatch_opr<PT>>& opr) const {
		opr->Materialize(0, *typed_col);
	}
	void operator()(const sp<dec_frequency_opr<PT>>& opr) const {
		opr->Materialize(0, *typed_col);
	}
	template <typename INDEX_PT>
	void operator()(const sp<dec_rle_map_opr<PT, INDEX_PT>>& opr) const {
		opr->Decode(0, typed_col->data);
	}
	template <typename INDEX_PT>
	void operator()(const sp<dec_dict_opr<PT, INDEX_PT>>& opr) const {
		const auto* key_p   = opr->Keys();
		const auto* index_p = opr->Index();
		for (n_t idx {0}; idx < CFG::VEC_SZ; ++idx) {
			typed_col->data.push_back(key_p[index_p[idx]]);
		}
	}
	void operator()(const auto& /*opr*/) const {FLS_UNREACHABLE()}

	up<TypedCol<PT>>& typed_col;
};

} // namespace

/*--------------------------------------------------------------------------------------------------------------------*\
 * enc_subintsplit_opr
\*--------------------------------------------------------------------------------------------------------------------*/
template <typename PT>
enc_subintsplit_opr<PT>::enc_subintsplit_opr(const PhysicalExpr& /*expr*/,
                                             const col_pt&      col,
                                             ColumnDescriptorT& column_descriptor,
                                             InterpreterState&  state)
    : col_viewer(col) {

	const n_t n_tuples = col_viewer.GetNTuples();
	const n_t n_vec    = (n_tuples + CFG::VEC_SZ - 1) / CFG::VEC_SZ;

	// Pick the split once, here, from evenly spaced vectors of the whole column. The selector caps how many vectors it
	// measures anyway, so handing it only the sample avoids flattening the entire column to choose 8 of its vectors.
	const auto config    = subintsplit::default_selector_config();
	const n_t  n_sampled = std::min<n_t>(std::max<n_t>(n_vec, 1), std::max<n_t>(config.max_sampled_vectors, 1));

	vector<span<const PT>> sampled_vectors;
	sampled_vectors.reserve(n_sampled);
	for (n_t i {0}; i < n_sampled; ++i) {
		const n_t vec_idx = i * std::max<n_t>(n_vec, 1) / n_sampled;
		const n_t offset  = vec_idx * CFG::VEC_SZ;
		if (offset >= n_tuples) {
			continue;
		}
		const n_t len = std::min<n_t>(CFG::VEC_SZ, n_tuples - offset);
		sampled_vectors.push_back(span<const PT> {col_viewer.Data(vec_idx), len});
	}

	auto plan = subintsplit::select_splits<PT>(span<const span<const PT>> {sampled_vectors}, n_vec, config);
	sections  = std::move(plan.segments);
	FLS_ASSERT_FALSE(sections.empty())

	const n_t n_sections = sections.size();

	section_tokens.reserve(n_sections);
	section_operand_counts.reserve(n_sections);
	section_exprs.reserve(n_sections);
	section_rowgroups.reserve(n_sections);
	section_col_descriptors.reserve(n_sections);

	const DataType section_data_type = std::is_same_v<PT, i64_pt> ? DataType::INT64 : DataType::INT32;

	for (n_t s {0}; s < n_sections; ++s) {
		const bw_t bit_start = sections[s].bit_start;
		const UT   mask      = section_mask<UT>(sections[s].width());

		// Choose this section's codec from the same sample already gathered for the split DP above --
		// real per-candidate TryExpr cost on the full column would multiply the DP's own sampling
		// budget by (candidates x n_sections), and pool comparisons only need relative cost between
		// candidates on the same sample, not absolute size against the real column.
		vector<PT> section_sample;
		section_sample.reserve(sampled_vectors.size() * CFG::VEC_SZ);
		for (const auto& vec : sampled_vectors) {
			for (const PT value : vec) {
				section_sample.push_back(static_cast<PT>((static_cast<UT>(value) >> bit_start) & mask));
			}
		}
		const OperatorToken token =
		    subintsplit::select_section_encoding<PT>(section_sample, std::max<n_t>(sampled_vectors.size(), 1));
		section_tokens.push_back(token);

		// Build the section's real child encoder over its FULL extracted-and-shifted values (every
		// vector, not just the sample above) -- this is what Encode() actually drives per vector below.
		auto section_col = make_unique<TypedCol<PT>>();
		section_col->data.reserve(n_tuples);
		section_col->null_map_arr.assign(n_tuples, 0);
		for (n_t v {0}; v < n_vec; ++v) {
			const n_t offset   = v * CFG::VEC_SZ;
			const n_t len      = std::min<n_t>(CFG::VEC_SZ, n_tuples - offset);
			const PT* vec_data = col_viewer.Data(v);
			for (n_t i {0}; i < len; ++i) {
				section_col->data.push_back(static_cast<PT>((static_cast<UT>(vec_data[i]) >> bit_start) & mask));
			}
		}

		// Mirrors Rowgroup::PopulateBiMap's exact loop (see subintsplit_section_selector.hpp for why):
		// codecs like Frequency/Dictionary/Constant read these stats directly off the column, not off
		// anything select_section_encoding computed above -- that was a different, sample-only column.
		auto& section_stats = section_col->m_stats;
		for (const PT value : section_col->data) {
			if (!section_stats.bimap_frequency.contains_value(value)) {
				const n_t next_idx = section_stats.bimap_frequency.size();
				section_stats.bimap_frequency.insert(next_idx, value);
			} else {
				const n_t existing_key = section_stats.bimap_frequency.get_key(value);
				section_stats.bimap_frequency.insert(existing_key, value);
			}
			section_stats.min = std::min(section_stats.min, value);
			section_stats.max = std::max(section_stats.max, value);
		}
		section_stats.n_nulls = 0;

		const bool is_constant_token =
		    token == OperatorToken::EXP_CONSTANT_I64 || token == OperatorToken::EXP_CONSTANT_I32;

		if (is_constant_token) {
			// enc_constant_opr is a bare marker struct (no Encode/MoveSegments; the value normally
			// travels via ColumnDescriptorT::max, which has no per-section slot here) -- write the
			// single value as our own block-based Segment instead. section_col->data is guaranteed
			// non-empty (n_tuples > 0) and, since select_section_encoding only reaches this token via
			// its constant pre-pass, every entry is the same value.
			auto constant_segment = make_unique<Segment>();
			constant_segment->MakeBlockBased();
			const PT constant_value = section_col->data[0];
			constant_segment->Flush(&constant_value, sizeof(PT));

			section_exprs.push_back(nullptr);
			section_constant_segments.push_back(std::move(constant_segment));
			section_rowgroups.emplace_back();
			section_col_descriptors.push_back(nullptr);
			section_operand_counts.push_back(1);
		} else {
			rowgroup_pt section_rowgroup;
			section_rowgroup.emplace_back(std::move(section_col));

			auto section_cd          = make_unique<ColumnDescriptorT>();
			section_cd->data_type    = section_data_type;
			section_cd->idx          = 0;
			section_cd->max          = make_unique<BinaryValueT>();
			section_cd->encoding_rpn = make_unique<RPNT>();
			section_cd->encoding_rpn->operator_tokens.push_back(token);

			InterpreterState section_state;
			sp<PhysicalExpr> section_expr =
			    Interpreter::Encoding::Interpret(*section_cd, section_rowgroup, section_state);

			// operand_tokens.size() (not section_state.cur_operand) is the authoritative segment
			// count: most operators grow both in lockstep via state.cur_operand++, but
			// enc_uncompressed_opr pushes a single hardcoded placeholder operand token without ever
			// touching state.cur_operand (harmless for a real top-level column, where it's always
			// operand 0 anyway -- but section_state.cur_operand would silently read back 0 here).
			const n_t operand_count = section_cd->encoding_rpn->operand_tokens.size();
			FLS_ASSERT_G(operand_count, 0)
			FLS_ASSERT_LE(operand_count, std::numeric_limits<uint8_t>::max())
			section_operand_counts.push_back(static_cast<uint8_t>(operand_count));

			section_exprs.push_back(std::move(section_expr));
			section_constant_segments.push_back(nullptr);
			section_rowgroups.push_back(std::move(section_rowgroup));
			section_col_descriptors.push_back(std::move(section_cd));
		}
	}

	header_segment = make_unique<Segment>();
	header_segment->MakeBlockBased();

	// Operand order must match MoveSegments exactly: the n-th operand token this operator pushes names the n-th
	// segment it hands over. The header is pushed last so the decoder can read it before it knows how far to reach.
	auto& [operator_tokens, operand_tokens] = *column_descriptor.encoding_rpn;
	for (n_t s {0}; s < n_sections; ++s) {
		for (n_t k {0}; k < section_operand_counts[s]; ++k) {
			operand_tokens.emplace_back(state.cur_operand++);
		}
	}
	operand_tokens.emplace_back(state.cur_operand++);
}

template <typename PT>
void enc_subintsplit_opr<PT>::PointTo(const n_t vec_idx) {
	col_viewer.PointTo(vec_idx);
	cur_vec_idx = vec_idx;
	for (auto& section_expr : section_exprs) {
		if (section_expr) { // null for Constant sections, which have no per-vector work at all
			section_expr->PointTo(vec_idx);
		}
	}
}

template <typename PT>
void enc_subintsplit_opr<PT>::Encode() {
	for (auto& section_expr : section_exprs) {
		if (section_expr) {
			ExprExecutor::execute(*section_expr, cur_vec_idx);
		}
	}
}

template <typename PT>
void enc_subintsplit_opr<PT>::Finalize() {
	for (auto& section_expr : section_exprs) {
		if (section_expr) {
			section_expr->Finalize();
		}
	}
	// Constant sections' single value segment was already Flushed at construction time (the whole
	// column's data is already known then; there's no per-vector accumulation to wait for).

	const n_t n_sections = sections.size();

	vector<uint8_t> header;
	header.reserve(1 + n_sections * 4);
	header.push_back(static_cast<uint8_t>(n_sections));
	for (n_t s {0}; s < n_sections; ++s) {
		const auto token = static_cast<uint16_t>(section_tokens[s]);
		header.push_back(static_cast<uint8_t>(sections[s].bit_start));
		header.push_back(static_cast<uint8_t>(token & 0xFF));
		header.push_back(static_cast<uint8_t>((token >> 8) & 0xFF));
		header.push_back(section_operand_counts[s]);
	}
	header_segment->Flush(header.data(), header.size());
}

template <typename PT>
void enc_subintsplit_opr<PT>::MoveSegments(vector<up<Segment>>& segments) {
	for (n_t s {0}; s < section_exprs.size(); ++s) {
		if (section_exprs[s]) {
			section_exprs[s]->MoveSegments(segments);
		} else {
			segments.push_back(std::move(section_constant_segments[s]));
		}
	}
	segments.push_back(std::move(header_segment));
}

template struct enc_subintsplit_opr<i64_pt>;
template struct enc_subintsplit_opr<i32_pt>;

/*--------------------------------------------------------------------------------------------------------------------*\
 * dec_subintsplit_opr
\*--------------------------------------------------------------------------------------------------------------------*/
template <typename PT>
dec_subintsplit_opr<PT>::dec_subintsplit_opr(PhysicalExpr& /*physical_expr*/,
                                             const ColumnView& column_view,
                                             InterpreterState& state)
    : header_segment_view(column_view.GetSegment(operand_segment_idx(column_view, state.cur_operand))) {

	// The header names the section count, so it has to be read before the section segments can even be located.
	header_segment_view.PointTo(0);
	const auto* header     = reinterpret_cast<const uint8_t*>(header_segment_view.data);
	const n_t   n_sections = header[0];
	FLS_ASSERT_NOT_ZERO(n_sections)

	bit_starts.reserve(n_sections);
	section_tokens.reserve(n_sections);
	section_operand_counts.reserve(n_sections);
	for (n_t s {0}; s < n_sections; ++s) {
		const uint8_t* record = header + 1 + (4 * s);
		bit_starts.push_back(static_cast<bw_t>(record[0]));
		const auto token = static_cast<uint16_t>(record[1]) | (static_cast<uint16_t>(record[2]) << 8);
		section_tokens.push_back(static_cast<OperatorToken>(token));
		section_operand_counts.push_back(record[3]);
	}

	const n_t operand_total = [&] {
		n_t total {0};
		for (const uint8_t count : section_operand_counts) {
			total += count;
		}
		return total;
	}();

	// Section records occupy the operand_total operands directly below the header, at offsets given by the running
	// prefix sum of each section's own operand count.
	const n_t first_operand = state.cur_operand - operand_total;

	section_exprs.assign(n_sections, nullptr);
	section_is_plain_ffor.assign(n_sections, false);
	bitpacked_segment_views.resize(n_sections);
	base_segment_views.resize(n_sections);
	bw_segment_views.resize(n_sections);
	section_constant_segment_views.resize(n_sections);
	section_decoded_cache.resize(n_sections);

	n_t running_offset {0};
	for (n_t s {0}; s < n_sections; ++s) {
		const n_t           section_first_operand = first_operand + running_offset;
		const OperatorToken token                 = section_tokens[s];

		if (token == OperatorToken::EXP_CONSTANT_I64 || token == OperatorToken::EXP_CONSTANT_I32) {
			section_constant_segment_views[s] = make_unique<SegmentView>(
			    column_view.GetSegment(operand_segment_idx(column_view, section_first_operand)));
			section_constant_segment_views[s]->PointTo(0); // block-based: one value for the whole column
		} else if (token == OperatorToken::EXP_FFOR_I64 || token == OperatorToken::EXP_FFOR_I32) {
			section_is_plain_ffor[s] = true;
			// Order matches enc_ffor_opr's own MoveSegments (bitpacked, bitwidth, base) -- see
			// extract_segments_visitor's FFOR case in physical_expression.cpp -- since this section
			// now goes through the real enc_ffor_opr via Interpreter::Encoding::Interpret rather than
			// a hand-written bitpacked/base/bitwidth triple.
			bitpacked_segment_views[s] = make_unique<SegmentView>(
			    column_view.GetSegment(operand_segment_idx(column_view, section_first_operand + 0)));
			bw_segment_views[s] = make_unique<SegmentView>(
			    column_view.GetSegment(operand_segment_idx(column_view, section_first_operand + 1)));
			base_segment_views[s] = make_unique<SegmentView>(
			    column_view.GetSegment(operand_segment_idx(column_view, section_first_operand + 2)));
		} else {
			section_decoded_cache[s].resize(CFG::VEC_SZ);

			// build_section_decode_expr's concrete constructors read `state.cur_operand` as the ABSOLUTE
			// index of the LAST operand they consume (dec_unffor_opr etc.'s own convention -- see this
			// operator's constructor above, which uses the identical convention for itself), not a
			// section-local count -- seed it with this section's own last operand index accordingly.
			InterpreterState local_state;
			local_state.cur_operand = section_first_operand + section_operand_counts[s] - 1;
			section_exprs[s]        = build_section_decode_expr<PT>(token, column_view, local_state);
		}

		running_offset += section_operand_counts[s];
	}

	section_scratch = make_unique<TypedCol<PT>>();

	state.cur_operand -= (operand_total + 1);
}

template <typename PT>
void dec_subintsplit_opr<PT>::DecodeSectionVector(const n_t s, const n_t vec_idx, PT* out) {
	auto& expr = *section_exprs[s];
	expr.PointTo(vec_idx);
	ExprExecutor::smart_execute(expr, vec_idx);

	section_scratch->data.clear();
	visit_dec(section_pull_visitor<PT> {section_scratch}, expr.operators.back());
	FLS_ASSERT_E(section_scratch->data.size(), CFG::VEC_SZ)
	copy<PT>(section_scratch->data.data(), out);
}

template <typename PT>
void dec_subintsplit_opr<PT>::PointTo(const n_t vec_idx) {
	const n_t n_sections = bit_starts.size();
	for (n_t s {0}; s < n_sections; ++s) {
		if (section_is_plain_ffor[s]) {
			bitpacked_segment_views[s]->PointTo(vec_idx);
			base_segment_views[s]->PointTo(vec_idx);
			bw_segment_views[s]->PointTo(vec_idx);
		} else if (section_tokens[s] != OperatorToken::EXP_CONSTANT_I64 &&
		           section_tokens[s] != OperatorToken::EXP_CONSTANT_I32) {
			// Constant needs no per-vector positioning at all; every other non-plain-FFOR section's
			// vector is decoded once here and cached, so ValueAt/GatherPointwise -- which don't take
			// vec_idx themselves, matching the plain-FFOR path's existing contract -- can just index it.
			DecodeSectionVector(s, vec_idx, section_decoded_cache[s].data());
		}
	}
}

template <typename PT>
void dec_subintsplit_opr<PT>::Decode(const n_t vec_idx) {
	constexpr bw_t TOTAL_BITS = static_cast<bw_t>(sizeof(UT) * 8);

	// The decode path never calls PhysicalExpr::PointTo — RowgroupReader::get_chunk goes straight to smart_execute —
	// so every decode operator points its own segment views, as dec_unffor_opr and dec_frequency_opr do.
	PointTo(vec_idx);

	const n_t n_sections = bit_starts.size();
	auto*     output     = reinterpret_cast<UT*>(data);

	for (n_t s {0}; s < n_sections; ++s) {
		const bw_t shift = bit_starts[s];
		const bw_t width = static_cast<bw_t>((s + 1 < n_sections ? bit_starts[s + 1] : TOTAL_BITS) - bit_starts[s]);
		const UT   mask  = section_mask<UT>(width);

		const UT* section_values;
		if (section_is_plain_ffor[s]) {
			const bw_t bw   = *reinterpret_cast<const bw_t*>(bw_segment_views[s]->data);
			const UT   base = *reinterpret_cast<const UT*>(base_segment_views[s]->data);
			unffor::unffor(reinterpret_cast<const UT*>(bitpacked_segment_views[s]->data), unffored_arr, bw, &base);
			section_values = unffored_arr;
		} else if (section_tokens[s] == OperatorToken::EXP_CONSTANT_I64 ||
		           section_tokens[s] == OperatorToken::EXP_CONSTANT_I32) {
			// PointTo already positioned this at the header/segment level; broadcast the one value.
			const UT constant_value =
			    static_cast<UT>(*reinterpret_cast<const PT*>(section_constant_segment_views[s]->data));
			for (n_t i {0}; i < CFG::VEC_SZ; ++i) {
				unffored_arr[i] = constant_value;
			}
			section_values = unffored_arr;
		} else {
			section_values = reinterpret_cast<const UT*>(section_decoded_cache[s].data());
		}

		// Section 0 writes each output element; every later section ORs into it. The branch is hoisted out of the
		// inner loop so both variants stay vectorisable.
		if (s == 0) {
			for (n_t i {0}; i < CFG::VEC_SZ; ++i) {
				output[i] = static_cast<UT>((section_values[i] & mask) << shift);
			}
		} else {
			for (n_t i {0}; i < CFG::VEC_SZ; ++i) {
				output[i] |= static_cast<UT>((section_values[i] & mask) << shift);
			}
		}
	}
}

template <typename PT>
PT dec_subintsplit_opr<PT>::ValueAt(const n_t idx) const {
	constexpr bw_t TOTAL_BITS = static_cast<bw_t>(sizeof(UT) * 8);

	const n_t n_sections = bit_starts.size();
	UT        value {0};

	for (n_t s {0}; s < n_sections; ++s) {
		UT section;
		if (section_is_plain_ffor[s]) {
			const bw_t bw   = *reinterpret_cast<const bw_t*>(bw_segment_views[s]->data);
			const UT   base = *reinterpret_cast<const UT*>(base_segment_views[s]->data);
			section = unffor_single<UT>(reinterpret_cast<const UT*>(bitpacked_segment_views[s]->data), bw, base, idx);
		} else if (section_tokens[s] == OperatorToken::EXP_CONSTANT_I64 ||
		           section_tokens[s] == OperatorToken::EXP_CONSTANT_I32) {
			section = static_cast<UT>(*reinterpret_cast<const PT*>(section_constant_segment_views[s]->data));
		} else {
			section = static_cast<UT>(section_decoded_cache[s][idx]);
		}

		const bw_t shift = bit_starts[s];
		const bw_t width = static_cast<bw_t>((s + 1 < n_sections ? bit_starts[s + 1] : TOTAL_BITS) - bit_starts[s]);
		value |= static_cast<UT>((section & section_mask<UT>(width)) << shift);
	}

	return static_cast<PT>(value);
}

template <typename PT>
PT dec_subintsplit_opr<PT>::PointAccess(const n_t vec_idx, const n_t idx) {
	PointTo(vec_idx);
	return ValueAt(idx);
}

template <typename PT>
void dec_subintsplit_opr<PT>::GatherPointwise(const n_t vec_idx, const idx_t* rows, const n_t n, PT* out) {
	constexpr bw_t TOTAL_BITS = static_cast<bw_t>(sizeof(UT) * 8);

	PointTo(vec_idx);

	const n_t n_sections = bit_starts.size();
	auto*     output     = reinterpret_cast<UT*>(out);

	// Sections outermost so each section's bit width, base and buffer pointer are loaded once for the whole gather
	// rather than once per row.
	for (n_t s {0}; s < n_sections; ++s) {
		const bw_t shift = bit_starts[s];
		const bw_t width = static_cast<bw_t>((s + 1 < n_sections ? bit_starts[s + 1] : TOTAL_BITS) - bit_starts[s]);
		const UT   mask  = section_mask<UT>(width);

		if (section_is_plain_ffor[s]) {
			const bw_t  bw     = *reinterpret_cast<const bw_t*>(bw_segment_views[s]->data);
			const UT    base   = *reinterpret_cast<const UT*>(base_segment_views[s]->data);
			const auto* packed = reinterpret_cast<const UT*>(bitpacked_segment_views[s]->data);
			if (s == 0) {
				for (n_t i {0}; i < n; ++i) {
					output[i] = static_cast<UT>((unffor_single<UT>(packed, bw, base, rows[i]) & mask) << shift);
				}
			} else {
				for (n_t i {0}; i < n; ++i) {
					output[i] |= static_cast<UT>((unffor_single<UT>(packed, bw, base, rows[i]) & mask) << shift);
				}
			}
		} else if (section_tokens[s] == OperatorToken::EXP_CONSTANT_I64 ||
		           section_tokens[s] == OperatorToken::EXP_CONSTANT_I32) {
			const UT value = static_cast<UT>(*reinterpret_cast<const PT*>(section_constant_segment_views[s]->data));
			const UT contribution = static_cast<UT>((value & mask) << shift);
			if (s == 0) {
				for (n_t i {0}; i < n; ++i) {
					output[i] = contribution;
				}
			} else {
				for (n_t i {0}; i < n; ++i) {
					output[i] |= contribution;
				}
			}
		} else {
			const auto& cache = section_decoded_cache[s];
			if (s == 0) {
				for (n_t i {0}; i < n; ++i) {
					output[i] = static_cast<UT>((static_cast<UT>(cache[rows[i]]) & mask) << shift);
				}
			} else {
				for (n_t i {0}; i < n; ++i) {
					output[i] |= static_cast<UT>((static_cast<UT>(cache[rows[i]]) & mask) << shift);
				}
			}
		}
	}
}

template <typename PT>
void dec_subintsplit_opr<PT>::GatherDecoded(const n_t vec_idx, const idx_t* rows, const n_t n, PT* out) {
	Decode(vec_idx);
	for (n_t i {0}; i < n; ++i) {
		out[i] = data[rows[i]];
	}
}

template <typename PT>
void dec_subintsplit_opr<PT>::Gather(const n_t vec_idx, const idx_t* rows, const n_t n, PT* out) {
	if (n >= gather_decode_threshold) {
		GatherDecoded(vec_idx, rows, n, out);
	} else {
		GatherPointwise(vec_idx, rows, n, out);
	}
}

template <typename PT>
void dec_subintsplit_opr<PT>::Materialize(const n_t vec_idx, TypedCol<PT>& typed_col) {
	typed_col.data.resize(typed_col.data.size() + CFG::VEC_SZ);
	copy<PT>(data, typed_col.data.data() + (vec_idx * CFG::VEC_SZ));
}

template struct dec_subintsplit_opr<i64_pt>;
template struct dec_subintsplit_opr<i32_pt>;

} // namespace fastlanes
