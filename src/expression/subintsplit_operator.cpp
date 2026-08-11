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
#include "fls/expression/interpreter.hpp"
#include "fls/expression/physical_expression.hpp"
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
	bitpacked_segments.reserve(n_sections);
	base_segments.reserve(n_sections);
	bitwidth_segments.reserve(n_sections);
	for (n_t s {0}; s < n_sections; ++s) {
		bitpacked_segments.push_back(make_unique<Segment>());
		base_segments.push_back(make_unique<Segment>());
		bitwidth_segments.push_back(make_unique<Segment>());
	}
	header_segment = make_unique<Segment>();
	header_segment->MakeBlockBased();

	// Operand order must match MoveSegments exactly: the n-th operand token this operator pushes names the n-th
	// segment it hands over. The header is pushed last so the decoder can read it before it knows how far to reach.
	auto& [operator_tokens, operand_tokens] = *column_descriptor.encoding_rpn;
	for (n_t s {0}; s < n_sections; ++s) {
		operand_tokens.emplace_back(state.cur_operand++);
		operand_tokens.emplace_back(state.cur_operand++);
		operand_tokens.emplace_back(state.cur_operand++);
	}
	operand_tokens.emplace_back(state.cur_operand++);
}

template <typename PT>
void enc_subintsplit_opr<PT>::PointTo(const n_t vec_idx) {
	col_viewer.PointTo(vec_idx);
}

template <typename PT>
void enc_subintsplit_opr<PT>::Encode() {
	const PT* data       = col_viewer.Data();
	const n_t n_sections = sections.size();

	for (n_t s {0}; s < n_sections; ++s) {
		const bw_t bit_start = sections[s].bit_start;
		const UT   mask      = section_mask<UT>(sections[s].width());

		// Extract the section, and take its min/max in the same pass: FFOR stores value - min, so the packed width is
		// bit_width(max - min) rather than the section's declared width.
		UT min_val = static_cast<UT>(~static_cast<UT>(0));
		UT max_val = 0;
		for (n_t i {0}; i < CFG::VEC_SZ; ++i) {
			const UT value = static_cast<UT>((static_cast<UT>(data[i]) >> bit_start) & mask);
			section_arr[i] = value;
			min_val        = std::min<UT>(min_val, value);
			max_val        = std::max<UT>(max_val, value);
		}

		const bw_t bw = static_cast<bw_t>(std::bit_width(static_cast<UT>(max_val - min_val)));

		ffor::ffor(section_arr, bitpacked_arr, bw, &min_val);

		bitpacked_segments[s]->Flush(bitpacked_arr, calculate_bitpacked_vector_size(bw));
		base_segments[s]->Flush(&min_val, sizeof(UT));
		bitwidth_segments[s]->Flush(&bw, sizeof(bw_t));
	}
}

template <typename PT>
void enc_subintsplit_opr<PT>::Finalize() {
	vector<uint8_t> header;
	header.reserve(sections.size() + 1);
	header.push_back(static_cast<uint8_t>(sections.size()));
	for (const auto& section : sections) {
		header.push_back(static_cast<uint8_t>(section.bit_start));
	}
	header_segment->Flush(header.data(), header.size());
}

template <typename PT>
void enc_subintsplit_opr<PT>::MoveSegments(vector<up<Segment>>& segments) {
	for (n_t s {0}; s < sections.size(); ++s) {
		segments.push_back(std::move(bitpacked_segments[s]));
		segments.push_back(std::move(base_segments[s]));
		segments.push_back(std::move(bitwidth_segments[s]));
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
	for (n_t s {0}; s < n_sections; ++s) {
		bit_starts.push_back(static_cast<bw_t>(header[1 + s]));
	}

	// Section triples occupy the 3 * n_sections operands directly below the header.
	const n_t first_operand = state.cur_operand - (3 * n_sections);
	bitpacked_segment_views.reserve(n_sections);
	base_segment_views.reserve(n_sections);
	bw_segment_views.reserve(n_sections);
	for (n_t s {0}; s < n_sections; ++s) {
		const n_t triple = first_operand + (3 * s);
		bitpacked_segment_views.push_back(column_view.GetSegment(operand_segment_idx(column_view, triple + 0)));
		base_segment_views.push_back(column_view.GetSegment(operand_segment_idx(column_view, triple + 1)));
		bw_segment_views.push_back(column_view.GetSegment(operand_segment_idx(column_view, triple + 2)));
	}

	state.cur_operand -= ((3 * n_sections) + 1);
}

template <typename PT>
void dec_subintsplit_opr<PT>::PointTo(const n_t vec_idx) {
	const n_t n_sections = bit_starts.size();
	for (n_t s {0}; s < n_sections; ++s) {
		bitpacked_segment_views[s].PointTo(vec_idx);
		base_segment_views[s].PointTo(vec_idx);
		bw_segment_views[s].PointTo(vec_idx);
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
		const bw_t bw   = *reinterpret_cast<const bw_t*>(bw_segment_views[s].data);
		const UT   base = *reinterpret_cast<const UT*>(base_segment_views[s].data);

		unffor::unffor(reinterpret_cast<const UT*>(bitpacked_segment_views[s].data), unffored_arr, bw, &base);

		const bw_t shift = bit_starts[s];
		const bw_t width = static_cast<bw_t>((s + 1 < n_sections ? bit_starts[s + 1] : TOTAL_BITS) - bit_starts[s]);
		const UT   mask  = section_mask<UT>(width);

		// Section 0 writes each output element; every later section ORs into it. The branch is hoisted out of the
		// inner loop so both variants stay vectorisable.
		if (s == 0) {
			for (n_t i {0}; i < CFG::VEC_SZ; ++i) {
				output[i] = static_cast<UT>((unffored_arr[i] & mask) << shift);
			}
		} else {
			for (n_t i {0}; i < CFG::VEC_SZ; ++i) {
				output[i] |= static_cast<UT>((unffored_arr[i] & mask) << shift);
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
		const bw_t bw   = *reinterpret_cast<const bw_t*>(bw_segment_views[s].data);
		const UT   base = *reinterpret_cast<const UT*>(base_segment_views[s].data);
		const UT   section =
		    unffor_single<UT>(reinterpret_cast<const UT*>(bitpacked_segment_views[s].data), bw, base, idx);

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
		const bw_t  bw     = *reinterpret_cast<const bw_t*>(bw_segment_views[s].data);
		const UT    base   = *reinterpret_cast<const UT*>(base_segment_views[s].data);
		const auto* packed = reinterpret_cast<const UT*>(bitpacked_segment_views[s].data);
		const bw_t  shift  = bit_starts[s];
		const bw_t  width  = static_cast<bw_t>((s + 1 < n_sections ? bit_starts[s + 1] : TOTAL_BITS) - bit_starts[s]);
		const UT    mask   = section_mask<UT>(width);

		if (s == 0) {
			for (n_t i {0}; i < n; ++i) {
				output[i] = static_cast<UT>((unffor_single<UT>(packed, bw, base, rows[i]) & mask) << shift);
			}
		} else {
			for (n_t i {0}; i < n; ++i) {
				output[i] |= static_cast<UT>((unffor_single<UT>(packed, bw, base, rows[i]) & mask) << shift);
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
