// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/include/fls/expression/subintsplit_section_selector.hpp
// ────────────────────────────────────────────────────────
//
// Chooses a codec for one SubIntSplit section, independently of and after
// `subintsplit::select_splits` has already fixed the bit-range boundaries. Mirrors Nimble's own
// two-phase design (SubIntSplitSelector.h's DP guesses a per-range codec only to steer where
// splits go and then discards that guess; the real per-section codec comes later, from
// `encodeNested` -- Nimble's ordinary top-level encoder-selection machinery run on the real
// extracted section values): `subintsplit_selector.hpp`'s DP cost oracle is untouched by this
// file, and this selector reuses FastLanes' own real, measured-cost wizard machinery
// (TryExpr/ChooseBestExpr, wizard_internal.hpp) the same way the top-level wizard already does
// for whole columns -- just pointed at one section's extracted values instead.
//
// Candidate pool: the same seven codecs already measured in bench_subintsplit_limited --
// Uncompressed, RLE, Dictionary (3 index widths), FFOR, FFOR_SLPATCH, FrequencyPartition -- plus
// an explicit Constant pre-pass (gather_statistics + constant_visit, mirroring the top-level
// wizard's own pre-pass order). SubIntSplit itself is excluded (no self-recursion). Constant is
// reachable here even though it cannot be forced via Connection::force_schema_pool: this file
// does not use force_schema_pool at all, it runs the same constant-detection pre-pass the
// top-level wizard runs before its own pool search, directly, on the section's own values.
#ifndef FLS_EXPRESSION_SUBINTSPLIT_SECTION_SELECTOR_HPP
#define FLS_EXPRESSION_SUBINTSPLIT_SECTION_SELECTOR_HPP

#include "fls/common/alias.hpp"
#include "fls/common/assert.hpp"
#include "fls/connection.hpp"
#include "fls/footer/operator_token_generated.h"
#include "fls/std/type_traits.hpp"
#include "fls/std/vector.hpp"
#include "fls/table/rowgroup.hpp"
#include "fls/wizard/wizard_internal.hpp"

namespace fastlanes {
namespace subintsplit {

namespace detail {

template <typename PT>
constexpr DataType section_data_type() {
	static_assert(std::is_same_v<PT, i64_pt> || std::is_same_v<PT, i32_pt>,
	              "SubIntSplit section selection currently supports i64/i32 columns only, matching "
	              "SubIntSplit's own scope");
	if constexpr (std::is_same_v<PT, i64_pt>) {
		return DataType::INT64;
	} else {
		return DataType::INT32;
	}
}

// Mirrors bench_subintsplit_limited's forced-row set, minus SubIntSplit itself (no
// self-recursion, matching Nimble's own exclusion of SubIntSplit from encodeNested's eligible
// set) and minus Constant (handled by the pre-pass below, since it can never be reached through
// this kind of pool search -- see docs/subintsplit.md).
template <typename PT>
vector<OperatorToken> section_candidate_pool() {
	if constexpr (std::is_same_v<PT, i64_pt>) {
		return {
		    OperatorToken::EXP_UNCOMPRESSED_I64,
		    OperatorToken::EXP_RLE_I64_U16,
		    OperatorToken::EXP_DICT_I64_FFOR_U08,
		    OperatorToken::EXP_DICT_I64_FFOR_U16,
		    OperatorToken::EXP_DICT_I64_FFOR_U32,
		    OperatorToken::EXP_FFOR_I64,
		    OperatorToken::EXP_FFOR_SLPATCH_I64,
		    OperatorToken::EXP_FREQUENCY_I64,
		};
	} else {
		return {
		    OperatorToken::EXP_UNCOMPRESSED_I32,
		    OperatorToken::EXP_RLE_I32_U16,
		    OperatorToken::EXP_DICT_I32_FFOR_U08,
		    OperatorToken::EXP_DICT_I32_FFOR_U16,
		    OperatorToken::EXP_DICT_I32_FFOR_U32,
		    OperatorToken::EXP_FFOR_I32,
		    OperatorToken::EXP_FFOR_SLPATCH_I32,
		    OperatorToken::EXP_FREQUENCY_I32,
		};
	}
}

} // namespace detail

// `section_values`: one section's already bit-extracted-and-shifted values (see
// enc_subintsplit_opr::Encode's per-vector extraction loop -- these are exactly its
// `section_arr` values, just gathered across the whole column instead of one vector), reinterpreted
// as PT bit-for-bit. Reinterpreting as signed PT is safe here even for a full-width section (the
// n_sections==1 case): every codec eligible for EXP_*_I64/I32 tokens already casts to its
// unsigned physical representation internally (e.g. enc_ffor_opr is only instantiated for
// unsigned physical types) before doing any width/comparison arithmetic, the same path real
// columns with values spanning the full signed range (e.g. snowflake IDs) already exercise.
//
// `n_vec`: the column's total vector count (matches RowgroupDescriptorT::m_n_vec elsewhere),
// used only to size TryExpr's sampling layout -- with a default Connection (sample_size == 0)
// TryExpr measures every vector, not a subsample.
template <typename PT>
[[nodiscard]] OperatorToken select_section_encoding(const vector<PT>& section_values, const n_t n_vec) {
	FLS_ASSERT_G(section_values.size(), 0)

	auto typed_col = make_unique<TypedCol<PT>>();
	typed_col->data = section_values;
	typed_col->null_map_arr.assign(section_values.size(), 0);

	// Mirrors Rowgroup::PopulateBiMap's exact loop: real columns reach TryExpr/constant_visit
	// through that same population, so this selector must populate stats the same way for the
	// codecs it evaluates (Frequency's most-frequent-value, Dictionary's cardinality, Constant's
	// bimap-size-one check) to behave identically to how they'd behave on a real top-level column.
	auto& stats = typed_col->m_stats;
	for (const PT value : section_values) {
		if (!stats.bimap_frequency.contains_value(value)) {
			const n_t next_idx = stats.bimap_frequency.size();
			stats.bimap_frequency.insert(next_idx, value);
		} else {
			const n_t existing_key = stats.bimap_frequency.get_key(value);
			stats.bimap_frequency.insert(existing_key, value);
		}
		stats.min = std::min(stats.min, value);
		stats.max = std::max(stats.max, value);
	}
	stats.n_nulls = 0;

	rowgroup_pt rowgroup;
	rowgroup.emplace_back(std::move(typed_col));

	vector<up<ColumnDescriptorT>> column_descriptors;
	auto                          column_descriptor = make_unique<ColumnDescriptorT>();
	column_descriptor->data_type                    = detail::section_data_type<PT>();
	column_descriptor->idx                          = 0;
	column_descriptor->max                          = make_unique<BinaryValueT>();
	column_descriptor->encoding_rpn                 = make_unique<RPNT>();
	column_descriptors.push_back(std::move(column_descriptor));

	RowgroupDescriptorT footer;
	footer.m_n_vec = n_vec;

	gather_statistics(rowgroup, column_descriptors);
	constant_visit(rowgroup[0], *column_descriptors[0]);

	if (!column_descriptors[0]->encoding_rpn->operator_tokens.empty()) {
		// The pre-pass already decided: the section is constant.
		return column_descriptors[0]->encoding_rpn->operator_tokens[0];
	}

	const Connection con; // default: no disabled encodings, sample_size 0 (measure every vector)

	for (const OperatorToken& token : detail::section_candidate_pool<PT>()) {
		const n_t size = TryExpr(rowgroup, *column_descriptors[0], token, footer, con);
		auto      res  = make_unique<ExpressionResultT>();
		res->operator_token = token;
		res->size            = size;
		column_descriptors[0]->expr_space.push_back(std::move(res));
	}

	return ChooseBestExpr(column_descriptors[0]->expr_space);
}

} // namespace subintsplit
} // namespace fastlanes

#endif // FLS_EXPRESSION_SUBINTSPLIT_SECTION_SELECTOR_HPP
