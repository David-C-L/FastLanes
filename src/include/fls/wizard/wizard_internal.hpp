// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/include/fls/wizard/wizard_internal.hpp
// ────────────────────────────────────────────────────────
//
// Declarations for a handful of wizard.cpp functions that already have external linkage (none are
// `static`) but were not previously declared in any header, so nothing outside wizard.cpp's own
// translation unit could call them. This header adds no new logic and changes no behavior -- it
// only exposes the existing, unmodified definitions in src/wizard/wizard.cpp for reuse.
//
// Motivating reuse: SubIntSplit's per-section codec selector (subintsplit_section_selector.hpp)
// needs the wizard's real, measured-cost pool search (TryExpr/ChooseBestExpr) and its
// constant-detection pre-pass (gather_statistics/constant_visit) to run on a section's extracted
// values, the same way the top-level wizard already runs them on a whole column.
#ifndef FLS_WIZARD_WIZARD_INTERNAL_HPP
#define FLS_WIZARD_WIZARD_INTERNAL_HPP

#include "fls/common/alias.hpp"
#include "fls/footer/column_descriptor_generated.h"
#include "fls/footer/operator_token_generated.h"
#include "fls/footer/rowgroup_descriptor_generated.h"
#include "fls/std/vector.hpp"
#include "fls/table/rowgroup.hpp"

namespace fastlanes {
/*--------------------------------------------------------------------------------------------------------------------*/
class Connection;
/*--------------------------------------------------------------------------------------------------------------------*/

// Runs `token` on `col[column_descriptor.idx]`, sampling `con.get_sample_size()` vectors (or all
// of `footer.m_n_vec` if that's 0 or larger than the column), and returns the real encoded size
// (extrapolated to the full column when sampled). This is the wizard's measured-not-modeled cost
// oracle -- it actually builds and runs the candidate PhysicalExpr via
// Interpreter::Encoding::Interpret, it does not estimate.
n_t TryExpr(const rowgroup_pt&       col,
            const ColumnDescriptorT& column_descriptor,
            const OperatorToken&     token,
            RowgroupDescriptorT&     footer,
            const Connection&        con);

// Picks the cheapest candidate (by TryExpr-measured size) from a column's populated expr_space.
// Asserts (FLS_ASSERT_FALSE, compiled out in Release) on an empty options list -- callers must
// guard against an empty pool themselves in Release builds.
OperatorToken ChooseBestExpr(const vector<up<ExpressionResultT>>& options);

// Copies each column's already-computed TypedStats (min/max/n_nulls) into its ColumnDescriptorT,
// recursing into STRUCT children. Must run before constant_visit/TryExpr, which read those
// copied-in stats and the column's own TypedStats directly.
void gather_statistics(const rowgroup_pt& rowgroup, vector<up<ColumnDescriptorT>>& column_descriptors);

// Structural pre-pass: if a column's stats say it is constant, appends the matching
// EXP_CONSTANT_* token directly to column_descriptor.encoding_rpn and returns without touching
// expr_space. This is the check force_schema_pool cannot reach (it's skipped entirely once a
// pool is forced), and the reason a Constant candidate needs its own explicit pre-pass rather
// than being just another pool entry -- see docs/subintsplit.md's force_schema_pool caveat.
void constant_visit(const col_pt& col, ColumnDescriptorT& column_descriptor);

} // namespace fastlanes

#endif // FLS_WIZARD_WIZARD_INTERNAL_HPP
