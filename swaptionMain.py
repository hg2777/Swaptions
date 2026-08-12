# -*- coding: utf-8 -*-
"""
European swaption pricing in the Bachelier (normal) framework.

Prices the swaption book, produces its FRTB SA sensitivities (GIRR delta,
curvature and vega) and reconciles all four outputs to RiskWatch. Kept as a
separate runner from main.py (the vanilla swap book) so the two orchestrations
stay independent; the curve loader, the FX converter and the underlying-swap
leg engine are shared.

The three sensitivity measures live one per module -- swaptionGirr,
swaptionCurvature, swaptionVega, over the shared scenario machinery in
swaptionScenario -- and are run together by swaptionSensitivity.

@author: E42656
"""

import pandas as pd

from linearInterpolation import load_curve_set, load_fx_rates, FxConverter
from volatilitySurface import SurfaceSet
from swaptionPortfolio import SwaptionPortfolio, read_surface_names
from swaptionDiagnostics import write_swaption_diagnostics
from swaptionSensitivity import swaption_sensitivities
from swaptionSensitivityDiagnostics import (
    write_swaption_sensitivity_diagnostics)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
XLSX = 'C:\\Users\\hgavr\\Documents\\Ethniki\\Work\\Pricing and Sensitivities\\Swaptions\\curves.xlsx'                 # workbook holding the curve tabs

INTERPOLATION_METHOD = 'linear'

# FX rates tab in the curves workbook, used to restate MtM into the EUR
# reporting currency (RiskWatch reports in reporting currency).
FX_SHEET           = 'FX rates'
REPORTING_CURRENCY = 'EUR'

# --- swaption book inputs ----------------------------------------------------
SWAPTION_INPUT_XLSX    = "C:\\Users\\hgavr\\Documents\\Ethniki\\Work\\Pricing and Sensitivities\\Swaptions\\Swaption_Input.xlsx"  # three-tab population workbook
SWAPTION_SHEET         = 'Swaption'  # option tab
SWAPTION_FIXED_SHEET   = 'Fixed'     # underlying fixed-leg tab
SWAPTION_FLOAT_SHEET   = 'Float'     # underlying float-leg tab
# Curve-name resolution. The 6-month forecast index name 'EUR-SWP-6M' is not
# bootstrapped separately; SwaptionPortfolio strips the '-6M' tenor tag to the
# simple curve ('EUR-SWP'), as the vanilla swap book does. Discount curves
# resolve as-is. This dict is only for additional explicit overrides.
SWAPTION_CURVE_ALIASES = {}
SWAPTION_VALUATION_DATE = '2025-12-31'
SWAPTION_RW_MTM_CSV     = 'C:\\Users\\hgavr\\Documents\\Ethniki\\Work\\Pricing and Sensitivities\\Swaptions\\frtb_sa_report.csv'  # None to skip the comparison
SWAPTION_RW_INSTRUMENT_COL = 'Instrument ID'
SWAPTION_RW_MTM_COL     = 'Mark To Market'
SWAPTION_OUT            = 'C:\\Users\\hgavr\\Documents\\Ethniki\\Work\\Pricing and Sensitivities\\Swaptions\\swaption_results.xlsx'
SWAPTION_DIAGNOSTICS_OUT = 'C:\\Users\\hgavr\\Documents\\Ethniki\\Work\\Pricing and Sensitivities\\Swaptions\\swaption_diagnostics.xlsx'
SWAPTION_SENSITIVITY_DIAGNOSTICS_OUT = 'C:\\Users\\hgavr\\Documents\\Ethniki\\Work\\Pricing and Sensitivities\\Swaptions\\swaption_sensitivity_diagnostics.xlsx'

# Troubleshooting: restrict the run to these DealNums. None -> whole book.
SWAPTION_ONLY_IDS = [

]

# --- FRTB SA sensitivities ---------------------------------------------------
# False to price only (skips the three sensitivity passes, their tabs and the
# sensitivity diagnostics workbook).
RUN_SENSITIVITIES = True

# GIRR delta vertices: labels and their day counts from the valuation date.
GIRR_TENOR_LABELS = ('0.25Y', '0.5Y', '1Y', '2Y', '3Y', '5Y',
                     '10Y', '15Y', '20Y', '30Y')
GIRR_TENOR_DAYS   = (90, 181, 365, 730, 1096, 1826, 3652, 5479, 7305, 10957)
GIRR_SHOCK        = 0.0001   # 1bp bump-and-reprice shock

# Curvature: parallel shift applied to the deal's risk-factor curves.
CURVATURE_SHOCK   = 0.017 / (2.0 ** 0.5)

# Curves a deal may READ without them being GIRR risk factors, so the
# curvature shift leaves them still. 'EUR-SWP-1M' is one: it appears nowhere
# among the 42 GIRR Risk Factor IDs of the FRTB SA report, and for the deal
# that forecasts off it RiskWatch publishes EUR-SWP delta rows only. Shifting
# the discount curve alone reproduces its curvature exactly; shifting the
# forecast curve too misses by 2300%.
# The GIRR delta pass is NOT filtered by this list -- it still shocks every
# curve the deal reads and reports a row per curve, so a computed sensitivity
# is never dropped; the rows for these curves reconcile to N/A.
NON_RISK_FACTOR_CURVES = ('EUR-SWP-1M',)

# Vega: the SA FRTB standard tenors, shared by the option-term and the
# swap-duration axis, and the relative volatility shock.
VEGA_TENOR_LABELS = ('6M', '1Y', '3Y', '5Y', '10Y')
VEGA_TENOR_DAYS   = (181, 365, 1096, 1826, 3652)
VEGA_REL_SHOCK    = 0.001


# ----------------------------------------------------------------------------
# Reconciliation summary
# ----------------------------------------------------------------------------

# How many deal to list individually before truncating them
MTM_LIST_LIMIT = 25

def _print_mtm_list(rows, mtm_col, heading):
    ''' Per deal MtM listing'''
    if not len(rows):
        return
    print(" {0}".format(heading))
    for i, (_, r) in enumerate(rows.iterrows()):
        if i >= MTM_LIST_LIMIT:
            print(" ,,, and {0} more (see the results workbook)".format(
                    len(rows) - MTM_LIST_LIMIT))
            break
        print(" {0:<14} {1:>18,.2f} {2}".format(
                r.get('DealNum', ''), r[mtm_col], r.get('Currency', '')))


def _print_rw_reconciliation(mtm):
    """Short RiskWatch reconciliation summary for the priced swaption book."""
    if 'MtM-RiskWatch' in mtm.columns:
        m = mtm[mtm['MtM-UAT'].notna() & mtm['MtM-RiskWatch'].notna()
                & (mtm['MtM-RiskWatch'] != 0)]
        rel = (m['MtM-UAT'] / m['MtM-RiskWatch'] - 1.0).abs() * 100.0
        print("  priced            : {0}".format(
            int(mtm['MtM-UAT'].notna().sum())))
        print("  compared to RW    : {0}".format(len(m)))
        if len(m):
            print("  median |UAT/RW-1| : {0:.6f}%".format(rel.median()))
            print("  within 0.01% / 1% : {0:.1f}% / {1:.1f}%".format(
                (rel <= 0.01).mean() * 100.0, (rel <= 1).mean() * 100.0))
        unmatched = mtm[mtm['MtM-UAT'].notna() & mtm['MtM-RiskWatch'].isna()]
        _print_mtm_list(unmatched, 'MtM-UAT', 
                            'no RiskWatch MtM for {0} deals'.format(len(unmatched)))
    else:
        priced = mtm[mtm['MtM'].notna()] if 'MtM' in mtm.columns else mtm
        print("  priced : {0}   (no RiskWatch report supplied -> MtM "
              "only)".format(len(priced)))
        _print_mtm_list(priced, 'MtM', 'MtM:')


def build_portfolio(curves, surfaces):
    """The swaption book, loaded once and shared by pricing and sensitivity."""
    fx = FxConverter(load_fx_rates(XLSX, FX_SHEET),
                     reporting=REPORTING_CURRENCY)

    return SwaptionPortfolio(curves, surfaces, SWAPTION_INPUT_XLSX,
                             SWAPTION_SHEET, SWAPTION_FIXED_SHEET,
                             SWAPTION_FLOAT_SHEET, SWAPTION_VALUATION_DATE,
                             rw_mtm_csv=SWAPTION_RW_MTM_CSV,
                             rw_instrument_col=SWAPTION_RW_INSTRUMENT_COL,
                             rw_mtm_col=SWAPTION_RW_MTM_COL,
                             only_ids=SWAPTION_ONLY_IDS,
                             curve_aliases=SWAPTION_CURVE_ALIASES,
                             fx=fx)


def price_swaptions(port, surfaces):
    """Swaption book run: price the book and compare MtM to RiskWatch."""
    print("=" * 95)
    print("SWAPTION : Bachelier pricing + RiskWatch MtM comparison")
    print("  workbook  : {0}   tabs: {1} / {2} / {3}".format(
        SWAPTION_INPUT_XLSX, SWAPTION_SHEET, SWAPTION_FIXED_SHEET,
        SWAPTION_FLOAT_SHEET))
    print("  valuation : {0}   RiskWatch report: {1}".format(
        SWAPTION_VALUATION_DATE, SWAPTION_RW_MTM_CSV))
    print("  surfaces  : {0}".format(sorted(surfaces.surfaces)))
    print("-" * 95)
    mtm = port.summary()
    _print_rw_reconciliation(mtm)
    return mtm


def write_results(mtm, girr, curvature, vega):
    """Every output tab of the run, written in one pass."""
    tabs = [('Swaption_MtM', mtm), ('GIRR_Delta', girr),
            ('Curvature', curvature), ('Vega', vega)]
    with pd.ExcelWriter(SWAPTION_OUT) as xl:
        for name, frame in tabs:
            if frame is None:
                continue
            frame.to_excel(xl, sheet_name=name, index=False, na_rep='N/A')

    print("=" * 95)
    print("written to {0}  (tabs: {1})".format(
        SWAPTION_OUT, ', '.join(n for n, f in tabs if f is not None)))
    print("=" * 95)


def main():
    surface_tabs = read_surface_names(SWAPTION_INPUT_XLSX, SWAPTION_SHEET)
    not_curves = [FX_SHEET] + surface_tabs
    
    curves = load_curve_set(XLSX, exclude=not_curves,
                            method=INTERPOLATION_METHOD)
    surfaces = SurfaceSet(path=XLSX)
    print("Curves loaded from: {0}".format(XLSX))
    print("Interpolation method: {0}".format(INTERPOLATION_METHOD))
    print("Volatility surfaces: {0}\n".format(sorted(surfaces.surfaces)))

    port = build_portfolio(curves, surfaces)
    mtm = price_swaptions(port, surfaces)

    girr = curvature = vega = None
    if RUN_SENSITIVITIES:
        girr, curvature, vega = swaption_sensitivities(
            port, GIRR_TENOR_DAYS, GIRR_TENOR_LABELS,
            girr_shock=GIRR_SHOCK,
            curvature_shock=CURVATURE_SHOCK,
            vega_tenor_days=VEGA_TENOR_DAYS,
            vega_tenor_labels=VEGA_TENOR_LABELS,
            vega_rel_shock=VEGA_REL_SHOCK,
            non_risk_curves=NON_RISK_FACTOR_CURVES,
            rw_report_csv=SWAPTION_RW_MTM_CSV)

    write_results(mtm, girr, curvature, vega)

    # Per-swaption diagnostics workbook (four-step trace), to pin any MtM
    # difference against RiskWatch to a step.
    write_swaption_diagnostics(SWAPTION_DIAGNOSTICS_OUT, port)

    # Per-swaption sensitivity diagnostics workbook: the legs and the four-step
    # trace, then the scenario behind all three measures.
    if RUN_SENSITIVITIES:
        write_swaption_sensitivity_diagnostics(
            SWAPTION_SENSITIVITY_DIAGNOSTICS_OUT, port,
            GIRR_TENOR_DAYS, GIRR_TENOR_LABELS,
            method=INTERPOLATION_METHOD,
            girr_shock=GIRR_SHOCK,
            curvature_shock=CURVATURE_SHOCK,
            vega_tenor_days=VEGA_TENOR_DAYS,
            vega_tenor_labels=VEGA_TENOR_LABELS,
            vega_rel_shock=VEGA_REL_SHOCK,
            non_risk_curves=NON_RISK_FACTOR_CURVES)

    return mtm, girr, curvature, vega


if __name__ == "__main__":
    main()
