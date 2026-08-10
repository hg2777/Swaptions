# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 2026

The three swaption FRTB SA sensitivity passes, brought together behind one
call for swaptionMain.py.

Each measure lives in its own module -- swaptionGirr, swaptionCurvature,
swaptionVega -- over the scenario machinery they share (swaptionScenario).
This module owns only what is common to running all three together:

    * ONE pass over the FRTB SA report, producing the three RiskWatch lookups
      (the report runs to tens of thousands of rows, so it is not read three
      times)
    * the combined run: run the three measures over one priced portfolio,
      attach the comparison and print the summary

Targets Python 2.7 (no f-strings, explicit float division, object base).

@author: E42656
"""

from collections import OrderedDict

import pandas as pd

from sensitivity import ONE_BP
from swaptionScenario import RW_SWAPTION_PREFIX, swaption_rw_id
from swaptionGirr import (swaption_girr_for_portfolio,
                          swaption_girr_with_riskwatch)
from swaptionCurvature import (CURVATURE_SHOCK,
                               swaption_curvature_for_portfolio,
                               swaption_curvature_with_riskwatch)
from swaptionVega import (VEGA_REL_SHOCK, VEGA_TENOR_DAYS, VEGA_TENOR_LABELS,
                          swaption_vega_for_portfolio,
                          swaption_vega_with_riskwatch)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


# ---------------------------------------------------------------------------
# RiskWatch report -- one pass, three lookups
# ---------------------------------------------------------------------------
def load_rw_swaption_sensitivities(path, id_col='Instrument ID',
                                   class_col='Risk Factor Class',
                                   factor_col='Risk Factor ID',
                                   vertex1_col='Risk Factor Vertex 1',
                                   vertex2_col='Risk Factor Vertex 2',
                                   value_col=(
                                       'Sensitivity Value '
                                       '(Reporting Currency)'),
                                   type_col='Sensitivity Type', verbose=True):
    '''
    RiskWatch swaption GIRR sensitivities, read in ONE pass over the FRTB SA
    report:

        delta     : {DealNum: {curve:   {tenor_years: value}}}
        curvature : {DealNum: {scenario: value}}
        vega      : {DealNum: {surface: {(option_years, duration_years): v}}}

    Only GIRR rows tagged "SWO <DealNum>" are read. Delta rows are keyed on
    the curve (Risk Factor ID) and Vertex 1; curvature rows carry the currency
    as their Risk Factor ID and no vertex; vega rows carry the surface with
    Vertex 1 = option term and Vertex 2 = swap duration.
    '''
    raw = pd.read_csv(path, dtype=str)
    raw.columns = [u'{0}'.format(c).strip() for c in raw.columns]

    delta = OrderedDict()
    curvature = OrderedDict()
    vega = OrderedDict()
    n_delta = n_curv = n_vega = 0

    for _, r in raw.iterrows():
        if u'GIRR' not in u'{0}'.format(r.get(class_col)).upper():
            continue
        inst = u'{0}'.format(r.get(id_col)).strip()
        if not inst.upper().startswith(RW_SWAPTION_PREFIX):
            continue

        deal = swaption_rw_id(inst)
        factor = u'{0}'.format(r.get(factor_col)).strip()
        stype = u'{0}'.format(r.get(type_col)).strip()
        v = pd.to_numeric(r.get(value_col), errors='coerce')
        if not deal or factor == '' or pd.isna(v):
            continue

        low = stype.lower()
        if low == 'delta':
            y = pd.to_numeric(r.get(vertex1_col), errors='coerce')
            if pd.isna(y):
                continue
            delta.setdefault(deal, OrderedDict()).setdefault(
                factor, {})[round(float(y), 6)] = float(v)
            n_delta += 1
        elif low.startswith('curvature'):
            curvature.setdefault(deal, OrderedDict())[stype] = float(v)
            n_curv += 1
        elif low == 'vega':
            y1 = pd.to_numeric(r.get(vertex1_col), errors='coerce')
            y2 = pd.to_numeric(r.get(vertex2_col), errors='coerce')
            if pd.isna(y1) or pd.isna(y2):
                continue
            key = (round(float(y1), 6), round(float(y2), 6))
            vega.setdefault(deal, OrderedDict()).setdefault(
                factor, {})[key] = float(v)
            n_vega += 1

    if verbose:
        print("[swaptionSensitivity] RW {0!r}: {1} delta / {2} curvature / "
              "{3} vega cells".format(path, n_delta, n_curv, n_vega))
    return delta, curvature, vega


# ---------------------------------------------------------------------------
# Reconciliation summary
# ---------------------------------------------------------------------------
def print_sensitivity_reconciliation(table, name, our_col):
    '''
    Short RiskWatch reconciliation summary for one sensitivity table.

    Every computed cell is in the table whether or not it matched a RiskWatch
    row, so the computed count and the matched count are reported separately;
    the error statistics run over the matched cells alone. A measure with no
    match at all still reports how many cells reached the workbook.
    '''
    rw_col = our_col.replace('-UAT', '-RiskWatch')
    err_col = '({0}/RW-1)%'.format(our_col)
    print("-" * 95)
    if table is None or len(table) == 0:
        print("  {0:<10}: no cells computed".format(name))
        return
    if err_col not in table.columns:
        print("  {0:<10}: {1} cells computed   (no RiskWatch report "
              "supplied)".format(name, len(table)))
        return

    matched = int(pd.to_numeric(table[rw_col], errors='coerce').notna().sum())
    print("  {0:<10}: {1} cells computed, {2} matched a RiskWatch row".format(
        name, len(table), matched))

    # Unmatched cells carry no error, so they are dropped from the statistics
    # rather than counted as misses -- the shares below are of the MATCHED
    # cells, not of everything computed.
    err = pd.to_numeric(table[err_col], errors='coerce').abs().dropna()
    if err.empty:
        return
    print("  {0:<10}  median |UAT/RW-1| : {1:.6f}%".format('', err.median()))
    print("  {0:<10}  within 0.01% / 1% : {1:.1f}% / {2:.1f}%".format(
        '', (err <= 0.01).mean() * 100.0, (err <= 1).mean() * 100.0))
    print("  {0:<10}  worst cell        : {1} {2:.6f}%".format(
        '', table.loc[err.idxmax(), 'ID'], err.max()))


# ---------------------------------------------------------------------------
# Combined run
# ---------------------------------------------------------------------------
def swaption_sensitivities(port, girr_tenor_days, girr_tenor_labels,
                           girr_shock=ONE_BP,
                           curvature_shock=CURVATURE_SHOCK,
                           vega_tenor_days=VEGA_TENOR_DAYS,
                           vega_tenor_labels=VEGA_TENOR_LABELS,
                           vega_rel_shock=VEGA_REL_SHOCK,
                           rw_report_csv=None, verbose=True):
    '''
    Run the three measures over an already-priced SwaptionPortfolio, attach
    the RiskWatch comparison to each and print the summary.

    Returns (girr, curvature, vega) as long-format DataFrames, each carrying
    its RiskWatch and error columns when a report was supplied.
    '''
    if verbose:
        print("=" * 95)
        print("SWAPTION : FRTB SA sensitivities + RiskWatch comparison")
        print("  GIRR      : {0}bp tent shock per physical curve on "
              "{1}".format(girr_shock * 10000.0, list(girr_tenor_labels)))
        print("  curvature : {0:.10f} parallel shift on every curve of the "
              "currency".format(curvature_shock))
        print("  vega      : sigma -> sigma * {0} spread over {1} x "
              "{1}".format(1.0 + vega_rel_shock, list(vega_tenor_labels)))

    girr = swaption_girr_for_portfolio(port, girr_tenor_days,
                                       girr_tenor_labels, shock=girr_shock)
    curvature = swaption_curvature_for_portfolio(port, shock=curvature_shock)
    vega = swaption_vega_for_portfolio(port, grid_days=vega_tenor_days,
                                       grid_labels=vega_tenor_labels,
                                       rel_shock=vega_rel_shock)

    rw_delta = rw_curvature = rw_vega = None
    if rw_report_csv:
        rw_delta, rw_curvature, rw_vega = load_rw_swaption_sensitivities(
            rw_report_csv, verbose=verbose)

    girr = swaption_girr_with_riskwatch(girr, rw_delta)
    curvature = swaption_curvature_with_riskwatch(curvature, rw_curvature)
    vega = swaption_vega_with_riskwatch(vega, rw_vega)

    if verbose:
        print_sensitivity_reconciliation(girr, 'GIRR', 'Delta-UAT')
        print_sensitivity_reconciliation(curvature, 'Curvature', 'CVR-UAT')
        print_sensitivity_reconciliation(vega, 'Vega', 'Vega-UAT')

    return girr, curvature, vega