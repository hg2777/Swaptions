# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 2026

GIRR delta sensitivities for European swaptions, per PHYSICAL curve -- the
RiskWatch risk factor.

A swaption reads every curve through its underlying swap: the discount leg
reads params['discount_curve'] and the float forwards read
params['forecast_curve']. For each swaption and tenor, each of the deal's
DISTINCT physical curves is tent-shocked once, by name, and the option is
fully repriced:

    Delta = (V_shocked - V_base) / shock

The shocked curve moves in EVERY role that reads it, so a single-curve deal
gets one row per tenor carrying its whole sensitivity (discounting and
forwards together) and a dual-curve deal gets one row per curve.

WHY THE SHOCK IS PER CURVE AND NOT PER ROLE: shocking the discount role and
the forecast role separately and summing the two deltas agrees with the
single-curve shock only to FIRST order in the bump; the summation drops the
discount x forward cross-term. The wrapper that applies it,
swaptionScenario.CurveShockedSet, therefore matches on the curve NAME rather
than on the role.

The IMPLIED VOLATILITY IS HELD AT ITS BASE VALUE through the shock: the
surface is not re-interpolated at the shocked moneyness (swaptionScenario).

The tenor grid (days from the valuation date + display labels) is configured
in swaptionMain.py and passed in; nothing here hardcodes the vertices.

GIRR_Delta output is long-format:  ID | Tenor | Curve | Delta

@author: E42656
"""

from collections import OrderedDict

import pandas as pd

from sensitivity import ONE_BP, _tent_shock_fn, tenors_from_days
from swaptionScenario import (CurveShockedSet, FixedVolSwaption,
                              attach_riskwatch, build_swaption_specs,
                              fx_factor, physical_curves, safe_years,
                              swaption_rw_id)

pd.set_option('display.max_columns', None)

# A GIRR cell can be an ANALYTIC zero. Where a deal discounts and forecasts off
# the SAME curve its float leg telescopes to N * (DF_start - DF_maturity), so a
# vertex whose tent touches only INTERMEDIATE schedule dates cancels exactly:
# each such date is one period's payment date and the next period's accrual
# start, and the two occurrences cancel. P1273030's 0.5Y vertex is one -- its
# tent reaches only 2026-07-31.
#
# Neither engine can print a clean zero there. The delta is a one-sided bump,
# (V_shocked - V_base) / shock, so the subtraction cancels ~16 significant
# figures of a PV and divides the residue by the shock: at 1bp on a 616k float
# leg that leaves ~1e-5. The residue does not scale with the shock (it falls to
# ~4e-9 at 500bp, where a real sensitivity would hold), which is what marks it
# as arithmetic rather than risk.
#
# Dividing one engine's residue by the other's produces a meaningless
# percentage -- -1.9e-05 against RiskWatch's +2.6e-05 reads as -173%. A cell is
# therefore treated as an agreed zero when BOTH sides sit below this fraction
# of the deal's own largest delta. At 1e-8 the floor is eight orders below the
# deal's biggest exposure, so no cell carrying real risk can be masked by it.
NEGLIGIBLE_REL_TOL = 1e-8


def _agreed_zeros(out, rel_tol):
    '''Set the error to 0.0 on cells where OUR delta and RiskWatch's are both
    negligible against the deal's own largest delta -- an analytic zero that
    both engines report as cancellation residue. Returns the row mask.'''
    ours = pd.to_numeric(out['Delta-UAT'], errors='coerce').abs()
    rw = pd.to_numeric(out['Delta-RiskWatch'], errors='coerce').abs()
    scale = pd.concat([ours, rw], axis=1).max(axis=1).groupby(
        out['ID']).transform('max')
    floor = scale * float(rel_tol)
    zero = (ours <= floor) & (rw <= floor) & rw.notna()
    out.loc[zero, '(Delta-UAT/RW-1)%'] = 0.0
    return zero


def swaption_girr_delta_long(curves, surfaces, specs, tenor_table,
                             shock=ONE_BP, fx=None):
    '''
    Long-format GIRR delta per swaption, tenor and PHYSICAL curve:

        ID | Tenor | Curve | Delta          (Delta = (V_shocked - V_base)/shock)

    Each distinct curve the deal reads is tent-shocked once with a single
    one-sided bump and the option is fully repriced on its base volatility.
    '''
    tenor_days = tenor_table['days'].values.astype(float)
    tenor_labels = list(tenor_table['tenor'])

    rows = []
    for spec in specs:
        params = spec['params']
        vol = spec['base_vol']
        factor = fx_factor(fx, params)

        v_base = FixedVolSwaption(curves, surfaces, params, vol).npv()

        for t, label in enumerate(tenor_labels):
            fn = _tent_shock_fn(tenor_days, t, shock)
            for cname in physical_curves(params):
                shocked = CurveShockedSet(curves, fn, cname)
                v = FixedVolSwaption(shocked, surfaces, params, vol).npv()
                rows.append(OrderedDict([
                    ('ID', spec['id']),
                    ('Tenor', label),
                    ('Curve', cname),
                    ('Delta', (v - v_base) / shock * factor),
                ]))

    out = pd.DataFrame(rows, columns=['ID', 'Tenor', 'Curve', 'Delta'])
    if not out.empty:
        out = out.sort_values('ID', kind='mergesort').reset_index(drop=True)
    return out


def swaption_girr_for_portfolio(port, tenor_days, tenor_labels, shock=ONE_BP):
    '''GIRR delta table for an already-constructed SwaptionPortfolio, so the
    pricing pass and the sensitivity pass share one portfolio (curves,
    surfaces and workbook loaded once).'''
    specs = build_swaption_specs(port)
    tenor_table = tenors_from_days(tenor_days, tenor_labels,
                                   port.valuation_date)
    return swaption_girr_delta_long(port.curves, port.surfaces, specs,
                                    tenor_table, shock=shock,
                                    fx=getattr(port, 'fx', None))


def swaption_girr_with_riskwatch(girr_long, rw_delta=None, sens_round=6,
                                 pct_round=4, negligible_rel_tol=NEGLIGIBLE_REL_TOL,
                                 verbose=True):
    '''
    GIRR delta with the RiskWatch comparison attached, matched on
    DealNum + curve + tenor:

        ID | Tenor | Curve | Delta-UAT [| Delta-RiskWatch | (Delta-UAT/RW-1)%]

    rw_delta is {DealNum: {curve: {tenor_years: value}}}, read once for all
    three measures by swaptionSensitivity.load_rw_swaption_sensitivities.

    Both delta columns are reported as computed. A cell where both sides are
    negligible against the deal's own largest delta is an analytic zero that
    neither engine can print cleanly, so its error is set to 0.0 rather than
    left as one cancellation residue over another; the cells this touches are
    listed when `verbose` (see NEGLIGIBLE_REL_TOL).
    '''
    if girr_long is None or len(girr_long) == 0:
        return girr_long

    out = girr_long.rename(columns={'Delta': 'Delta-UAT'}).copy()
    out['Delta-UAT'] = out['Delta-UAT'].round(sens_round)

    if rw_delta is not None:
        def _rw(did, curve, tenor):
            yrs = safe_years(tenor)
            return rw_delta.get(swaption_rw_id(did), {}).get(
                curve, {}).get(round(yrs, 6))

        out = attach_riskwatch(out, 'Delta-UAT',
                               [_rw(d, c, t) for d, c, t
                                in zip(out['ID'], out['Curve'], out['Tenor'])],
                               sens_round, pct_round)

        zero = _agreed_zeros(out, negligible_rel_tol)
        if verbose and zero.any():
            for _, r in out[zero].iterrows():
                print("[swaptionGirr] {0} {1} {2}: analytic zero "
                      "(UAT {3:.3e} / RW {4:.3e}) -> reconciled".format(
                          r['ID'], r['Curve'], r['Tenor'],
                          r['Delta-UAT'], r['Delta-RiskWatch']))

    out['_yrs'] = out['Tenor'].apply(safe_years)
    return (out.sort_values(['ID', 'Curve', '_yrs'], kind='mergesort')
               .drop(columns='_yrs').reset_index(drop=True))
