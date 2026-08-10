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
                                 pct_round=4):
    '''
    GIRR delta with the RiskWatch comparison attached, matched on
    DealNum + curve + tenor:

        ID | Tenor | Curve | Delta-UAT [| Delta-RiskWatch | (Delta-UAT/RW-1)%]

    rw_delta is {DealNum: {curve: {tenor_years: value}}}, read once for all
    three measures by swaptionSensitivity.load_rw_swaption_sensitivities.
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

    out['_yrs'] = out['Tenor'].apply(safe_years)
    return (out.sort_values(['ID', 'Curve', '_yrs'], kind='mergesort')
               .drop(columns='_yrs').reset_index(drop=True))