# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 2026

GIRR curvature sensitivities for European swaptions.

A parallel shift of 0.017 / sqrt(2) is applied to the WHOLE curve set -- every
curve the deal reads moves together -- and the option is repriced in each
direction:

    CVR_up   = V(+shift) - V_base
    CVR_down = V(-shift) - V_base

The figure reported is the RAW MtM change: no division by the shift and no
delta term subtracted, matching the 'Curvature Up' / 'Curvature Down' rows of
the RiskWatch FRTB SA report and the reference workbook's Curvature tab.

WHY THE WHOLE SET AND NOT ONE CURVE.  RiskWatch's curvature risk factor sits
at the currency bucket rather than at a single curve: its Risk Factor ID is
the currency ('EUR', 'USD') and it carries no vertex. Shocking the discount
curve and the forecast curve separately would also drop the cross-term of a
shift this large -- 1.2 percentage points, two orders of magnitude beyond the
GIRR bump, where the option's convexity is the point of the measure rather
than a rounding error.

The IMPLIED VOLATILITY IS HELD AT ITS BASE VALUE through the shift: the
surface is not re-interpolated at the shifted moneyness (swaptionScenario).

Curvature output is long-format:  ID | Currency | Scenario | CVR

@author: E42656
"""

import math
from collections import OrderedDict

import pandas as pd

from swaptionScenario import (NEGLIGIBLE_ABS_TOL, FixedVolSwaption,
                              ParallelShockedSet, attach_riskwatch,
                              build_swaption_specs, negligible_agreed,
                              physical_curves, reporting_factor,
                              swaption_rw_id)

pd.set_option('display.max_columns', None)

# The parallel shift applied to every curve of the deal's currency. The FRTB
# GIRR curvature risk weight is 0.017 and the report quotes it as the Shock;
# the shift actually applied is that weight over sqrt(2), which is what
# reconciles to RiskWatch.
CURVATURE_SHOCK = 0.017 / math.sqrt(2.0)

# Scenario labels, matching the report's 'Sensitivity Type' values.
CURVATURE_UP = 'Curvature Up'
CURVATURE_DOWN = 'Curvature Down'


def swaption_curvature_long(curves, surfaces, specs, shock=CURVATURE_SHOCK,
                            fx=None, non_risk_curves=()):
    '''
    Long-format GIRR curvature per swaption and direction:

        ID | Currency | Scenario | CVR

    Every RISK-FACTOR curve the deal reads is shifted in parallel by +/- shock
    and the option is repriced on its base volatility. A curve the deal reads
    that is not a GIRR risk factor (non_risk_curves, configured in
    swaptionMain.py) stays still: RiskWatch reports such a deal's curvature as
    if only its remaining curves moved.
    '''
    rows = []
    for spec in specs:
        params = spec['params']
        vol = spec['base_vol']
        factor = reporting_factor(fx, params)
        currency = u'{0}'.format(params['currency']).strip()
        shifted_curves = physical_curves(params, non_risk_curves)

        v_base = FixedVolSwaption(curves, surfaces, params, vol).npv()

        for scenario, sign in ((CURVATURE_UP, 1.0), (CURVATURE_DOWN, -1.0)):
            shifted = ParallelShockedSet(curves, sign * shock, shifted_curves)
            v = FixedVolSwaption(shifted, surfaces, params, vol).npv()
            rows.append(OrderedDict([
                ('ID', spec['id']),
                ('Currency', currency),
                ('Scenario', scenario),
                ('CVR', (v - v_base) * factor),
            ]))

    out = pd.DataFrame(rows, columns=['ID', 'Currency', 'Scenario', 'CVR'])
    if not out.empty:
        out = out.sort_values('ID', kind='mergesort').reset_index(drop=True)
    return out


def swaption_curvature_for_portfolio(port, shock=CURVATURE_SHOCK,
                                     non_risk_curves=()):
    '''Curvature table for an already-constructed SwaptionPortfolio.'''
    specs = build_swaption_specs(port)
    return swaption_curvature_long(port.curves, port.surfaces, specs,
                                   shock=shock, fx=getattr(port, 'fx', None),
                                   non_risk_curves=non_risk_curves)


def swaption_curvature_with_riskwatch(curvature_long, rw_curvature=None,
                                      sens_round=6, pct_round=4):
    '''
    Curvature with the RiskWatch comparison attached, matched on
    DealNum + scenario:

        ID | Currency | Scenario | CVR-UAT [| CVR-RiskWatch | (CVR-UAT/RW-1)%]

    rw_curvature is {DealNum: {scenario: value}}, read once for all three
    measures by swaptionSensitivity.load_rw_swaption_sensitivities.
    '''
    if curvature_long is None or len(curvature_long) == 0:
        return curvature_long

    out = curvature_long.rename(columns={'CVR': 'CVR-UAT'}).copy()
    out['CVR-UAT'] = out['CVR-UAT'].round(sens_round)

    if rw_curvature is not None:
        rw_vals = [rw_curvature.get(swaption_rw_id(d), {}).get(s)
                   for d, s in zip(out['ID'], out['Scenario'])]
        out = attach_riskwatch(out, 'CVR-UAT', rw_vals, sens_round, pct_round)
        # both sides negligible -> an agreed zero, not a percentage of noise
        out.loc[negligible_agreed(out['CVR-UAT'], out['CVR-RiskWatch'],
                                  NEGLIGIBLE_ABS_TOL),
                '(CVR-UAT/RW-1)%'] = 0.0

    return (out.sort_values(['ID', 'Scenario'], kind='mergesort')
               .reset_index(drop=True))
