# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 2026

Vega sensitivities for European swaptions, spread over the SA FRTB vega grid.

The vega P&L is the option repriced with its implied volatility scaled by
(1 + 0.001), differenced against the base and divided by the relative shock:

    VegaPL = ( V(sigma * 1.001) - V(sigma) ) / 0.001

Nothing else moves: the curves, and therefore the swap yield and the
moneyness, are the base ones, and the volatility is scaled directly rather
than re-read off the surface (swaptionScenario.FixedVolSwaption).

The P&L is then spread over the standard tenors (6M, 1Y, 3Y, 5Y, 10Y) on TWO
axes:

    OPTION TERM    -- the option's expiry measured from the valuation date
    SWAP DURATION  -- the underlying's own length, its maturity measured from
                      its real start date

Each axis is bracketed by the two adjacent grid tenors and split by the share
of the gap between them,

    w_lower = (upper - t) / (upper - lower)      w_upper = 1 - w_lower

giving up to four (option term, swap duration) cells, each carrying

    Vega(cell) = w_option * w_duration * VegaPL

A point sitting on a grid tenor, or outside the grid, collapses to a single
tenor on that axis and the zero-weight cells are dropped -- RiskWatch does not
report them either.

NOTE ON THE SWAP-DURATION AXIS.  It is measured from the underlying's start,
NOT from the valuation date. On the reference deal that is 1461 days
(2030-08-27 less 2026-08-27, bracketed 3Y/5Y at 0.5/0.5); from the valuation
date it would be 1700 days and the weights 0.83/0.17, and the cells would not
reconcile. The option term IS measured from the valuation date, and the two
are not the same measurement -- the expiry sits two business days before the
underlying's start.

RiskWatch reports these as 'Vega' rows keyed on the surface (Risk Factor ID)
with Vertex 1 = option term and Vertex 2 = swap duration, both in years.

Vega output is long-format:
    ID | Surface | Option Term | Swap Duration | Option Weight
       | Duration Weight | Vega PL | Vega

@author: E42656
"""

from collections import OrderedDict

import pandas as pd

from swaptionScenario import (FixedVolSwaption, attach_riskwatch,
                              build_swaption_specs, fx_factor, safe_years,
                              swaption_rw_id)

pd.set_option('display.max_columns', None)

# The RELATIVE volatility shock, sigma -> sigma * (1 + VEGA_REL_SHOCK).
VEGA_REL_SHOCK = 0.001

# SA FRTB vega grid, shared by the option-term and the swap-duration axis.
# Day counts follow the GIRR vertices, so 6M is 181 days and 3Y is 1096 rather
# than 182.5 / 1095.
VEGA_TENOR_LABELS = ('6M', '1Y', '3Y', '5Y', '10Y')
VEGA_TENOR_DAYS = (181, 365, 1096, 1826, 3652)


def vega_tenor_weights(t_days, grid_days=VEGA_TENOR_DAYS,
                       grid_labels=VEGA_TENOR_LABELS):
    '''
    [(tenor label, weight), ...] splitting a point across the two SA FRTB
    tenors that bracket it, by the share of the gap between them:

        w_lower = (upper - t) / (upper - lower)      w_upper = 1 - w_lower

    A point on a grid tenor, or outside the grid, collapses to that single
    tenor with weight 1; zero-weight tenors are dropped.
    '''
    x = [float(d) for d in grid_days]
    labels = list(grid_labels)
    t = float(t_days)

    if t <= x[0]:
        return [(labels[0], 1.0)]
    if t >= x[-1]:
        return [(labels[-1], 1.0)]

    for i in range(len(x) - 1):
        if x[i] <= t <= x[i + 1]:
            w_upper = (t - x[i]) / (x[i + 1] - x[i])
            out = []
            if w_upper < 1.0:
                out.append((labels[i], 1.0 - w_upper))
            if w_upper > 0.0:
                out.append((labels[i + 1], w_upper))
            return out

    raise ValueError('{0} days sits outside the vega grid {1}'.format(
        t, list(grid_days)))


def vega_axes(params):
    '''(option term days, swap duration days) for one deal.

    The option term runs from the valuation date to expiry; the swap duration
    runs from the underlying's own real start date to its maturity. The legs
    carry the start, and the earlier of the two is taken so a deal whose legs
    disagree still measures the underlying from when it actually begins (the
    portfolio flags that disagreement as a blotter issue).
    '''
    valuation = pd.Timestamp(params['valuation_date'])
    expiry = pd.Timestamp(params['expiry_date'])
    maturity = pd.Timestamp(params['maturity_date'])
    start = min(pd.Timestamp(params['fixed_real_start_date']),
                pd.Timestamp(params['float_real_start_date']))
    return (expiry - valuation).days, (maturity - start).days


def swaption_vega_long(curves, surfaces, specs, grid_days=VEGA_TENOR_DAYS,
                       grid_labels=VEGA_TENOR_LABELS,
                       rel_shock=VEGA_REL_SHOCK, fx=None):
    '''
    Long-format vega per swaption and (option term, swap duration) cell:

        ID | Surface | Option Term | Swap Duration | Option Weight
           | Duration Weight | Vega PL | Vega
    '''
    rows = []
    for spec in specs:
        params = spec['params']
        vol = spec['base_vol']
        factor = fx_factor(fx, params)

        option_days, duration_days = vega_axes(params)

        v_base = FixedVolSwaption(curves, surfaces, params, vol).npv()
        v_up = FixedVolSwaption(curves, surfaces, params,
                                vol * (1.0 + rel_shock)).npv()
        vega_pl = (v_up - v_base) / rel_shock

        for o_label, o_w in vega_tenor_weights(option_days, grid_days,
                                               grid_labels):
            for d_label, d_w in vega_tenor_weights(duration_days, grid_days,
                                                   grid_labels):
                rows.append(OrderedDict([
                    ('ID', spec['id']),
                    ('Surface', u'{0}'.format(
                        params['vol_surface_name']).strip()),
                    ('Option Term', o_label),
                    ('Swap Duration', d_label),
                    ('Option Weight', o_w),
                    ('Duration Weight', d_w),
                    ('Vega PL', vega_pl * factor),
                    ('Vega', o_w * d_w * vega_pl * factor),
                ]))

    out = pd.DataFrame(rows, columns=[
        'ID', 'Surface', 'Option Term', 'Swap Duration', 'Option Weight',
        'Duration Weight', 'Vega PL', 'Vega'])
    if not out.empty:
        out = out.sort_values('ID', kind='mergesort').reset_index(drop=True)
    return out


def swaption_vega_for_portfolio(port, grid_days=VEGA_TENOR_DAYS,
                                grid_labels=VEGA_TENOR_LABELS,
                                rel_shock=VEGA_REL_SHOCK):
    '''Vega table for an already-constructed SwaptionPortfolio.'''
    specs = build_swaption_specs(port)
    return swaption_vega_long(port.curves, port.surfaces, specs,
                              grid_days=grid_days, grid_labels=grid_labels,
                              rel_shock=rel_shock,
                              fx=getattr(port, 'fx', None))


def swaption_vega_with_riskwatch(vega_long, rw_vega=None, sens_round=6,
                                 pct_round=4):
    '''
    Vega with the RiskWatch comparison attached, matched on
    DealNum + surface + (option term, swap duration):

        ID | Surface | Option Term | Swap Duration | Option Weight
           | Duration Weight | Vega PL | Vega-UAT
           [| Vega-RiskWatch | (Vega-UAT/RW-1)%]

    rw_vega is {DealNum: {surface: {(option_years, duration_years): value}}},
    read once for all three measures by
    swaptionSensitivity.load_rw_swaption_sensitivities.
    '''
    if vega_long is None or len(vega_long) == 0:
        return vega_long

    out = vega_long.rename(columns={'Vega': 'Vega-UAT'}).copy()
    out['Vega-UAT'] = out['Vega-UAT'].round(sens_round)

    if rw_vega is not None:
        def _rw(did, surface, option, duration):
            key = (round(safe_years(option), 6), round(safe_years(duration), 6))
            return rw_vega.get(swaption_rw_id(did), {}).get(
                surface, {}).get(key)

        out = attach_riskwatch(out, 'Vega-UAT',
                               [_rw(d, s, o, u) for d, s, o, u
                                in zip(out['ID'], out['Surface'],
                                       out['Option Term'],
                                       out['Swap Duration'])],
                               sens_round, pct_round)

    out['_o'] = out['Option Term'].apply(safe_years)
    out['_d'] = out['Swap Duration'].apply(safe_years)
    return (out.sort_values(['ID', 'Surface', '_o', '_d'], kind='mergesort')
               .drop(columns=['_o', '_d']).reset_index(drop=True))