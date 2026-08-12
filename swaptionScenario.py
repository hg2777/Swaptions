# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 2026

Scenario machinery shared by the three swaption FRTB SA sensitivity measures
(swaptionGirr, swaptionCurvature, swaptionVega).

Everything the three measures have in common sits here rather than in any one
of them, so none has to import another and the combined runner
(swaptionSensitivity) can import all three without a cycle:

    * FixedVolSwaption   -- the reprice used by every scenario, with the
                            implied volatility pinned rather than re-read off
                            the surface
    * ParallelShockedSet -- a whole-curve-set parallel shift (curvature)
    * build_swaption_specs / fx_factor / physical_curves -- reading a priced
                            SwaptionPortfolio into scenario inputs
    * swaption_rw_id / attach_riskwatch / safe_years -- the RiskWatch
                            reconciliation shape the three measures share

WHY THE VOLATILITY IS PINNED.  Shocking the curves moves the swap yield and
therefore the moneyness, but the surface is NOT re-interpolated at that
shocked moneyness. The reference workbook is explicit about this: on its GIRR
and Curvature tabs the 'Volatility Implied' cell carries the base MtM tab's
interpolated number as a constant, not the '=K46' surface formula the MtM tabs
carry. Re-interpolating instead moves the 3Y GIRR cell of the reference deal
by 44%, so the choice is load-bearing rather than cosmetic. The volatility is
moved only by the vega scenario, and there it is moved directly.

Targets Python 2.7 (no f-strings, explicit float division, object base).

@author: E42656
"""

import numpy as np
import pandas as pd

from swaptionPricing import Swaption, position_sign

pd.set_option('display.max_columns', None)

# RiskWatch tags swaption rows of the FRTB SA report as "SWO <DealNum>".
RW_SWAPTION_PREFIX = 'SWO'

# Below this ABSOLUTE size a sensitivity is not a number either engine can
# print meaningfully: it is the residue of cancelling ~16 significant figures
# of a PV, not risk. Where OUR figure and RiskWatch's are both this small the
# cell is an agreed zero and its error is reported as 0.0, rather than as one
# engine's residue divided by the other's (which reads as -173% on a pair like
# -1.9e-05 against +2.6e-05).
NEGLIGIBLE_ABS_TOL = 0.0001


def reporting_factor(fx, params):
    '''The single multiplier taking a computed figure to a reported one: the
    FX factor into the reporting currency, times the position sign.'''
    return fx_factor(fx, params) * position_sign(params)


def negligible_agreed(ours, rw, abs_tol=NEGLIGIBLE_ABS_TOL):
    '''True where both sides are present and both are smaller than abs_tol --
    an analytic zero both engines report as cancellation residue.'''
    o = pd.to_numeric(ours, errors='coerce').abs()
    r = pd.to_numeric(rw, errors='coerce').abs()
    return (o < abs_tol) & (r < abs_tol) & o.notna() & r.notna()


# ---------------------------------------------------------------------------
# Repricing
# ---------------------------------------------------------------------------
class FixedVolSwaption(Swaption):
    '''
    Swaption repriced with the implied volatility PINNED to a supplied value.

    The GIRR and curvature scenarios pin it to the base volatility, so the
    shock reaches the price through the curves alone; the vega scenario pins
    it to the base volatility scaled by the relative shock, so the shock
    reaches the price through the volatility alone. Both read the underlying
    swap's legs through the unchanged pricing engine.
    '''

    def __init__(self, curves, surfaces, params, volatility):
        Swaption.__init__(self, curves, surfaces, params)
        self._volatility = float(volatility)

    def volatility(self):
        return self._volatility


class CurveShockedSet(object):
    '''
    Wrap a CurveSet and add a tent shock to ONE physical curve, matched by
    name. Every .rate() call on that curve -- from the underlying's discount
    leg and from its float forward projection alike -- sees the shocked rates;
    every other curve passes through to the underlying (unshocked) set. This
    mirrors the RiskWatch risk factor: the shock is applied to the curve
    itself, once, wherever the deal reads it.
    '''

    def __init__(self, base, shock_fn, shocked_curve):
        self._base = base
        self._shock_fn = shock_fn
        self._shocked = u'{0}'.format(shocked_curve).strip()

    def rate(self, curve_name, t_days):
        z = self._base.rate(curve_name, t_days)
        if u'{0}'.format(curve_name).strip() == self._shocked:
            return np.asarray(z, dtype=float) + self._shock_fn(t_days)
        return z

    def __getattr__(self, name):
        return getattr(self._base, name)


class ParallelShockedSet(object):
    '''
    Wrap a CurveSet and add a constant parallel shift to its curves.

    RiskWatch's GIRR curvature risk factor sits at the currency bucket, not at
    a single curve (its Risk Factor ID is 'EUR' / 'USD' with no vertex), so
    every curve carrying that bucket's risk shifts together and the option is
    repriced once per direction.

    `names` limits the shift to those curves; None shifts every curve. A curve
    the deal READS without it being a GIRR risk factor has to be left still --
    it is a projection input, and shifting it would load its move onto the
    currency bucket (see physical_curves).
    '''

    def __init__(self, base, shift, names=None):
        self._base = base
        self._shift = float(shift)
        self._names = None if names is None else set(
            u'{0}'.format(n).strip() for n in names)

    def rate(self, curve_name, t_days):
        z = self._base.rate(curve_name, t_days)
        if self._names is not None and \
                u'{0}'.format(curve_name).strip() not in self._names:
            return z
        return np.asarray(z, dtype=float) + self._shift

    def __getattr__(self, name):
        return getattr(self._base, name)


# ---------------------------------------------------------------------------
# Scenario inputs
# ---------------------------------------------------------------------------
def build_swaption_specs(port):
    '''
    Priced swaptions in a SwaptionPortfolio -> sensitivity specs
    (id + params + base volatility).

    Runs off port.swaptions, so every deal in scope has already passed the
    portfolio's curve and surface availability checks and carries a volatility
    interpolated off its own surface at the base moneyness. That base
    volatility is the one every scenario is priced on.
    '''
    if not port.swaptions:
        port.price()

    specs = []
    for deal_num in port.swaptions:
        swo = port.swaptions[deal_num]
        specs.append({
            'id': deal_num,
            'params': swo.params,
            'base_vol': swo.volatility(),
        })
    return specs


def fx_factor(fx, params):
    '''Factor restating a deal's sensitivity into the reporting currency.'''
    if fx is None:
        return 1.0
    return fx.factor(params['currency'])


def physical_curves(params, non_risk=()):
    '''
    Distinct physical curves the deal reads, in role order, less any named in
    `non_risk`.

    A curve can be READ by the pricer without being a GIRR RISK FACTOR.
    'EUR-SWP-1M' is one: it appears nowhere in the FRTB SA report's 42 GIRR
    Risk Factor IDs, and RiskWatch publishes only EUR-SWP delta rows for the
    deal that forecasts off it. Its own curvature confirms the same -- shifting
    the deal's discount curve alone reproduces RiskWatch exactly, while
    shifting the forecast curve with it misses by 2300%.

    The list is configured in swaptionMain.py and is passed by the CURVATURE
    pass only. The GIRR delta pass still shocks every curve the deal reads and
    reports a row per curve, so a computed sensitivity is never dropped for
    want of a RiskWatch counterpart; those rows simply reconcile to N/A.
    '''
    skip = set(u'{0}'.format(n).strip() for n in non_risk)
    phys = []
    for c in (u'{0}'.format(params['discount_curve']).strip(),
              u'{0}'.format(params['forecast_curve']).strip()):
        if c and c not in phys and c not in skip:
            phys.append(c)
    return phys


# ---------------------------------------------------------------------------
# RiskWatch reconciliation
# ---------------------------------------------------------------------------
def norm_id(s):
    '''int id 1001 and float-string '1001.0' both reconcile to '1001'.'''
    s = u'{0}'.format(s).strip()
    if s.endswith('.0') and s[:-2].isdigit():
        s = s[:-2]
    return s


def label_to_years(label):
    '''".25Y" / "0.25Y" / "6M" / "10Y" -> years.'''
    s = u'{0}'.format(label).strip().upper()
    if s.endswith('M'):
        return float(s[:-1]) / 12.0
    if s.endswith('Y'):
        body = s[:-1]
        if body.startswith('.'):
            body = '0' + body
        return float(body)
    return float(s)


def safe_years(label):
    '''label_to_years, with an unreadable label sorting last.'''
    try:
        return label_to_years(label)
    except (ValueError, TypeError):
        return float('inf')


def pct_diff(our, rw):
    '''(our / rw - 1) * 100, or NaN if either side is missing or rw is 0.'''
    if our is None or rw is None:
        return float('nan')
    try:
        if pd.isna(our) or pd.isna(rw):
            return float('nan')
    except (TypeError, ValueError):
        pass
    if rw == 0:
        return float('nan')
    return (our / rw - 1.0) * 100.0


def swaption_rw_id(s):
    '''
    Reconcile a swaption id to its DealNum. RiskWatch tags swaption rows as
    "SWO <DealNum>"; strip the tag and any quotes. Our tables are already
    keyed on DealNum, so this is a no-op for our ids.
    '''
    s = u'{0}'.format(s).strip()
    if s.upper().startswith(RW_SWAPTION_PREFIX):
        s = s[len(RW_SWAPTION_PREFIX):].strip().strip('\'"').strip()
    return norm_id(s)


def attach_riskwatch(table, our_col, rw_vals, sens_round, pct_round):
    '''
    Attach the RiskWatch column and the error column to a sensitivity table.

    'Delta-UAT' -> 'Delta-RiskWatch' + '(Delta-UAT/RW-1)%'.

    EVERY computed cell is kept, whether or not it matched a RiskWatch row: a
    deal absent from the FRTB SA report still carries its own sensitivity, and
    dropping it would hide a computed figure behind a reporting gap. An
    unmatched cell carries None in both attached columns and writes as 'N/A'.
    The reconciliation summary counts the matched cells separately so the two
    are never confused.
    '''
    rw_col = our_col.replace('-UAT', '-RiskWatch')
    err_col = '({0}/RW-1)%'.format(our_col)
    table[rw_col] = [round(v, sens_round) if v is not None else None
                     for v in rw_vals]
    table[err_col] = [round(pct_diff(u, v), pct_round)
                      for u, v in zip(table[our_col], table[rw_col])]
    return table
