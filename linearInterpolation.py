# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:20:11 2026

Curve construction and interpolation.

@author: E42656
"""


import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

class Interpolation(object):

    LINEAR_ALIASES = ('linear', 'lin', 'l')
    CUBIC_ALIASES = ('cubic', 'cubicspline', 'cubic_spline', 'spline', 'c')

    def __init__(self, tenors_days, rates, method='linear'):
        x = np.asarray(tenors_days, dtype=float)
        y = np.asarray(rates, dtype=float)
        order = np.argsort(x)
        self.x = x[order]
        self.y = y[order]

        # CubicSpline needs strictly increasing x; drop duplicate tenors
        keep = np.concatenate(([True], np.diff(x) > 0))
        self.x = x[keep]
        self.y = y[keep]

        self.method = str(method).strip().lower()

        if self.method in self.CUBIC_ALIASES:
            if len(self.x) < 2:
                raise ValueError(
                        'Cubic interpolation needs at least 2 distinct tenors.')
            from scipy.interpolate import CubicSpline
            self._spline = CubicSpline(self.x, self.y)
        elif self.method in self.LINEAR_ALIASES:
            self._spline = None
        else:
            raise ValueError(
                    "Unknown interpolation method '{0}'. "
                    "Use 'linear' or 'cubic'.".format(method))

    def __call__(self, t_days):
        if self._spline is None:
            return np.interp(t_days, self.x, self.y)

        t = np.clip(np.asarray(t_days, dtype=float), self.x[0], self.x[-1])
        result = self._spline(t)

        if np.isscalar(t_days):
            return float(result)
        return result

def curve_from_frame(df, tenor_col=0, rate_col=1, method='linear'):
    sub = df.iloc[:, [tenor_col, rate_col]].copy()
    sub.columns = ['tenor', 'rate']
    sub['tenor'] = pd.to_numeric(sub['tenor'], errors='coerce')
    sub['rate'] = pd.to_numeric(sub['rate'], errors='coerce')
    sub = sub.dropna()
    return Interpolation(sub['tenor'].values, sub['rate'].values, method=method)


class CurveSet(object):
    def __init__(self, curves):
        self.curves = curves

    def rate(self, curve_name, t_days):
        key = str(curve_name).strip()

        if key not in self.curves:
            raise KeyError('Curve {0} not in curves. Current curves: {1}'.format(
                    key, list(self.curves)))

        return self.curves[key](t_days)


def _filter_tabs(sheet_names, exclude):
    exclude_lower = set(str(e).strip().lower() for e in exclude)
    return [s for s in sheet_names if s.strip().lower() not in exclude_lower]

def load_curve_set(path, curve_tabs=None, exclude=(), tenor_col=0, rate_col=1,
                   header=None, method='linear'):

    xls = pd.ExcelFile(path)
    if curve_tabs is None:
        curve_tabs = _filter_tabs(xls.sheet_names, exclude)

    curves = {}
    for name in curve_tabs:
        tab = pd.read_excel(path, sheet_name=name, header=header)
        curves[name.strip()] = curve_from_frame(tab, tenor_col, rate_col,
               method=method)
    return CurveSet(curves)


# ---------------------------------------------------------------------------
# FX rates (curves workbook 'FX rates' tab) -> EUR reporting-currency converter
# ---------------------------------------------------------------------------
def load_fx_rates(path, sheet='FX rates', header=None):
    '''
    Read the 'FX rates' tab (two columns: pair label, rate) into
    {PAIR -> rate}, e.g. {'EURUSD': 1.1742, 'EURCHF': 0.9304}. Feeds an
    FxConverter that restates MtM / GIRR into the EUR reporting currency.
    '''
    df = pd.read_excel(path, sheet_name=sheet, header=header)
    rates = {}
    for _, r in df.iterrows():
        pair = u'{0}'.format(r.iloc[0]).strip().upper()
        val = pd.to_numeric(r.iloc[1], errors='coerce') if df.shape[1] > 1 \
            else float('nan')
        if pair and not pd.isna(val):
            rates[pair] = float(val)
    return rates


class FxConverter(object):
    '''
    Restate an amount from a deal's own currency into the reporting currency
    (EUR) using the 'FX rates' tab.

    The tab quotes one rate per row as 'EURXXX' = units of XXX per 1 EUR
    (e.g. EURUSD = 1.1742 -> 1 EUR buys 1.1742 USD), so an amount in XXX is
    worth amount / rate in EUR. The reporting currency itself (and any blank
    currency) passes through with factor 1.0; an inverse 'XXXEUR' quote is also
    honoured if that is how a pair happens to be stored.
    '''
    def __init__(self, rates, reporting='EUR'):
        self.reporting = u'{0}'.format(reporting).strip().upper()
        self.rates = {}
        for k, v in dict(rates).items():
            key = u'{0}'.format(k).strip().upper()
            try:
                self.rates[key] = float(v)
            except (TypeError, ValueError):
                continue

    def factor(self, currency):
        '''Multiplicative factor taking an amount in `currency` to reporting.'''
        ccy = u'{0}'.format(currency).strip().upper()
        if ccy == '' or ccy == self.reporting:
            return 1.0
        direct = self.reporting + ccy        # 'EURUSD': units of XXX per 1 EUR
        if direct in self.rates and self.rates[direct] != 0.0:
            return 1.0 / self.rates[direct]
        inverse = ccy + self.reporting       # 'USDEUR': EUR per 1 XXX
        if inverse in self.rates:
            return float(self.rates[inverse])
        raise KeyError(
            "No FX rate to convert {0} into {1} (need '{2}' or '{3}')".format(
                ccy, self.reporting, direct, inverse))

    def to_reporting(self, amount, currency):
        '''Convert `amount` (deal currency) into the reporting currency.'''
        if amount is None:
            return amount
        return amount * self.factor(currency)