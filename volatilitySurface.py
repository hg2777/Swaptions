# -*- coding: utf-8 -*-
"""
Normal (Bachelier) swaption volatility surfaces.

A surface tab in the curves workbook (e.g. 'SWNVol-EUR-5Y') is a grid of
normal volatilities quoted against

    * an OPTION TERM axis  -- absolute expiry dates, across the columns
    * a MONEYNESS axis     -- strike minus forward swap yield, down the rows

Layout as read (header=None):

        A              B              C            ...
    1                  Option Term
    2   Moneyness      2026-01-30     2026-03-31   ...
    3   -0.05          0.0197759      0.01586189   ...
    4   -0.0475        0.01918666     0.01541597   ...

The header row is located by its first cell ('Moneyness'), so a tab with
leading blank/title rows reads without an offset argument.

Interpolation is BILINEAR and matches the reference workbook: linear in
moneyness within each expiry column, then linear in calendar days between the
two bracketing expiry columns. Outside the grid the surface CLAMPS (flat
extrapolation) on both axes rather than running the linear slope on -- a
linearly extrapolated normal vol can turn negative, which is not a price.
Clamping is not evidenced by the reference workbook (its deal interpolates
strictly inside the grid) and is flagged as an open item.

Targets Python 2.7 (no f-strings, explicit float division, object base).

@author: E42656
"""

import re

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# Header cell that identifies the moneyness column / the grid's first row.
_MONEYNESS_LABEL = 'moneyness'


def _norm_header(v):
    return re.sub(r'[^a-z0-9]', '', u'{0}'.format(v).lower())


class VolatilitySurface(object):
    '''
    Normal-volatility grid over (option expiry date, moneyness).

    expiries  : ascending list of Timestamps (the Option Term axis)
    moneyness : ascending 1-D array of moneyness levels (strike - swap yield)
    vols      : 2-D array shaped (len(moneyness), len(expiries))
    '''

    def __init__(self, name, expiries, moneyness, vols):
        self.name = u'{0}'.format(name).strip()

        exp = [pd.Timestamp(d) for d in expiries]
        mny = np.asarray(moneyness, dtype=float)
        grid = np.asarray(vols, dtype=float)

        if grid.shape != (len(mny), len(exp)):
            raise ValueError(
                'Surface {0}: vol grid is {1}, expected ({2}, {3})'.format(
                    self.name, grid.shape, len(mny), len(exp)))

        # Both axes ascending, so np.interp's clamping behaves as documented.
        col_order = np.argsort([d.value for d in exp])
        row_order = np.argsort(mny)
        self.expiries = [exp[i] for i in col_order]
        self.moneyness = mny[row_order]
        self.vols = grid[np.ix_(row_order, col_order)]

        # Expiry axis interpolated on calendar days. Ordinals are used as the
        # abscissa: only DIFFERENCES enter the linear weight, so the origin
        # cancels and no valuation date is needed here.
        self._expiry_days = np.array(
            [float(d.toordinal()) for d in self.expiries], dtype=float)

    def vol(self, expiry_date, moneyness):
        '''Normal volatility at (expiry_date, moneyness), bilinear + clamped.'''
        t = float(pd.Timestamp(expiry_date).toordinal())
        m = float(moneyness)

        # 1) collapse the moneyness axis inside every expiry column
        by_expiry = np.array(
            [np.interp(m, self.moneyness, self.vols[:, j])
             for j in range(self.vols.shape[1])], dtype=float)

        # 2) collapse the expiry axis
        return float(np.interp(t, self._expiry_days, by_expiry))

    def is_inside(self, expiry_date, moneyness):
        '''False when the point sits outside the grid and the vol is clamped.'''
        t = float(pd.Timestamp(expiry_date).toordinal())
        m = float(moneyness)
        return bool(self._expiry_days[0] <= t <= self._expiry_days[-1]
                    and self.moneyness[0] <= m <= self.moneyness[-1])

    def __repr__(self):
        return '<VolatilitySurface {0}: {1} expiries x {2} moneyness>'.format(
            self.name, len(self.expiries), len(self.moneyness))


def load_volatility_surface(path, sheet, header=None):
    '''Read one surface tab of the curves workbook into a VolatilitySurface.'''
    name = u'{0}'.format(sheet).strip()
    
    available = pd.ExcelFile(path).sheet_names
    if name not in available:
        raise KeyError(
                'no tab {0!r} in {1}. Tabs present: {2}'.format(
                        name, path, list(available)))
    raw = pd.read_excel(path, sheet_name=name, header=header)

    # Locate the header row by its 'Moneyness' label in the first column.
    head = None
    for i in range(raw.shape[0]):
        if _norm_header(raw.iat[i, 0]) == _MONEYNESS_LABEL:
            head = i
            break
    if head is None:
        raise ValueError(
            "Surface tab {0!r} has no {1!r} header cell in its first "
            "column".format(name, _MONEYNESS_LABEL.title()))

    # Expiry dates run across the header row; a column without a readable date
    # (a spacer or a stray label) is dropped together with its vol column.
    cols, expiries = [], []
    for j in range(1, raw.shape[1]):
        d = pd.to_datetime(raw.iat[head, j], errors='coerce')
        if pd.isna(d):
            continue
        cols.append(j)
        expiries.append(pd.Timestamp(d))
    if not cols:
        raise ValueError(
            'Surface tab {0!r} has no option-term dates on its header '
            'row'.format(name))

    body = raw.iloc[head + 1:, :]
    mny = pd.to_numeric(body.iloc[:, 0], errors='coerce')
    keep = mny.notna().values
    if not keep.any():
        raise ValueError(
            'Surface tab {0!r} has no numeric moneyness levels'.format(name))

    grid = body.iloc[:, cols].apply(pd.to_numeric, errors='coerce')
    return VolatilitySurface(name, expiries, mny[keep].values,
                             grid.values[keep, :])


def load_volatility_surfaces(path, sheets, header=None):
    '''{tab name -> VolatilitySurface} for the named surface tabs.'''
    out = {}
    for s in sheets:
        name = u'{0}'.format(s).strip()
        out[name] = load_volatility_surface(path, name, header=header)
    return out


class SurfaceSet(object):
    '''Name -> VolatilitySurface lookup, mirroring linearInterpolation's
    CurveSet so a missing surface fails with the same shape of message.'''

    def __init__(self, surfaces=None, path=None, header=None):
        self.surfaces = dict(surfaces) if surfaces else {}
        self.path = path
        self.header = header
        self.errors = {}  #tab names that failed to load
        
    def has(self, name):
        key = u'{0}'.format(name).strip()
        if key in self.surfaces:
            return True
        if key in self.errors:
            return False
        if not key:
            self.errors[key] = 'the deal names no volatility surface'
            return False
        if self.path is None:
            self.errors[key] = 'no working path'
            return False
        try:
            self.surfaces[key] = load_volatility_surface(self.path, key, 
                         header=self.header)
        except Exception as e:
            try:
                detail = u'{0}'.format(e)
            except Exception:
                detail = repr(e)
            self.errors[key] = u'{0}: {1}'.format(type(e).__name__, detail)
            return False
        return True
    
    def surface(self, name):
        key = u'{0}'.format(name).strip()
        if not self.has(key):
            raise KeyError(
                'Volatility surface {0} not loaded. Current surfaces: '
                '{1}'.format(key, self.path, sorted(self.surfaces)))
        return self.surfaces[key]

    def vol(self, name, expiry_date, moneyness):
        return self.surface(name).vol(expiry_date, moneyness)
