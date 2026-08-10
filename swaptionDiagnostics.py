# -*- coding: utf-8 -*-
"""
Per-swaption diagnostics workbook.

Writes one tab per deal laid out like the reference pricing workbook, so any
MtM difference against RiskWatch can be traced cell for cell:

    Step 1  float-leg cash-flow table  -- schedule dates, days, discount and
            forecast zero rates, their discount factors, the projected forward
            and the discounted cash flow
    Step 2  fixed-leg annuity table    -- accrual, discount factor and their
            product, plus the swap yield
    Step 3  volatility interpolation   -- the four bracketing surface corners,
            the moneyness-interpolated row, and the volatility read off it
    Step 4  Bachelier terms            -- T, h, cdf, pdf and the MtM

A leading 'Summary' tab carries one row per deal with the four steps side by
side.

Targets Python 2.7 (no f-strings, explicit float division, object base).

@author: E42656
"""

from collections import OrderedDict

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# Excel caps a sheet name at 31 characters.
_MAX_SHEET_NAME = 31


def _sheet_name(prefix, deal_num):
    name = u'{0}{1}'.format(prefix, deal_num)
    return name[:_MAX_SHEET_NAME]


def _leg_table(swaption, leg):
    '''Cash-flow table for one leg of the underlying, with both curves shown.

    Mirrors the reference workbook: the schedule dates, the day offset from the
    valuation date, the discount-curve zero and its DF, the forecast-curve zero
    and its DF, and (float only) the projected forward and its cash flow.
    '''
    swap = swaption.swap
    df = swap.fixed_full if leg == 'fixed' else swap.float_full
    if df.empty:
        return pd.DataFrame()

    disc_curve = swap.params['discount_curve']
    fore_curve = swap.params['forecast_curve']

    rows = []
    for _, r in df.iterrows():
        pay = r['payment_date']
        z_disc, df_disc = swap._zero_and_df(disc_curve, pay)
        z_fore, df_fore = swap._zero_and_df(fore_curve, pay)
        rows.append(OrderedDict([
            ('Period Start (unadj)', r['period_start']),
            ('Period End (unadj)',   r['period_end']),
            ('Accrual Start',        r['adj_start']),
            ('Accrual End',          r['adj_end']),
            ('Payment Date',         pay),
            ('Days',                 swap._days(pay)),
            ('{0} zero'.format(disc_curve), z_disc),
            ('{0} DF'.format(disc_curve),   df_disc),
            ('{0} zero'.format(fore_curve), z_fore),
            ('{0} DF'.format(fore_curve),   df_fore),
            ('Accrual',              r['accrual']),
            ('Rate',                 r['rate']),
            ('Cash Flow',            r['cash_flow']),
            ('PV',                   r['pv']),
        ]))
    out = pd.DataFrame(rows)

    if leg == 'fixed':
        # the annuity is the accrual-weighted sum of the discount factors
        out['Accrual x DF'] = out['Accrual'] * out['{0} DF'.format(disc_curve)]
    return out


def _vol_interpolation_table(swaption):
    '''The surface corners bracketing (expiry, moneyness) and the two linear
    interpolations that produce the volatility actually used.'''
    surface = swaption.surfaces.surface(swaption.params['vol_surface_name'])
    expiry = swaption.expiry
    m = swaption.moneyness()

    days = surface._expiry_days
    t = float(pd.Timestamp(expiry).toordinal())
    mny = surface.moneyness

    # bracketing indices, clamped at the grid edges
    j_hi = int(np.clip(np.searchsorted(days, t), 1, len(days) - 1))
    j_lo = j_hi - 1
    i_hi = int(np.clip(np.searchsorted(mny, m), 1, len(mny) - 1))
    i_lo = i_hi - 1

    rows = []
    for i in (i_lo, i_hi):
        rows.append(OrderedDict([
            ('Moneyness', float(mny[i])),
            ('{0:%Y-%m-%d}'.format(surface.expiries[j_lo]),
             float(surface.vols[i, j_lo])),
            ('Option expiry {0:%Y-%m-%d}'.format(expiry), None),
            ('{0:%Y-%m-%d}'.format(surface.expiries[j_hi]),
             float(surface.vols[i, j_hi])),
        ]))

    # the deal's own moneyness row: linear between the two bracketing rows
    v_lo = float(np.interp(m, mny, surface.vols[:, j_lo]))
    v_hi = float(np.interp(m, mny, surface.vols[:, j_hi]))
    rows.append(OrderedDict([
        ('Moneyness', m),
        ('{0:%Y-%m-%d}'.format(surface.expiries[j_lo]), v_lo),
        ('Option expiry {0:%Y-%m-%d}'.format(expiry), swaption.volatility()),
        ('{0:%Y-%m-%d}'.format(surface.expiries[j_hi]), v_hi),
    ]))
    return pd.DataFrame(rows)


def _steps_table(swaption):
    '''The four pricing steps as a two-column label/value table.'''
    steps = swaption.steps()
    labels = [
        ('Step 1  Float Leg NPV', 'Float Leg PV'),
        ('Step 2  Annuity', 'Annuity'),
        ('Step 2  Swap Yield', 'Swap Yield'),
        ('Step 3  Strike', 'Strike'),
        ('Step 3  Moneyness (K - y)', 'Moneyness'),
        ('Step 3  Volatility (normal)', 'Volatility'),
        ('Step 4  T (ACT/365)', 'T (ACT/365)'),
        ('Step 4  h', 'h'),
        ('Step 4  cdf', 'cdf'),
        ('Step 4  pdf', 'pdf'),
        ('Step 4  MtM', 'MtM'),
    ]
    rows = [OrderedDict([('Step', label), ('Value', steps[key])])
            for label, key in labels]
    rows.append(OrderedDict([('Step', 'Option type'),
                             ('Value', swaption.option_type())]))
    rows.append(OrderedDict([('Step', 'Position on exercise'),
                             ('Value', swaption.position())]))
    rows.append(OrderedDict([('Step', 'Notional'),
                             ('Value', swaption.notional)]))
    rows.append(OrderedDict([('Step', 'Volatility interpolated (not clamped)'),
                             ('Value', swaption.vol_is_interpolated())]))
    return pd.DataFrame(rows)


def write_swaption_diagnostics(path, portfolio):
    '''
    Write the diagnostics workbook for every priced swaption in `portfolio`.

    One 'Summary' tab plus, per deal, a 'Steps_', a 'Float_', a 'Fixed_' and a
    'Vol_' tab.
    '''
    swaptions = portfolio.swaptions
    if not swaptions:
        print('No priced swaptions -- diagnostics workbook not written.')
        return

    summary_rows = []
    for deal_num, swo in swaptions.items():
        row = OrderedDict([('DealNum', deal_num),
                           ('Option', swo.option_type()),
                           ('Notional', swo.notional),
                           ('Expiry', swo.expiry)])
        row.update(swo.steps())
        summary_rows.append(row)

    with pd.ExcelWriter(path) as xl:
        pd.DataFrame(summary_rows).to_excel(xl, sheet_name='Summary',
                                            index=False, na_rep='N/A')
        for deal_num, swo in swaptions.items():
            _steps_table(swo).to_excel(
                xl, sheet_name=_sheet_name('Steps_', deal_num),
                index=False, na_rep='N/A')
            _leg_table(swo, 'float').to_excel(
                xl, sheet_name=_sheet_name('Float_', deal_num),
                index=False, na_rep='N/A')
            _leg_table(swo, 'fixed').to_excel(
                xl, sheet_name=_sheet_name('Fixed_', deal_num),
                index=False, na_rep='N/A')
            _vol_interpolation_table(swo).to_excel(
                xl, sheet_name=_sheet_name('Vol_', deal_num),
                index=False, na_rep='N/A')

    print('written to {0}  ({1} deal(s) traced)'.format(path, len(swaptions)))
