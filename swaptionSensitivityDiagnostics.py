# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 2026

Per-swaption sensitivity diagnostics workbook.

Writes a workbook (alongside swaption_results.xlsx) whose two tabs let an MtM
or sensitivity difference against RiskWatch be traced deal by deal, curve by
curve and scenario by scenario:

    'MtM Analytics'  : for each swaption, the underlying's priced FIXED leg and
                   FLOAT leg tables (schedule dates, accruals, rates, discount
                   factors, PVs), the Bachelier four-step trace (float leg PV,
                   annuity, swap yield, moneyness, volatility, T, h, cdf, pdf,
                   MtM) and a small summary carrying notional, strike, expiry
                   and position.

    'Sensitivities' : the scenario behind all THREE measures, in three
                   sections. This is where this workbook differs from the
                   vanilla swap one, which carries GIRR scenario curves alone.

                     GIRR      -- one block per altered curve (each deal's
                                  discount and forecast curve, de-duplicated):
                                  the unchanged curve ('BaseRate'), a parallel
                                  1bp column and one column per GIRR tenor
                                  showing the curve after that tenor's tent
                                  shock.
                     CURVATURE -- one block per altered curve showing the
                                  curve shifted up and down by the curvature
                                  shift, then a per-deal table carrying the
                                  base MtM, both shifted MtMs and the two CVR
                                  figures.
                     VEGA      -- a per-deal table carrying the base and
                                  shocked volatility, the two axis
                                  measurements in days, the base and shocked
                                  MtM and the vega P&L, then the weight grid:
                                  every (option term, swap duration) cell with
                                  its two weights, their product and the vega
                                  it carries.

The scenario tables are rebuilt here rather than read back off the sensitivity
run, so each block is an independent trace of the same shock rather than a
restatement of the answer.

Targets Python 2.7 (no f-strings, .format(), object base classes).

@author: E42656
"""

import pandas as pd

from sensitivity import (ONE_BP, build_sensitivity_table, get_curve_nodes,
                         tenors_from_days)
from swaptionScenario import (FixedVolSwaption, ParallelShockedSet,
                              build_swaption_specs, physical_curves)
from swaptionCurvature import CURVATURE_DOWN, CURVATURE_SHOCK, CURVATURE_UP
from swaptionVega import (VEGA_REL_SHOCK, VEGA_TENOR_DAYS, VEGA_TENOR_LABELS,
                          vega_axes, vega_tenor_weights)

ANALYTICS_SHEET = 'MtM Analytics'
SENSITIVITY_SHEET = 'Sensitivities'

# Per-leg columns shown in the analytics tables, in order. Filtered to those
# the pricer actually produced, so it is safe if a column is absent.
LEG_COLS = ['period_start', 'period_end', 'accrual_start', 'payment_date',
            'accrual', 'rate', 'disc_rate', 'disc_df', 'cash_flow', 'pv']

DATE_FMT = 'YYYY-MM-DD'


def _dates_only(frame):
    '''Return a copy with any datetime column reduced to a plain date.'''
    out = frame.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.date
    return out


def _put(xl, sheet, row, frame, header=True):
    '''Write a frame at `row` (date-only) and return the next free row index.'''
    _dates_only(frame).to_excel(xl, sheet_name=sheet, startrow=row,
                                index=False, header=header)
    return row + len(frame) + (1 if header else 0)


def altered_curves(specs):
    '''
    De-duplicated list of every curve that gets shocked across the book, in
    first-seen order: each deal's discount curve then its forecast curve.
    '''
    names = []
    for spec in specs:
        for nm in physical_curves(spec['params']):
            if nm not in names:
                names.append(nm)
    return names


def swaption_leg_tables(swo):
    '''(fixed-leg df, float-leg df) of a priced swaption's underlying swap.'''

    def _leg(df):
        if df is None or len(df) == 0:
            return pd.DataFrame()
        cols = [c for c in LEG_COLS if c in df.columns]
        return df[cols].copy()

    return _leg(swo.swap.fixed_full), _leg(swo.swap.float_full)


def _section(xl, row, title):
    '''Write a section banner and return the next free row index.'''
    return _put(xl, SENSITIVITY_SHEET, row,
                pd.DataFrame([[title]]), header=False)


def _girr_blocks(xl, row, curves, names, valuation_date, tenor_table,
                 method, shock):
    '''One tent-shock scenario block per altered curve.'''
    row = _section(xl, row, 'GIRR : tent-shocked scenario curves '
                            '({0}bp)'.format(shock * 10000.0))
    for name in names:
        nd, nr = get_curve_nodes(curves, name)
        table = build_sensitivity_table(valuation_date, nd, nr, tenor_table,
                                        method=method, shock=shock)
        row = _put(xl, SENSITIVITY_SHEET, row,
                   pd.DataFrame([['Curve: {0}'.format(name)]]), header=False)
        row = _put(xl, SENSITIVITY_SHEET, row, table)
        row += 1
    return row + 1


def _curvature_blocks(xl, row, curves, surfaces, specs, names, shift):
    '''Shifted curve nodes per altered curve, then the per-deal CVR trace.'''
    row = _section(xl, row, 'CURVATURE : parallel shift of {0:.10f} on every '
                            'curve of the currency'.format(shift))

    for name in names:
        nd, nr = get_curve_nodes(curves, name)
        table = pd.DataFrame({'Days': nd, 'BaseRate': nr})
        table['Up'] = table['BaseRate'] + shift
        table['Down'] = table['BaseRate'] - shift
        row = _put(xl, SENSITIVITY_SHEET, row,
                   pd.DataFrame([['Curve: {0}'.format(name)]]), header=False)
        row = _put(xl, SENSITIVITY_SHEET, row,
                   table[['Days', 'BaseRate', 'Up', 'Down']])
        row += 1

    rows = []
    for spec in specs:
        params, vol = spec['params'], spec['base_vol']
        base = FixedVolSwaption(curves, surfaces, params, vol).npv()
        up = FixedVolSwaption(ParallelShockedSet(curves, shift),
                              surfaces, params, vol).npv()
        down = FixedVolSwaption(ParallelShockedSet(curves, -shift),
                                surfaces, params, vol).npv()
        rows.append([spec['id'], params.get('currency', ''), vol,
                     round(base, 2), round(up, 2), round(down, 2),
                     round(up - base, 2), round(down - base, 2)])

    row = _put(xl, SENSITIVITY_SHEET, row, pd.DataFrame(
        rows, columns=['ID', 'Currency', 'Volatility', 'MtM Base', 'MtM Up',
                       'MtM Down', CURVATURE_UP, CURVATURE_DOWN]))
    return row + 2


def _vega_blocks(xl, row, curves, surfaces, specs, grid_days, grid_labels,
                 rel_shock):
    '''Per-deal vega P&L trace, then the (option term x swap duration) grid.'''
    row = _section(xl, row, 'VEGA : sigma -> sigma * {0}, spread over '
                            '{1}'.format(1.0 + rel_shock, list(grid_labels)))

    trace, grid = [], []
    for spec in specs:
        params, vol = spec['params'], spec['base_vol']
        option_days, duration_days = vega_axes(params)

        base = FixedVolSwaption(curves, surfaces, params, vol).npv()
        up = FixedVolSwaption(curves, surfaces, params,
                              vol * (1.0 + rel_shock)).npv()
        vega_pl = (up - base) / rel_shock

        trace.append([spec['id'], u'{0}'.format(
            params['vol_surface_name']).strip(), vol, vol * (1.0 + rel_shock),
            option_days, duration_days, round(base, 2), round(up, 2),
            round(vega_pl, 2)])

        for o_label, o_w in vega_tenor_weights(option_days, grid_days,
                                               grid_labels):
            for d_label, d_w in vega_tenor_weights(duration_days, grid_days,
                                                   grid_labels):
                grid.append([spec['id'], o_label, o_w, d_label, d_w,
                             o_w * d_w, round(o_w * d_w * vega_pl, 2)])

    row = _put(xl, SENSITIVITY_SHEET, row, pd.DataFrame(
        trace, columns=['ID', 'Surface', 'Volatility', 'Volatility Shocked',
                        'Option Term (days)', 'Swap Duration (days)',
                        'MtM Base', 'MtM Shocked', 'Vega PL']))
    row += 1
    row = _put(xl, SENSITIVITY_SHEET, row, pd.DataFrame(
        grid, columns=['ID', 'Option Term', 'Option Weight', 'Swap Duration',
                       'Duration Weight', 'Weight', 'Vega']))
    return row + 2


def write_swaption_sensitivity_diagnostics(
        path, port, tenor_days, tenor_labels, method='linear',
        girr_shock=ONE_BP, curvature_shock=CURVATURE_SHOCK,
        vega_tenor_days=VEGA_TENOR_DAYS, vega_tenor_labels=VEGA_TENOR_LABELS,
        vega_rel_shock=VEGA_REL_SHOCK, verbose=True):
    '''
    Build the two-tab sensitivity diagnostics workbook for the priced
    swaption book.

    tenor_days / tenor_labels come from swaptionMain.py, so the GIRR section
    shocks exactly the same grid as the sensitivity run.
    '''
    specs = build_swaption_specs(port)
    if not specs:
        if verbose:
            print('[swaptionSensitivityDiagnostics] no swaptions to write; '
                  'skipped {0}'.format(path))
        return path

    curves, surfaces = port.curves, port.surfaces
    tenor_table = tenors_from_days(tenor_days, tenor_labels,
                                   port.valuation_date)
    names = altered_curves(specs)

    with pd.ExcelWriter(path, engine='openpyxl',
                        date_format=DATE_FMT, datetime_format=DATE_FMT) as xl:
        # --- Analytics tab : legs, four-step trace and summary, per deal ----
        r = 0
        for spec in specs:
            params = spec['params']
            swo = port.swaptions[spec['id']]
            fixed_t, float_t = swaption_leg_tables(swo)

            header = ('Swaption {0}  ({1} {2})   {3}  discount={4}  '
                      'forecast={5}  surface={6}'.format(
                          spec['id'], params.get('currency', ''),
                          swo.option_type(), swo.position(),
                          params.get('discount_curve', ''),
                          params.get('forecast_curve', ''),
                          params.get('vol_surface_name', '')))
            r = _put(xl, ANALYTICS_SHEET, r,
                     pd.DataFrame([[header]]), header=False)

            r = _put(xl, ANALYTICS_SHEET, r,
                     pd.DataFrame([['UNDERLYING FIXED LEG']]), header=False)
            if len(fixed_t):
                r = _put(xl, ANALYTICS_SHEET, r, fixed_t)
            r += 1

            r = _put(xl, ANALYTICS_SHEET, r,
                     pd.DataFrame([['UNDERLYING FLOAT LEG']]), header=False)
            if len(float_t):
                r = _put(xl, ANALYTICS_SHEET, r, float_t)
            r += 1

            r = _put(xl, ANALYTICS_SHEET, r,
                     pd.DataFrame([['BACHELIER STEPS']]), header=False)
            steps = swo.steps()
            r = _put(xl, ANALYTICS_SHEET, r, pd.DataFrame(
                [[k, steps[k]] for k in steps], columns=['Step', 'Value']))
            r += 1

            summary = pd.DataFrame(
                [[round(float(params['notional']), 2), params['strike'],
                  pd.Timestamp(params['expiry_date']),
                  pd.Timestamp(params['maturity_date']),
                  swo.option_type(), swo.position(), round(swo.npv(), 2)]],
                columns=['Notional', 'Strike', 'Expiry', 'Swap Maturity',
                         'Option', 'Position', 'MtM'])
            r = _put(xl, ANALYTICS_SHEET, r, summary)
            r += 2

        # --- Sensitivities tab : all three measures -------------------------
        r = 0
        r = _girr_blocks(xl, r, curves, names, port.valuation_date,
                         tenor_table, method, girr_shock)
        r = _curvature_blocks(xl, r, curves, surfaces, specs, names,
                              curvature_shock)
        r = _vega_blocks(xl, r, curves, surfaces, specs, vega_tenor_days,
                         vega_tenor_labels, vega_rel_shock)

    if verbose:
        print('[swaptionSensitivityDiagnostics] wrote {0}  (tabs: {1}, {2}) '
              'for {3} swaptions, {4} curves'.format(
                  path, ANALYTICS_SHEET, SENSITIVITY_SHEET, len(specs),
                  len(names)))
    return path