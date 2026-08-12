# -*- coding: utf-8 -*-
"""
Swaption book pricing from a three-tab population workbook, reconciled against
the RiskWatch FRTB SA report.

Reads one Excel workbook with the same TRANSPOSED layout as the vanilla swap
population workbook (field label per row in column A, one deal per column),
the three records matched on DealNum:

  * a 'Swaption' tab (the option itself)
  * a 'Fixed'    tab (Swap Fixed Leg of the underlying)
  * a 'Float'    tab (Swap Pre-determined Leg of the underlying)

Scope (current):
  * European swaptions on VANILLA underlyings -- the Variable Notional field
    carries a single constant notional; multi-step (amortizing) schedules are
    skipped

Pricing is delegated entirely to swaptionPricing.Swaption (which in turn builds
the underlying's legs with preDeterminedSwapPricing.Swap); this module reads
the tabs, applies the scope filter, matches the records, translates each into
the params dict Swaption understands, and (optionally) compares our MtM to the
'Mark To Market' column of the FRTB SA report, where swaptions appear in
'Instrument ID' as "SWO <DealNum>".

FIELDS ABSENT FROM THE CURRENT WORKBOOK.  The population workbook carries no
day-count basis and no calendar-adjustment flag for either leg. Both fall back
to the explicit module constants below (FIXED_BASIS_DEFAULT /
FLOAT_BASIS_DEFAULT / CALENDAR_ADJUSTMENT_DEFAULT) and every deal that falls
back is reported once, so the defaults are never applied silently. Should the
workbook grow a 'Day Count Basis' or 'Calendar Adjustment' row it is read in
preference to the constant.

RISKWATCH IS NOT AN INPUT.  The Swaption tab's 'Volatility' field is
RiskWatch's OUTPUT implied volatility for the deal, not a market input; it is
read for reporting only and never enters the price. The volatility used is
always the one interpolated off the surface named by the deal.

Targets Python 2.7 (no f-strings, explicit float division, object base).

@author: E42656
"""

import re
from collections import OrderedDict

import pandas as pd

from swaptionPricing import Swaption

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

INPUT_XLSX = 'Swaption_Input.xlsx'
SWAPTION_SHEET = 'Swaption'
FIXED_SHEET = 'Fixed'
FLOAT_SHEET = 'Float'
VALUATION_DATE = '2025-12-31'

# RiskWatch FRTB SA report (set to None to skip the MtM comparison).
RW_MTM_CSV = None
RW_INSTRUMENT_COL = 'Instrument ID'
RW_MTM_COL = 'Mark To Market'
# Instrument ID tag carried by swaption rows of the FRTB report.
RW_SWAPTION_PREFIX = 'SWO'

# Strike Price is quoted in PERCENT (1.2 -> 0.012).
STRIKE_IS_PERCENT = True

# Float-leg spread, when the workbook carries one, is a DECIMAL (0.001 = 10bp).
SPREAD_IS_PERCENT = False

# The Swaption tab's 'Volatility' row is RiskWatch's OUTPUT implied vol, quoted
# in PERCENT ('0.7417278602 % SMP actual/365'). Restated to a decimal so the
# reported column is directly comparable to the volatility we interpolate.
RW_VOL_IS_PERCENT = True

# --- defaults for fields the current workbook does not carry ----------------
# Day-count bases. RiskWatch quotes the underlying's coupons as
# '% SMP 30/360' (fixed) and '% SMP actual/360' (float).
FIXED_BASIS_DEFAULT = '30/360'
FLOAT_BASIS_DEFAULT = 'Actual/360'
# Calendar Adjustment. RiskWatch carries True on the swaption and on both legs
# of the underlying, so accrual and projection run on the ROLLED dates.
CALENDAR_ADJUSTMENT_DEFAULT = True

# Map a workbook curve-index name to the curve-tab / RiskWatch risk-factor
# name. Unmapped names pass through the '-6M' strip below. Set in the runner.
CURVE_ALIASES = {}

# Strings that mean "no value" (blank cells, Excel error tokens).
_NA_TOKENS = ('', 'nan', 'none', 'na', 'n/a', '#value!', '#n/a', '#ref!',
              '#div/0!', '#name?', '#num!', '#null!')

_TRUE_TOKENS = ('true', 't', '1', 'yes', 'y', 'call', 'payer', 'pay')
_FALSE_TOKENS = ('false', 'f', '0', 'no', 'n', 'put', 'receiver', 'receive')


def _u(v):
    '''Coerce a CSV header or cell to unicode without tripping Python 2.7's
    implicit ASCII decode. Under 2.7 pandas hands back CSV text as bytes, so a
    non-ASCII byte inside u'{0}'.format(v) would raise UnicodeDecodeError.
    utf-8 first, latin-1 as a never-fail fallback.'''
    if isinstance(v, bytes):
        try:
            return v.decode('utf-8')
        except UnicodeDecodeError:
            return v.decode('latin-1')
    return u'{0}'.format(v)


def _norm_header(col):
    return re.sub(r'[^a-z0-9]', '', _u(col).lower())


def _norm_id(v):
    s = u'{0}'.format(v).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s


def _is_blank(value):
    return _u(value).strip().lower() in _NA_TOKENS


def _resolve_col(df, wanted):
    '''Find a column by normalised name (tolerant of case / trailing spaces).'''
    target = _norm_header(wanted)
    for c in df.columns:
        if _norm_header(c) == target:
            return c
    return None


# ---------------------------------------------------------------------------
# Field parsers
# ---------------------------------------------------------------------------
def _clean_num(x):
    '''Parse a numeric cell; blanks / Excel error tokens -> None.

    Thousands separators are stripped. Unlike the swap reader's helper a comma
    is NOT read as a decimal point, because the swaption workbook quotes
    notionals in the '500,000,000.0000000000' form.
    '''
    s = _u(x).replace('%', '').replace(',', '').strip()
    if s.lower() in _NA_TOKENS:
        return None
    return float(s)


def _maybe_num(x):
    '''Parse a numeric cell, returning None for anything unparseable rather
    than raising. Used where a cell is being TESTED for a number -- scanning
    the tokens of a notional schedule, where a date or a currency tag is an
    expected non-number.'''
    try:
        return _clean_num(x)
    except (TypeError, ValueError):
        return None


def to_rate(x, rate_is_percent=STRIKE_IS_PERCENT):
    '''Percent (or decimal) cell -> decimal rate. Missing -> 0.0.'''
    v = _clean_num(x)
    if v is None:
        return 0.0
    return v / 100.0 if rate_is_percent else v


def to_spread(x, spread_is_percent=SPREAD_IS_PERCENT):
    '''Float-leg spread cell -> decimal. Missing / blank -> 0.0.'''
    v = _clean_num(x)
    if v is None:
        return 0.0
    return v / 100.0 if spread_is_percent else v


def parse_flag(value, default=None):
    '''Boolean-ish cell -> bool. Unknown / blank -> `default`.

    Accepts the Excel booleans as well as the call/put and payer/receiver
    wordings, so a workbook that spells the direction out still reads.
    '''
    if isinstance(value, bool):
        return value
    s = _u(value).strip().lower()
    if s in _TRUE_TOKENS:
        return True
    if s in _FALSE_TOKENS:
        return False
    return default


def term_to_years(term):
    '''"6 Months" / "12-Months" / "6M" -> 0.5 / 1.0 / 0.5 (years).'''
    digits = re.findall(r'\d+', _u(term))
    if not digits:
        raise ValueError('Cannot read a term from {0!r}'.format(term))
    return int(digits[0]) / 12.0


def parse_notional_field(raw):
    '''
    'Variable Notional' -> (notional, n_entries, final_date).

    Three quotations appear across the tabs and all three are handled:

        500000000                                   plain number
        ' 500,000,000.0000000000'                   thousands separators
        '2030/08/27 500,000,000.0000000000 EUR'     dated, with a currency tag

    A brace/pipe schedule string (amortizing) reports its entry count so the
    vanilla filter can skip multi-step notionals, and its last date -- the
    underlying swap's maturity -- is returned as `final_date`.
    '''
    s = _u(raw).strip().strip('{}').strip()
    if s.lower() in _NA_TOKENS:
        raise ValueError('Empty notional field')

    v = _maybe_num(s)
    if v is not None:
        return v, 1, None

    entries = []
    for chunk in s.split('|'):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split()
        date = pd.to_datetime(parts[0], errors='coerce')
        rest = parts[1:] if not pd.isna(date) else parts
        amount = None
        for tok in rest:
            amount = _maybe_num(tok)         # skips the trailing currency tag
            if amount is not None:
                break
        if amount is None:
            raise ValueError(
                'No notional amount in {0!r}'.format(chunk))
        entries.append((date, amount))

    if not entries:
        raise ValueError('Empty notional schedule: {0!r}'.format(raw))
    entries.sort(key=lambda p: (pd.Timestamp.min if pd.isna(p[0]) else p[0]))
    last_date = entries[-1][0]
    return (entries[0][1], len(entries),
            None if pd.isna(last_date) else pd.Timestamp(last_date))


# ---------------------------------------------------------------------------
# Three-tab population workbook reader (transposed: field per row, deal per col)
# ---------------------------------------------------------------------------
_SHEET_CANON = {
    'name':                 'name',
    'type':                 'leg_type',
    'calloption':           'call_option',
    'maturitydate':         'maturity_date',
    'effectivedate':        'effective_date',
    'realstartdate':        'real_start_date',
    'variablenotional':     'notional_field',
    'strikeprice':          'strike_price',
    'term':                 'term',
    'currency':             'currency',
    'discountcurve':        'discount_curve',
    'underlyingcurveindex': 'curve_index',
    'volatilitysurface':    'vol_surface',
    'volatility':           'volatility_rw',
    'businessdayrule':      'business_day_rule',
    'coupongenerationmethod': 'coupon_generation_method',
    'daycountbasis':        'day_count_basis',
    'calendaradjustment':   'calendar_adjustment',
    'lastresetrate':        'last_reset_rate',
    # float-leg spread over the index (optional row); absent -> 0.0
    'spread':               'float_spread',
    'floatspread':          'float_spread',
    'basisspread':          'float_spread',
}

# Name prefixes stripped when recovering a DealNum from a record's Name.
_NAME_PREFIXES = ('swaption_', 'swaption ', 'swo ', 'swn ')


def _deal_num_from_name(name):
    '''"Underlying Float Leg of P1565300" / "Swaption_P1565300" -> "P1565300".'''
    s = _u(name).strip()
    idx = s.lower().rfind('leg of ')
    if idx != -1:
        s = s[idx + len('leg of '):].strip()
    else:
        low = s.lower()
        for p in _NAME_PREFIXES:
            if low.startswith(p):
                s = s[len(p):].strip()
                break
    return _norm_id(s)


def _read_swaption_sheet(path, sheet):
    '''
    Read one transposed tab into a list of per-deal dict records keyed by
    canonical field names. Column A holds the field labels; each further column
    is one deal. The top index row and any column without a Name are ignored.
    '''
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    if raw.shape[1] < 2:
        return []

    keys = []
    for lab in raw.iloc[:, 0]:
        if lab is None or (isinstance(lab, float) and pd.isna(lab)):
            keys.append(None)
        else:
            keys.append(_SHEET_CANON.get(_norm_header(lab)))

    records = []
    for j in range(1, raw.shape[1]):
        rec = {}
        for i, key in enumerate(keys):
            if key is not None:
                rec[key] = raw.iat[i, j]
        name = rec.get('name')
        if name is None or _is_blank(name):
            continue                       # empty trailing column
        rec['deal_num'] = _deal_num_from_name(name)
        records.append(rec)
    return records

def read_surface_names(path, sheet=SWAPTION_SHEET):
    '''
    Reads the 'Volatility Surface' tab in the curves sheet.
    '''    
    names = []
    for rec in _read_swaption_sheet(path, sheet):
        name = _u(rec.get('vol_surface', '')).strip()
        if name and not _is_blank(name) and name not in names:
            names.append(name)
    return names


def load_rw_swaption_mtm(path, instrument_col=RW_INSTRUMENT_COL,
                         mtm_col=RW_MTM_COL, prefix=RW_SWAPTION_PREFIX):
    '''
    {DealNum -> Mark To Market} from the FRTB SA report.

    Swaption rows carry Instrument ID like "SWO P1565300"; the leading tag (and
    any quotes) are stripped to recover the DealNum. Rows of every other
    instrument type are ignored.
    '''
    df = pd.read_csv(path, dtype=str)
    inst_c = _resolve_col(df, instrument_col)
    mtm_c = _resolve_col(df, mtm_col)
    if inst_c is None or mtm_c is None:
        raise KeyError(
            'FRTB report missing columns {0!r}/{1!r}. Found: {2}'.format(
                instrument_col, mtm_col, list(df.columns)))

    tag = _u(prefix).strip().upper()
    out = {}
    for _, r in df.iterrows():
        inst = _u(r[inst_c]).strip()
        if not inst.upper().startswith(tag):
            continue
        deal = _norm_id(inst[len(tag):].strip().strip('\'"').strip())
        mtm = _clean_num(r[mtm_c])
        if deal and mtm is not None:
            out[deal] = mtm
    return out


# ---------------------------------------------------------------------------
# Portfolio: load -> filter to vanilla -> match on DealNum -> price
# ---------------------------------------------------------------------------
class SwaptionPortfolio(object):

    def __init__(self, curves, surfaces, input_xlsx=INPUT_XLSX,
                 swaption_sheet=SWAPTION_SHEET, fixed_sheet=FIXED_SHEET,
                 float_sheet=FLOAT_SHEET, valuation_date=VALUATION_DATE,
                 strike_is_percent=STRIKE_IS_PERCENT,
                 spread_is_percent=SPREAD_IS_PERCENT,
                 rw_vol_is_percent=RW_VOL_IS_PERCENT,
                 rw_mtm_csv=RW_MTM_CSV, rw_instrument_col=RW_INSTRUMENT_COL,
                 rw_mtm_col=RW_MTM_COL, rw_prefix=RW_SWAPTION_PREFIX,
                 only_ids=None, curve_aliases=CURVE_ALIASES,
                 fixed_basis_default=FIXED_BASIS_DEFAULT,
                 float_basis_default=FLOAT_BASIS_DEFAULT,
                 calendar_adjustment_default=CALENDAR_ADJUSTMENT_DEFAULT,
                 fx=None):
        self.curves = curves
        self.surfaces = surfaces
        self.input_xlsx = input_xlsx
        self.swaption_sheet = swaption_sheet
        self.fixed_sheet = fixed_sheet
        self.float_sheet = float_sheet
        self.valuation_date = valuation_date
        self.strike_is_percent = strike_is_percent
        self.spread_is_percent = spread_is_percent
        self.rw_vol_is_percent = rw_vol_is_percent

        # defaults for rows the workbook does not carry; every fallback is
        # collected and reported once by price()
        self.fixed_basis_default = fixed_basis_default
        self.float_basis_default = float_basis_default
        self.calendar_adjustment_default = calendar_adjustment_default

        # workbook curve-index name -> curve-tab / RiskWatch name
        self.curve_aliases = dict(curve_aliases) if curve_aliases else {}
        self.fx = fx

        self.rw_mtm_csv = rw_mtm_csv
        self.rw_instrument_col = rw_instrument_col
        self.rw_mtm_col = rw_mtm_col
        self.rw_prefix = rw_prefix

        # optional troubleshooting filter: restrict the book to these DealNums
        self.only_ids = (set(_norm_id(i) for i in only_ids)
                         if only_ids else None)

        self.swaptions = OrderedDict()    # deal_num -> swaptionPricing.Swaption
        self.skipped = []                 # (deal_num, reason)
        self.rw_mtm = {}                  # deal_num -> RiskWatch MtM
        self.results = None               # DataFrame, set by price()
        self.warnings = []                # (deal_num, message)

    # -- scope filter --------------------------------------------------------
    @staticmethod
    def _is_vanilla(row):
        '''Vanilla = a single (constant) notional entry.'''
        try:
            return parse_notional_field(row.get('notional_field'))[1] == 1
        except Exception:
            return False

    def _alias_curve(self, name):
        '''Resolve a workbook curve-index name to a curve-tab name.

        An explicit curve_aliases entry wins. Otherwise a '6 months' index name
        (carrying a '-6M' tenor tag) is not bootstrapped separately, so the
        '-6M' tag is REMOVED and any variant suffix is preserved:
        'EUR-SWP-6M' -> 'EUR-SWP'. All other names, including the discount
        curves, pass through unchanged. Matches the vanilla swap book.'''
        key = _u(name).strip()
        if key in self.curve_aliases:
            return self.curve_aliases[key]
        return key.replace('-6M', '')

    def _warn(self, deal_num, message):
        self.warnings.append((deal_num, message))

    # -- inputs --------------------------------------------------------------
    def load_triples(self):
        '''[(swaption_rec, fixed_rec, float_rec), ...] matched on DealNum.'''
        options = _read_swaption_sheet(self.input_xlsx, self.swaption_sheet)
        fixed = _read_swaption_sheet(self.input_xlsx, self.fixed_sheet)
        floating = _read_swaption_sheet(self.input_xlsx, self.float_sheet)

        fixed_by_num = {}
        for r in fixed:
            fixed_by_num[r['deal_num']] = r
        float_by_num = {}
        for r in floating:
            float_by_num[r['deal_num']] = r

        triples = []
        self.skipped = []
        for op in options:
            num = op['deal_num']
            if self.only_ids is not None and num not in self.only_ids:
                continue                      # outside the troubleshooting scope
            fx = fixed_by_num.get(num)
            fl = float_by_num.get(num)
            if fx is None:
                self.skipped.append((num, 'no matching fixed leg'))
            elif fl is None:
                self.skipped.append((num, 'no matching float leg'))
            elif not self._is_vanilla(fx):
                self.skipped.append(
                    (num, 'fixed leg not vanilla (multi-step notional)'))
            elif not self._is_vanilla(fl):
                self.skipped.append(
                    (num, 'float leg not vanilla (multi-step notional)'))
            else:
                triples.append((op, fx, fl))
        return triples

    def _basis(self, rec, default, deal_num, leg):
        '''Day-count basis for one leg. The current workbook carries no basis
        row, so the explicit module default applies and the fallback is
        recorded for the run summary.'''
        raw = rec.get('day_count_basis')
        if not _is_blank(raw):
            return _u(raw).strip()
        self._warn(deal_num,
                   '{0} leg day-count basis not in workbook -> {1!r}'.format(
                       leg, default))
        return default

    def _calendar_adjustment(self, rec, deal_num, leg):
        '''Calendar-adjustment flag for one leg, with the same explicit
        fallback treatment as the day-count basis.'''
        flag = parse_flag(rec.get('calendar_adjustment'), None)
        if flag is not None:
            return flag
        self._warn(deal_num,
                   '{0} leg calendar adjustment not in workbook -> {1!r}'.format(
                       leg, self.calendar_adjustment_default))
        return self.calendar_adjustment_default

    def _maturity(self, fx, fl, deal_num):
        '''Underlying swap maturity: the Float tab's Maturity Date, else the
        Fixed tab's.

        The Swaption tab carries no notional row, so its notional schedule is
        no longer available as a last-resort source of the final date; a deal
        with no Maturity Date on either leg tab is a blotter error and is
        raised rather than inferred.
        '''
        for rec, tab in ((fl, 'Float'), (fx, 'Fixed')):
            raw = rec.get('maturity_date')
            if not _is_blank(raw):
                return pd.Timestamp(pd.to_datetime(raw)), tab
        raise ValueError(
            'No Maturity Date on either leg tab for deal {0}'.format(deal_num))

    def _notional(self, fx, fl, deal_num):
        '''Underlying notional, taken from the FIXED leg.

        The Swaption tab carries no notional row: the option and its
        underlying are the same trade, so the notional is the legs' own. The
        fixed leg's is used and a float leg quoting a different amount is
        flagged rather than silently dropped, mirroring _discount_curve.
        '''
        fixed_n = parse_notional_field(fx.get('notional_field'))[0]
        try:
            float_n = parse_notional_field(fl.get('notional_field'))[0]
        except Exception:
            float_n = None
        if float_n is not None and float_n != fixed_n:
            self._warn(deal_num,
                       'legs quote different notionals ({0:,.2f} fixed / '
                       '{1:,.2f} float); priced on the fixed leg'.format(
                           fixed_n, float_n))
        return fixed_n

    def _discount_curve(self, fx, fl, deal_num):
        '''One discount curve drives both legs of the underlying. The fixed
        leg's is used; a float leg quoting a different curve is flagged rather
        than silently dropped.'''
        fixed_curve = self._alias_curve(fx.get('discount_curve', ''))
        float_curve = self._alias_curve(fl.get('discount_curve', ''))
        if float_curve and float_curve != fixed_curve:
            self._warn(deal_num,
                       'legs quote different discount curves ({0} fixed / {1} '
                       'float); priced on {0}'.format(fixed_curve, float_curve))
        return fixed_curve

    def _build_params(self, op, fx, fl):
        '''Translate a matched swaption/fixed/float triple into a Swaption
        params dict.'''
        deal_num = op['deal_num']
        notional = self._notional(fx, fl, deal_num)
        maturity, _ = self._maturity(fx, fl, deal_num)

        expiry = pd.Timestamp(pd.to_datetime(op['maturity_date']))
        fixed_start = pd.Timestamp(pd.to_datetime(fx['real_start_date']))
        float_start = pd.Timestamp(pd.to_datetime(fl['real_start_date']))

        # The swaption's Effective Date is the underlying's start; the legs
        # carry their own Real Start Dates and those drive the schedules. A
        # disagreement is a data issue for the blotter, so it is flagged.
        eff = op.get('effective_date')
        if not _is_blank(eff):
            eff = pd.Timestamp(pd.to_datetime(eff))
            if eff != fixed_start or eff != float_start:
                self._warn(deal_num,
                           'Effective Date {0:%Y-%m-%d} differs from a leg '
                           'Real Start Date (fixed {1:%Y-%m-%d} / float '
                           '{2:%Y-%m-%d}); schedules run off the leg '
                           'dates'.format(eff, fixed_start, float_start))

        is_call = parse_flag(op.get('call_option'), None)
        if is_call is None:
            raise ValueError(
                "Deal {0}: unreadable 'Call Option' flag {1!r}".format(
                    deal_num, op.get('call_option')))

        return {
            # identification / reporting
            'id':                deal_num,
            'deal_num':          deal_num,
            'instrument_type':   _u(op.get('leg_type', 'Swaption')).strip()
                                 or 'Swaption',
            'currency':          _u(op.get('currency', '')).strip(),

            # option terms consumed by swaptionPricing.Swaption
            'valuation_date':    self.valuation_date,
            'notional':          notional,
            'is_call':           is_call,
            'strike':            to_rate(op.get('strike_price'),
                                         self.strike_is_percent),
            'expiry_date':       expiry,
            'vol_surface_name':  _u(op.get('vol_surface', '')).strip(),
            # RiskWatch's OUTPUT implied vol -- reporting only, never priced on
            'volatility_rw':     to_rate(op.get('volatility_rw'),
                                         self.rw_vol_is_percent),

            # underlying swap, consumed by preDeterminedSwapPricing.Swap
            'position':          'pay' if is_call else 'receive',
            'fixed_real_start_date': fixed_start,
            'float_real_start_date': float_start,
            'maturity_date':     maturity,
            'fixed_rate':        to_rate(op.get('strike_price'),
                                         self.strike_is_percent),
            'fixed_term_years':  term_to_years(fx['term']),
            'float_term_years':  term_to_years(fl['term']),
            'fixed_basis':       self._basis(fx, self.fixed_basis_default,
                                             deal_num, 'fixed'),
            'float_basis':       self._basis(fl, self.float_basis_default,
                                             deal_num, 'float'),
            'fixed_calendar_adjustment':
                self._calendar_adjustment(fx, deal_num, 'fixed'),
            'float_calendar_adjustment':
                self._calendar_adjustment(fl, deal_num, 'float'),
            'fixed_business_day_rule':
                _u(fx.get('business_day_rule', '')).strip(),
            'float_business_day_rule':
                _u(fl.get('business_day_rule', '')).strip(),
            # Coupon Generation Method: which end the schedule is built from.
            # Blank -> Forward, i.e. the legacy schedule (see
            # preDeterminedSwapPricing.parse_coupon_generation).
            'fixed_coupon_generation':
                _u(fx.get('coupon_generation_method', '')).strip(),
            'float_coupon_generation':
                _u(fl.get('coupon_generation_method', '')).strip(),
            'float_index':       _u(fl.get('curve_index', '')).strip(),
            'float_spread':      to_spread(fl.get('float_spread'),
                                           self.spread_is_percent),
            # The underlying starts after the valuation date, so every floating
            # period projects; no fixing is consumed. A workbook that carries
            # one is honoured.
            'last_reset_rate':   (float('nan')
                                  if _is_blank(fl.get('last_reset_rate'))
                                  else _clean_num(fl.get('last_reset_rate'))),
            'discount_curve':    self._discount_curve(fx, fl, deal_num),
            'forecast_curve':    self._alias_curve(fl.get('curve_index', '')),
        }

    # -- pricing -------------------------------------------------------------
    def price(self):
        '''Price every vanilla swaption; return the summary DataFrame.'''
        triples = self.load_triples()
        self.warnings = []

        self.rw_mtm = {}
        if self.rw_mtm_csv:
            self.rw_mtm = load_rw_swaption_mtm(self.rw_mtm_csv,
                                               self.rw_instrument_col,
                                               self.rw_mtm_col,
                                               self.rw_prefix)
        compare = bool(self.rw_mtm_csv)

        self.swaptions = OrderedDict()
        rows = []
        available = set(self.curves.curves.keys())
        for op, fx, fl in triples:
            try:
                params = self._build_params(op, fx, fl)
            except Exception as e:
                self.skipped.append((op['deal_num'],
                                     'unreadable inputs: {0}'.format(e)))
                continue

            # skip cleanly when the deal cannot/should not be priced here
            missing = [c for c in (params['discount_curve'],
                                   params['forecast_curve'])
                       if c not in available]
            if missing:
                self.skipped.append((params['deal_num'],
                                     'curve(s) not loaded: {0}'.format(missing)))
                continue
            if not self.surfaces.has(params['vol_surface_name']):
                self.skipped.append(
                    (params['deal_num'],
                     'volatility surface {0!r} not loaded: {1}'.format(
                         params['vol_surface_name'],
                         self.surfaces.reason(params['vol_surface_name']))))
                continue

            err = ''
            try:
                swo = Swaption(self.curves, self.surfaces, params)
                self.swaptions[params['deal_num']] = swo
                float_pv = swo.float_leg_pv()
                annuity = swo.annuity()
                swap_yield = swo.swap_yield()
                moneyness = swo.moneyness()
                vol = swo.volatility()
                t_exp = swo.time_to_expiry()
                h = swo.h()
                mtm = swo.npv()
                if not swo.vol_is_interpolated():
                    self._warn(params['deal_num'],
                               'expiry / moneyness outside the {0} grid -> '
                               'volatility clamped to a grid edge'.format(
                                   params['vol_surface_name']))
            except Exception as e:
                float_pv = annuity = swap_yield = moneyness = float('nan')
                vol = t_exp = h = mtm = float('nan')
                err = str(e)

            # Restate the float leg PV and the MtM from the deal's own currency
            # into the EUR reporting currency (RiskWatch reports in reporting
            # currency). No FX converter -> factor 1.0, so a EUR book is
            # unchanged. The annuity, yield, moneyness, vol and h are rates or
            # year fractions and are currency-free.
            #
            # A currency the 'FX rates' tab does not quote skips THAT DEAL with
            # a recorded reason; it must not take the rest of the book down
            # with it, and the deal must not be reported in its own currency
            # beside deals restated into EUR -- a column mixing units is worse
            # than a missing row.
            try:
                fx_factor = (self.fx.factor(params['currency'])
                             if self.fx is not None else 1.0)
            except KeyError as e:
                self.skipped.append((params['deal_num'],
                                     'no FX rate to the reporting currency: '
                                     '{0}'.format(e)))
                continue
            float_pv = float_pv * fx_factor
            mtm = mtm * fx_factor

            row = OrderedDict([
                ('DealNum',        params['deal_num']),
                ('ID',             params['id']),
                ('Type',           params['instrument_type']),
                ('Currency',       params['currency']),
                ('Option',         'CALL' if params['is_call'] else 'PUT'),
                ('Position',       'Payer of Fixed Leg' if params['is_call']
                                   else 'Receiver of Fixed Leg'),
                ('Notional',       params['notional']),
                ('Strike',         params['strike']),
                ('Expiry',         params['expiry_date']),
                ('Swap Start',     params['float_real_start_date']),
                ('Swap Maturity',  params['maturity_date']),
                ('Float Leg PV',   round(float_pv, 2)),
                ('Annuity',        annuity),
                ('Swap Yield',     swap_yield),
                ('Moneyness',      moneyness),
                ('Vol Surface',    params['vol_surface_name']),
                ('Volatility',     vol),
                ('Vol-RiskWatch',  params['volatility_rw']),
                ('T (ACT/365)',    t_exp),
                ('h',              h),
            ])

            if compare:
                rw = self.rw_mtm.get(params['deal_num'], float('nan'))
                if pd.notna(mtm) and pd.notna(rw) and rw != 0:
                    pct = (mtm / rw - 1.0) * 100.0
                else:
                    pct = float('nan')
                row['MtM-UAT'] = round(mtm, 2) if pd.notna(mtm) else mtm
                row['MtM-RiskWatch'] = round(rw, 2) if pd.notna(rw) \
                    else float('nan')
                row['(MtM-UAT/RW-1)%'] = round(pct, 6) if pd.notna(pct) \
                    else float('nan')
            else:
                row['MtM'] = round(mtm, 2) if pd.notna(mtm) else mtm

            row['Error'] = err
            rows.append(row)

        out = pd.DataFrame(rows)
        if 'Error' in out.columns and (out['Error'] == '').all():
            out = out.drop(columns=['Error'])
        self.results = out

        if self.skipped:
            from collections import Counter
            reasons = Counter(r for _, r in self.skipped)
            print('Skipped {0} deal(s):'.format(len(self.skipped)))
            for reason, n in reasons.most_common():
                print('  {0:>4d}  {1}'.format(n, reason))

        if self.warnings:
            from collections import Counter
            notes = Counter(m for _, m in self.warnings)
            print('Input notes ({0} across {1} deal(s)):'.format(
                len(self.warnings), len(set(d for d, _ in self.warnings))))
            for message, n in notes.most_common():
                print('  {0:>4d}  {1}'.format(n, message))

        return out

    def summary(self):
        '''The per-swaption results DataFrame (prices on first call if needed).'''
        if self.results is None:
            self.price()
        return self.results
