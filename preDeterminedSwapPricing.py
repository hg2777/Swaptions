# -*- coding: utf-8 -*-
"""
Vanilla predetermined interest rate swap pricing.

@author: E42656
"""

import re

import pandas as pd
from dateutil.relativedelta import MO, TH
from pandas.tseries.offsets import DateOffset
from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday,
                                    GoodFriday, EasterMonday, nearest_workday,
                                    next_monday, next_monday_or_tuesday)

pd.set_option('display.max_columns', None)


# ---------------------------------------------------------------------------
# Payment-date holiday calendars: US federal, UK bank and EU TARGET holidays.
# Payment dates roll Following over weekends and the holidays of the DEAL's
# own currency (USD -> US federal, GBP -> UK bank, EUR -> EU TARGET; a
# currency without a mapped calendar contributes weekends only) -- the same
# convention as ccsPricing.
# ---------------------------------------------------------------------------
class _USHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year', month=1, day=1, observance=nearest_workday),
        Holiday('MLK Day', month=1, day=1, offset=DateOffset(weekday=MO(3))),
        Holiday('Presidents Day', month=2, day=1, offset=DateOffset(weekday=MO(3))),
        Holiday('Memorial Day', month=5, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday('Juneteenth', month=6, day=19, start_date='2021-06-19',
                observance=nearest_workday),
        Holiday('Independence Day', month=7, day=4, observance=nearest_workday),
        Holiday('Labor Day', month=9, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday('Columbus Day', month=10, day=1, offset=DateOffset(weekday=MO(2))),
        Holiday('Veterans Day', month=11, day=11, observance=nearest_workday),
        Holiday('Thanksgiving', month=11, day=1, offset=DateOffset(weekday=TH(4))),
        Holiday('Christmas', month=12, day=25, observance=nearest_workday),
    ]


class _UKHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year', month=1, day=1, observance=next_monday),
        GoodFriday,
        EasterMonday,
        Holiday('Early May Bank Holiday', month=5, day=1,
                offset=DateOffset(weekday=MO(1))),
        Holiday('Spring Bank Holiday', month=5, day=31,
                offset=DateOffset(weekday=MO(-1))),
        Holiday('Summer Bank Holiday', month=8, day=31,
                offset=DateOffset(weekday=MO(-1))),
        Holiday('Christmas', month=12, day=25, observance=next_monday),
        Holiday('Boxing Day', month=12, day=26,
                observance=next_monday_or_tuesday),
    ]


class _EUHolidays(AbstractHolidayCalendar):
    '''EU TARGET closing days (fixed calendar days, no weekend observance).'''
    rules = [
        Holiday('New Year', month=1, day=1),
        GoodFriday,
        EasterMonday,
        Holiday('Labour Day', month=5, day=1),
        Holiday('Christmas', month=12, day=25),
        Holiday('Goodwill Day', month=12, day=26),
    ]


_CCY_CALENDARS = {'USD': _USHolidays, 'GBP': _UKHolidays, 'EUR': _EUHolidays}
# 'Business Day Rule' calendar tags -> holiday calendar. The tag carried by
# the rule field (CalEUR / CalUSD / CalGBP) selects the calendar, superseding
# the deal's Currency field. An unmapped tag (e.g. CalCHF) passes through as
# its ISO code and contributes weekends only, per _holiday_set.
_CALENDAR_TAGS = {'CALEUR': 'EUR', 'CALUSD': 'USD', 'CALGBP': 'GBP'}
_HOLIDAY_RANGE = ('1990-01-01', '2099-12-31')
_HOLIDAYS = {}


def _holiday_set(currencies=None):
    '''Union of the holiday calendars mapped to `currencies` (an iterable
    of ISO codes) over the working range, built once per combination.
    Unmapped currencies contribute no holidays; None -> every mapped
    calendar (the legacy US/UK/EU union).'''
    if currencies is None:
        ccys = tuple(sorted(_CCY_CALENDARS))
    else:
        ccys = tuple(sorted(
            set(u'{0}'.format(c).strip().upper() for c in currencies)
            & set(_CCY_CALENDARS)))
    if ccys not in _HOLIDAYS:
        days = set()
        for c in ccys:
            cal = _CCY_CALENDARS[c]()
            for d in cal.holidays(pd.Timestamp(_HOLIDAY_RANGE[0]),
                                  pd.Timestamp(_HOLIDAY_RANGE[1])):
                days.add(pd.Timestamp(d).normalize())
        _HOLIDAYS[ccys] = frozenset(days)
    return _HOLIDAYS[ccys]


# ---------------------------------------------------------------------------
# 'Business Day Rule' field: "Regular|Modified Following x-day (CalEUR)"
#
#   Regular / Modified : Modified rolls BACKWARD either when the forward roll
#                        would land in a later month, or when the date itself
#                        sits in the run of non-business days that OPENS its
#                        month (see apply_business_day_rule); Regular always
#                        rolls forward.
#   x-day              : business days stepped on top of the roll. x=0 is the
#                        next business day (the first business day on or after
#                        the schedule date), x=1 the one after that, and so on.
#                        Under a Modified roll-back the same x steps BACKWARD.
#   (CalXXX)           : the holiday calendar, read here rather than from the
#                        deal's Currency field.
# ---------------------------------------------------------------------------
_BDR_OFFSET_RE = re.compile(r'(-?\d+)\s*-?\s*day')
_BDR_CALENDAR_RE = re.compile(r'cal\s*([a-z]{3})')


def parse_business_day_rule(value, default_currency=None):
    '''"Modified Following 0-day (CalEUR)" -> (modified, offset, calendar).

        modified : True for 'Modified ...', False for 'Regular ...'
        offset   : the x of "x-day", as a business-day count (>= 0)
        calendar : ISO code behind the CalXXX tag; falls back to
                   `default_currency` when the field carries no tag.

    A blank / missing field parses to (False, 0, default_currency): plain
    Following on the deal currency's calendar, i.e. the legacy behaviour.
    '''
    s = u'{0}'.format(value).strip().lower()

    modified = 'modified' in s

    m = _BDR_OFFSET_RE.search(s)
    offset = abs(int(m.group(1))) if m else 0

    calendar = default_currency
    c = _BDR_CALENDAR_RE.search(s)
    if c:
        code = c.group(1).upper()
        calendar = _CALENDAR_TAGS.get('CAL' + code, code)

    return modified, offset, calendar


# ---------------------------------------------------------------------------
# 'Coupon Generation Method' field: Forward | Backward
#
#   Forward  : boundaries step FORWARD from the leg's Real Start Date and the
#              schedule ends on the maturity date (_schedule_boundaries).
#   Backward : boundaries step BACKWARD from the maturity date and the
#              schedule opens on the leg's Real Start Date
#              (_schedule_boundaries_backward).
#
# A blank / missing field is Forward, so a workbook without the column keeps
# the legacy schedule.
# ---------------------------------------------------------------------------
COUPON_GENERATION_BACKWARD = ('backward', 'backwards', 'back', 'b')
COUPON_GENERATION_FORWARD = ('forward', 'forwards', 'fwd', 'f', '')


def parse_coupon_generation(value):
    '''"Backward" -> True, "Forward" / blank -> False.

    An unrecognised value is rejected rather than silently defaulted: a
    schedule generated from the wrong end is not a difference a reconciliation
    would attribute correctly.
    '''
    s = u'{0}'.format('' if value is None else value).strip().lower()
    if s in ('nan', 'none'):
        s = ''
    if s in COUPON_GENERATION_BACKWARD:
        return True
    if s in COUPON_GENERATION_FORWARD:
        return False
    raise ValueError(
        "Coupon Generation Method {0!r} is neither 'Forward' nor "
        "'Backward'".format(value))


def _business_day(d, holidays, step_days, offset=0):
    '''First business day on or after (step_days +1) / on or before (-1) `d`,
    then `offset` further business days in that same direction.'''
    delta = pd.Timedelta(days=step_days)
    while d.weekday() >= 5 or d in holidays:
        d = d + delta
    for _ in range(offset):
        d = d + delta
        while d.weekday() >= 5 or d in holidays:
            d = d + delta
    return d


def _opens_the_month(d, holidays):
    '''
    True when `d` is a non-business day AND every day from the 1st of its
    month up to `d` is a non-business day too -- i.e. `d` sits inside the
    unbroken run of weekend/holiday dates that OPENS the month, with no
    business day yet behind it in that month.

    2026-08-02 (Sunday) qualifies: 2026-08-01 is a Saturday, so the month has
    not opened for business. 2026-08-03 (Monday) does not -- it is itself a
    business day. A holiday on the 4th does not either, because the 3rd was a
    business day and the run is broken.
    '''
    probe = pd.Timestamp(d).normalize()
    first = probe.replace(day=1)
    step = pd.Timedelta(days=1)
    while probe >= first:
        if probe.weekday() < 5 and probe not in holidays:
            return False
        probe = probe - step
    return True


def apply_business_day_rule(dt, holidays, modified=False, offset=0):
    '''Roll one schedule date under the Business Day Rule.

    Forward branch: the next business day (first business day on or after
    `dt`), then `offset` further business days forward. Regular always keeps
    that result.

    Under Modified the roll is taken BACKWARD instead -- the previous business
    day, then `offset` business days back -- in either of two cases:

      1. the forward result falls in a LATER month (the classic Modified
         Following roll-back), or
      2. `dt` sits in the run of non-business days that OPENS its month
         (_opens_the_month), so rolling forward would carry a date belonging
         to the turn of the month into the new month rather than settling it
         on the old month's last business day.

    Case 2 fires where case 1 cannot: a date on the 1st or 2nd rolls forward
    WITHIN its own month, so the later-month test never trips. 2026-08-02 is a
    Sunday behind a Saturday 1st, and settles on Friday 2026-07-31 rather than
    Monday 2026-08-03.
    '''
    d = pd.Timestamp(dt).normalize()

    forward = _business_day(d, holidays, 1, offset)
    if modified and ((forward.year, forward.month) != (d.year, d.month)
                     or _opens_the_month(d, holidays)):
        return _business_day(d, holidays, -1, offset)
    return forward

def _act_act_isda(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    
    if end <= start: 
        return 0
    total = 0.0
    cursor = start
    while cursor < end:
        next_year = pd.Timestamp(year=cursor.year + 1, month=1, day=1)
        seg_end = min(end, next_year)
        days_in_year = 366 if cursor.is_leap_year else 365
        total += (seg_end - cursor).days / float(days_in_year)
        cursor = seg_end
    return total

def _thirty_360_us(start, end):
    '''US 30/360 (bond basis) day-count, expressed in years.

    Differs from the European 30E/360 in how the 31st is truncated: only the
    START date is unconditionally pulled back to the 30th; the END date is
    pulled back only when the (adjusted) start day is already 30, so an end
    date on the 31st against a start earlier in the month keeps its extra day.
    '''
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    d1 = start.day
    d2 = end.day
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    return (360 * (end.year - start.year)
            + 30 * (end.month - start.month)
            + (d2 - d1)) / 360.0
    
def year_fraction(valuation_date, payment_date, basis):
    valuation_date = pd.Timestamp(valuation_date)
    payment_date = pd.Timestamp(payment_date)
    
    days = (payment_date - valuation_date).days
    b = str(basis).strip().lower()
    
    if b in ('actual/360', 'act/360', 'actual360'):
        return days / 360.0
    elif b in ('actual/365', 'act/365', 'actual365'):
        return days / 365.0
    elif b in ('30/360', 'us 30/360', '30u/360'):
        return _thirty_360_us(valuation_date, payment_date)
    elif b in ('actual/actual', 'act/act', 'actualactual'):
        return _act_act_isda(valuation_date, payment_date)
    
    
    return days / 365.0


def swap_year_fraction(start, end, basis):
    '''Year fraction for a swap leg. 30/360 -> US 30/360 (bond basis); rest ->
    bond engine.'''
    b = str(basis).strip().lower().replace('smp', '').strip()
    if b in ('30/360', '30u/360', 'us 30/360', '360/360', 'bond'):
        return _thirty_360_us(start, end)
    return year_fraction(start, end, b)


def _on_day(dt, day):
    '''`dt` moved to `day` of its OWN month, clamped to that month's last day
    (so a 31st anchor lands on 30 / 28 / 29 in shorter months).'''
    dt = pd.Timestamp(dt)
    return dt.replace(day=min(int(day), dt.days_in_month))


class Swap:

    def __init__(self, curves, params):
        self.curves = curves
        self.params = params

        self.valuation = pd.Timestamp(params['valuation_date'])
        self.notional = float(params['notional'])

        # full working frames (all helper columns kept for internal use)
        self.fixed_full = self._value_leg('fixed')
        self.float_full = self._value_leg('float')

        # ----------------------------------------------------------- curve / DF
    def _adjust_payment_date(self, dt, holidays, adjust=True,
                             modified=False, offset=0):
        '''Business Day Rule roll (see apply_business_day_rule): Regular or
        Modified Following over `holidays`, plus the rule's x-day offset.
        adjust False -> leave the unadjusted schedule date.'''
        d = pd.Timestamp(dt).normalize()
        if not adjust:
            return d
        return apply_business_day_rule(d, holidays, modified, offset)

    def _schedule_boundaries(self, term_years, real_start_date,
                             match_maturity_day):
        '''Period boundaries generated FORWARD from the leg's Real Start Date.

        The schedule steps one `term_years` period at a time from the Real
        Start Date (the start of the leg's first / live accrual period) and
        always ends on the maturity date. Where the stub falls is set by
        `match_maturity_day`:

        match_maturity_day True  -- every generated boundary carries the DAY of
            the maturity date, so the FIRST cash flow runs from the real start
            date to that day and every later period is regular. This is the
            fixed leg always, and the float leg when the real start date's day
            is greater than the maturity date's day.

        match_maturity_day False -- the boundaries carry the day of the real
            start date, so every period is regular and the FINAL cash flow runs
            on to the maturity date. This is the float leg when the real start
            date's day is smaller than the maturity date's day. (The two are
            identical when the days match.)
        '''
        start = pd.Timestamp(real_start_date)
        maturity = pd.Timestamp(self.params['maturity_date'])
        months = int(round(float(term_years) * 12))

        day = maturity.day if match_maturity_day else start.day

        boundaries = [start]
        i = 1
        while True:
            nxt = _on_day(start + DateOffset(months=months * i), day)
            if nxt >= maturity:
                break
            if not match_maturity_day:
                # back stub: the last regular date is absorbed into the final
                # cash flow, which runs on to maturity
                follow = _on_day(start + DateOffset(months=months * (i + 1)), day)
                if follow > maturity:
                    break
            boundaries.append(nxt)
            i += 1
        boundaries.append(maturity)          # ascending: real start ... maturity
        return boundaries

    def _schedule_boundaries_backward(self, term_years, real_start_date):
        '''Period boundaries generated BACKWARD from the maturity date.

        The schedule is anchored on the MATURITY date and steps one
        `term_years` period back at a time, every boundary carrying the day of
        the maturity date (clamped into shorter months by _on_day). Each
        boundary is subtracted from the maturity date itself rather than from
        the previous boundary, so a month-end anchor cannot drift.

        Stepping stops once a subtraction reaches the leg's Real Start Date:
        the opening boundary is the LATER of that last subtraction and the
        real start date, so the schedule can never open before the leg does.
        Where the subtraction lands short of the real start date the opening
        period is a front stub; where it lands exactly on it every period is
        regular.

        The boundaries returned are UNADJUSTED schedule dates, as for the
        forward generator; _build_leg rolls them under the leg's Business Day
        Rule to get the payment dates.
        '''
        start = pd.Timestamp(real_start_date)
        maturity = pd.Timestamp(self.params['maturity_date'])
        months = int(round(float(term_years) * 12))
        if months <= 0:
            return [start, maturity]

        day = maturity.day
        boundaries = [maturity]
        i = 1
        while True:
            prev = _on_day(maturity - DateOffset(months=months * i), day)
            if prev <= start:
                break
            boundaries.append(prev)
            i += 1

        # the opening boundary: the later of the last subtraction and the
        # leg's own start date
        boundaries.append(max(prev, start))
        boundaries.reverse()                 # ascending: real start ... maturity
        return boundaries

    def _build_leg(self, leg):
        '''Schedule + accrual for one leg (cash flows still to settle).

        The discount/payment date always rolls under this leg's Business Day
        Rule. The accrual -- and, for the float leg, the forward projection
        (see _value_leg) -- rolls with this leg's calendar_adjustment flag:
        rolled when True, left on the unadjusted schedule when False.
        '''
        maturity = pd.Timestamp(self.params['maturity_date'])
        if leg == 'fixed':
            term, basis = self.params['fixed_term_years'], self.params['fixed_basis']
            adjust = bool(self.params.get('fixed_calendar_adjustment', False))
            rule = self.params.get('fixed_business_day_rule')
            real_start = pd.Timestamp(self.params['fixed_real_start_date'])
            backward = parse_coupon_generation(
                    self.params.get('fixed_coupon_generation'))
            # fixed leg: the first cash flow always matches the maturity day
            match_maturity_day = True
        else:
            term, basis = self.params['float_term_years'], self.params['float_basis']
            adjust = bool(self.params.get('float_calendar_adjustment', False))
            rule = self.params.get('float_business_day_rule')
            real_start = pd.Timestamp(self.params['float_real_start_date'])
            backward = parse_coupon_generation(
                    self.params.get('float_coupon_generation'))
            # float leg: a real start day PAST the maturity day puts the stub
            # at the front (first cash flow runs to the maturity day); a real
            # start day BEFORE it puts the stub at the back (final cash flow
            # runs on to maturity).
            match_maturity_day = real_start.day > maturity.day

        # Coupon Generation Method picks which end the schedule is built from.
        # Backward is anchored on the maturity date, so the stub placement
        # rules of the forward generator (match_maturity_day) do not apply.
        if backward:
            boundaries = self._schedule_boundaries_backward(term, real_start)
        else:
            boundaries = self._schedule_boundaries(term, real_start,
                                                   match_maturity_day)
        # The Business Day Rule names its own calendar (CalEUR / CalUSD /
        # CalGBP); the Currency field is only the fallback for a leg whose
        # rule is blank.
        modified, offset, calendar = parse_business_day_rule(
                rule, self.params.get('currency'))
        holidays = _holiday_set([calendar])

        rows = []
        for i in range(len(boundaries) - 1):
            rows.append({
                'period_start':  boundaries[i],      # unadjusted schedule start
                'period_end':    boundaries[i + 1],  # unadjusted schedule end
            })
        df = pd.DataFrame(rows)

        # keep only cash flows that settle after the valuation date
        df = df[df['period_end'] > self.valuation].reset_index(drop=True)
        if df.empty:
            return df
        
        # 'Calendar Adjustment' from the CSV: when True, roll the period dates to
        # the next business day (Following); when False, keep the unadjusted
        # schedule. The accrual, the float forward projection and the
        # discount/payment date all use these SAME (rolled-or-not) dates.
        df['adj_start'] = df['period_start'].apply(
            lambda d: self._adjust_payment_date(d, holidays, adjust,
                                                modified, offset))
        df['adj_end'] = df['period_end'].apply(
            lambda d: self._adjust_payment_date(d, holidays, adjust,
                                                modified, offset))
        # the discount/payment date ALWAYS rolls under the Business Day Rule,
        # regardless of the accrual adjustment flag
        df['payment_date'] = df['period_end'].apply(
            lambda d: self._adjust_payment_date(d, holidays, True,
                                                modified, offset))
        df['accrual'] = [swap_year_fraction(s, e, basis)
                         for s, e in zip(df['adj_start'], df['adj_end'])]
        
        return df

    def _days(self, dt):
        return (pd.Timestamp(dt) - self.valuation).days

    def _zero_and_df(self, curve_name, dt):
        d = self._days(dt)
        if d <= 0:
            return 0.0, 1.0
        z = float(self.curves.rate(curve_name, d))
        df = 1.0 / (1.0 + z) ** (d / 365.0)
        return z, df

    def _forward_rate(self, start, end, basis):
        tau = swap_year_fraction(start, end, basis)
        if tau <= 0:
            return 0.0
        _, df_start = self._zero_and_df(self.params['forecast_curve'], start)
        _, df_end = self._zero_and_df(self.params['forecast_curve'], end)
        return (df_start / df_end - 1.0) / tau

    def _period_is_reset(self, period_start):
        '''
        True when a floating period carries a KNOWN fixing (the last reset
        rate) rather than a projected forward.

        A period is fixed once it has started -- i.e. the valuation date is
        strictly PAST its start date. The first accrual period (the one
        STARTING at the leg's real start date) therefore uses the last reset
        rate only when the valuation date is after that date; on/before it, the
        period -- like every future period -- is projected forward off the
        forecast curve.
        '''
        return self.valuation > pd.Timestamp(period_start)

    def _value_leg(self, leg):
        df = self._build_leg(leg)
        if df.empty:
            return df

        disc_curve = self.params['discount_curve']

        # payment_date (business-day adjusted) is built in _build_leg;
        # discounting is done off that adjusted date.
        df['days'] = df['payment_date'].apply(self._days)
        zd = df['payment_date'].apply(lambda d: self._zero_and_df(disc_curve, d))
        df['disc_rate'] = [z for z, _ in zd]
        df['disc_df'] = [f for _, f in zd]

        if leg == 'fixed':
            df['rate'] = float(self.params['fixed_rate'])
        else:
            basis = self.params['float_basis']
            # Float-leg spread (decimal): added on top of the projected forward
            # rate. An already-reset period uses its known fixing (the Last
            # Reset Rate) as stored.
            spread = float(self.params.get('float_spread', 0.0) or 0.0)
            rates = []
            for _, r in df.iterrows():
                if self._period_is_reset(r['period_start']):
                    # first accrual period has already started -> use the known
                    # last reset rate as stored
                    rates.append(float(self.params['last_reset_rate']))
                else:
                    # not yet started -> project the forward over the SAME
                    # interval as the accrual, then add the spread
                    rates.append(self._forward_rate(
                        r['adj_start'], r['adj_end'], basis) + spread)
            df['rate'] = rates

        df['cash_flow'] = self.notional * df['rate'] * df['accrual']
        df['pv'] = df['cash_flow'] * df['disc_df']
        return df

    def fixed_leg_pv(self):
        return float(self.fixed_full['pv'].sum()) if not self.fixed_full.empty else 0.0

    def float_leg_pv(self):
        return float(self.float_full['pv'].sum()) if not self.float_full.empty else 0.0

    def _is_payer(self):
        '''True if we PAY fixed (payer / "paying" swap).'''
        pos = str(self.params.get('position', 'pay')).strip().lower()
        return pos in ('pay', 'payer', 'pay_fixed', 'paying')

    def npv(self):
        '''
        Dirty NPV from the holder's perspective.
            payer    : receive float, pay fixed  -> float - fixed
            receiver : receive fixed, pay float  -> fixed - float
        '''
        if self._is_payer():
            return self.float_leg_pv() - self.fixed_leg_pv()
        return self.fixed_leg_pv() - self.float_leg_pv()
