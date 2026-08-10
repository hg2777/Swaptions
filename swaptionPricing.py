# -*- coding: utf-8 -*-
"""
European swaption pricing in the Bachelier (normal) framework.

The underlying is a vanilla pre-determined IRS, so its two legs are built by
preDeterminedSwapPricing.Swap -- the schedule generation, Business Day Rule
roll, day-count conventions and curve handling are shared with the vanilla swap
book rather than reimplemented here.

Pricing runs in four steps:

  1. FLOAT LEG NPV.  Project each floating period off the forecast curve and
     discount off the discount curve, exactly as for a pre-determined swap:

         V_float = SUM_i  N * f_i * tau_i * DF(t_i)

  2. SWAP YIELD.  Build the fixed-leg annuity on the FIXED leg's cash-flow
     dates and read the forward swap rate off the float leg:

         A = SUM_j  tau_j * DF(t_j)              (fixed-leg accruals + DFs)
         y = V_float / (N * A)

  3. MONEYNESS AND VOLATILITY.  Moneyness is quoted strike-over-forward,

         m = K - y

     and the normal volatility is read off the surface by bilinear
     interpolation at (option expiry, m): linear across moneyness, then linear
     in calendar days across the option-term axis (volatilitySurface).

  4. BACHELIER MtM.  With T = (expiry - valuation) / 365 and

         h = (y - K) / (sigma * sqrt(T))

     payer (call on the swap rate)    : [ (y - K) N(h)  + sigma sqrt(T) n(h) ]
     receiver (put on the swap rate)  : [ (K - y) N(-h) + sigma sqrt(T) n(h) ]

     each scaled by N * A. N(.) and n(.) are the standard normal cumulative and
     probability density functions.

The option premium is a long position and is non-negative for both types; the
payer/receiver flag selects the payoff, not a sign.

Targets Python 2.7 (no f-strings, explicit float division, object base).

@author: E42656
"""

import math

import pandas as pd
from scipy.stats import norm

from preDeterminedSwapPricing import Swap

pd.set_option('display.max_columns', None)

# Day basis for the option's time to expiry. RiskWatch quotes the swaption
# volatility as 'actual/365', so sqrt(T) runs on ACT/365 from the valuation
# date to the option's maturity (expiry) date.
VOL_TIME_BASIS = 365.0


class Swaption(object):
    '''
    European swaption on a vanilla pre-determined IRS.

    params (in addition to every key preDeterminedSwapPricing.Swap consumes):

        valuation_date   : pricing date
        notional         : option / underlying-swap notional
        is_call          : True  -> call on the swap rate = PAYER swaption
                           False -> put  on the swap rate = RECEIVER swaption
        strike           : strike rate as a DECIMAL (0.012 for 1.2%)
        expiry_date      : the option's maturity date
        vol_surface_name : surface tab consulted for the normal volatility

    `surface` is a volatilitySurface.SurfaceSet.
    '''

    def __init__(self, curves, surface, params):
        self.curves = curves
        self.surfaces = surface
        self.params = params

        self.valuation = pd.Timestamp(params['valuation_date'])
        self.notional = float(params['notional'])
        self.expiry = pd.Timestamp(params['expiry_date'])
        self.strike = float(params['strike'])
        self.is_call = bool(params['is_call'])

        # Underlying swap: both legs built by the shared vanilla IRS engine.
        self.swap = Swap(curves, params)

    # -- step 1 : float leg NPV ---------------------------------------------
    def float_leg_pv(self):
        '''Projected-and-discounted PV of the underlying's floating leg.'''
        return self.swap.float_leg_pv()

    # -- step 2 : swap yield -------------------------------------------------
    def annuity(self):
        '''Fixed-leg annuity SUM tau_j * DF(t_j) on the fixed cash-flow dates.

        Read off the fixed leg's own accruals and discount factors, so it
        carries the fixed leg's day-count basis, term and Business Day Rule.
        '''
        fixed = self.swap.fixed_full
        if fixed.empty:
            return 0.0
        return float((fixed['accrual'] * fixed['disc_df']).sum())

    def swap_yield(self):
        '''Forward par swap rate y = V_float / (N * A).'''
        a = self.annuity()
        if a == 0.0 or self.notional == 0.0:
            return 0.0
        return self.float_leg_pv() / self.notional / a

    # -- step 3 : moneyness and implied volatility ---------------------------
    def moneyness(self):
        '''Surface moneyness, quoted strike over forward: m = K - y.'''
        return self.strike - self.swap_yield()

    def volatility(self):
        '''Normal volatility read off the surface at (expiry, moneyness).'''
        return self.surfaces.vol(self.params['vol_surface_name'],
                                 self.expiry, self.moneyness())

    def vol_is_interpolated(self):
        '''False when (expiry, moneyness) sits outside the surface grid and
        the volatility is therefore clamped to an edge rather than
        interpolated.'''
        return self.surfaces.surface(
            self.params['vol_surface_name']).is_inside(
                self.expiry, self.moneyness())

    # -- step 4 : Bachelier MtM ---------------------------------------------
    def time_to_expiry(self):
        '''ACT/365 year fraction from the valuation date to option expiry.'''
        days = (self.expiry - self.valuation).days
        return days / VOL_TIME_BASIS

    def h(self):
        '''Normal moneyness parameter h = (y - K) / (sigma sqrt(T)).'''
        sigma, t = self.volatility(), self.time_to_expiry()
        denom = sigma * math.sqrt(t) if t > 0 else 0.0
        if denom == 0.0:
            return float('inf') if self.swap_yield() > self.strike \
                else float('-inf')
        return (self.swap_yield() - self.strike) / denom

    def cdf(self):
        '''N(h) for a payer, N(-h) for a receiver -- the exercise term.'''
        h = self.h()
        return float(norm.cdf(h)) if self.is_call else float(norm.cdf(-h))

    def pdf(self):
        '''n(h), the standard normal density -- the time-value term.'''
        h = self.h()
        if h in (float('inf'), float('-inf')):
            return 0.0
        return float(norm.pdf(h))

    def npv(self):
        '''
        Bachelier MtM of the option, from the holder's (long) perspective.

            payer    : [ (y - K) N(h)  + sigma sqrt(T) n(h) ] * N * A
            receiver : [ (K - y) N(-h) + sigma sqrt(T) n(h) ] * N * A
        '''
        y, k = self.swap_yield(), self.strike
        intrinsic = (y - k) if self.is_call else (k - y)
        sigma_root_t = self.volatility() * math.sqrt(max(self.time_to_expiry(),
                                                         0.0))
        return ((intrinsic * self.cdf() + sigma_root_t * self.pdf())
                * self.notional * self.annuity())

    # -- reporting -----------------------------------------------------------
    def option_type(self):
        return 'CALL' if self.is_call else 'PUT'

    def position(self):
        '''Direction on the underlying swap if the option is exercised.'''
        return 'Payer of Fixed Leg' if self.is_call else 'Receiver of Fixed Leg'

    def steps(self):
        '''The four pricing steps as an ordered dict, for diagnostics.'''
        from collections import OrderedDict
        return OrderedDict([
            ('Float Leg PV', self.float_leg_pv()),
            ('Annuity', self.annuity()),
            ('Swap Yield', self.swap_yield()),
            ('Strike', self.strike),
            ('Moneyness', self.moneyness()),
            ('Volatility', self.volatility()),
            ('T (ACT/365)', self.time_to_expiry()),
            ('h', self.h()),
            ('cdf', self.cdf()),
            ('pdf', self.pdf()),
            ('MtM', self.npv()),
        ])
