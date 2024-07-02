import pandas as pd
import numpy as np
from model import GoalProbEvaluator
import yfinance as yf

from utils import get_eom_dates

# ALL WEATHER PORTFOLIO (MONTHLY REBALANCED)
df = yf.download(['SPY', 'TLT', 'IEF', 'GLD', 'DBC'])['Adj Close'].dropna()
rtns = np.log(df/df.shift(1)).fillna(0)


rebal_dates = get_eom_dates(rtns, start='2008-1-1', end='2024-6-30')
weights = pd.DataFrame(np.array([[0.3, 0.4, 0.15, 0.075, 0.075]] * len(rebal_dates)), columns=rtns.columns,
                       index=rebal_dates)


# INITIALIZE CONFIG
config = {
    'initial_W': 100,
    'T': len(weights.index),
    'bankrupt': 5,  # bankruptcy level
    'h': 1 / 12,  # monthly / if you want to calculate quarterly, set it to 1 / 4
    'imax': 1000,
    'rho': 2.5,   # a scale parameter
    'lb': [0] * weights.shape[-1],
    'ub': [1] * weights.shape[-1]
}
est_config = {
    'er_method': 'bayes_stein',
    'cov_method': 'epo',
    'window': 252 * 30
}

# RUN
gpe = GoalProbEvaluator(rtns, weights, config, est_config)

cashflows = np.ones((config["T"],)) * 0
gpe.add_cashflow(cashflows)

gpe.fit(fit_dist=True, display_result=True)

gpe.evaluate(goals=[200, 300], init_wealth=None, start_idx=0, end_idx=-1)
gpe.evaluate(goals=[205], init_wealth=200, start_idx=-3, end_idx=-1)
