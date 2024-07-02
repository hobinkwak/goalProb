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
    'lb': [0.25, 0.35, 0.10, 0.025, 0.025],  # A little buffer
    'ub': [0.35, 0.45, 0.20, 0.125, 0.125]
}
est_config = {
    'er_method': 'bayes_stein',
    'cov_method': 'rmt',
    'window': 252 * 30
}

# RUN
gpe = GoalProbEvaluator(rtns, weights, config, est_config)

cashflows = np.ones((config["T"],)) * 0
gpe.add_cashflow(cashflows)

gpe.fit(fit_dist=True, display_result=True)

start_idx = rebal_dates.get_loc(rebal_dates[rebal_dates > '2020-1'][0])
end_idx = rebal_dates.get_loc(rebal_dates[rebal_dates > '2021-12'][0])
gpe.evaluate(goals=[220, 250], init_wealth=200, start_idx=start_idx, end_idx=end_idx, decimal=4)
