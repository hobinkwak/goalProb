from collections import defaultdict
from collections.abc import Iterable
from itertools import product

import numpy as np
import pandas as pd
from scipy import optimize

from portfolio.estimators import ExpectedReturns, Covariance
from utils import DistFit


class GoalProbEvaluator:

    def __init__(self, rtns, actions, config, estimator_config):

        assert isinstance(rtns, pd.DataFrame), "rtns should be dataframe type"
        assert isinstance(actions, pd.DataFrame), "actions should be dataframe type"

        self.rtns = rtns
        self.actions = actions
        self.preprocess_data()

        self.config = config
        self.est_config = estimator_config
        self.C = np.zeros((config["T"],))
        self.wealth_states = None

        self.expected_prob = None
        self.wealth_probs = None
        self.expected_wealth = None
        self.tran_probs = []
        self.goal_probs = {}

        self.goals = defaultdict(lambda: [[0], np.array([[0, 0]])])
        self.distribution = self.compute_gaussian_pdf
        self.dist_param = (0, 1)

    def preprocess_data(self):
        self.rtns.index = pd.to_datetime(self.rtns.index)
        self.actions.index = pd.to_datetime(self.actions.index)

    def add_cashflow(self, cashflows):
        self.C = cashflows

    @staticmethod
    def compute_gaussian_pdf(x, mu=0, sigma=1):
        """
        compute p.d.f. value of (x, mu, sigma) of gaussian distribution
        """
        return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
            -((x - mu) ** 2) / 2 * sigma ** 2
        )

    @staticmethod
    def compute_gbm(w, mu, vol, t, z):
        """
        compute Geometric Brownian Motion
        """
        return w * np.exp((mu - 0.5 * vol ** 2) * t + vol * np.sqrt(t) * z)

    @staticmethod
    def compute_z_from_gbm(w, w_prev, mu, sigma, h):
        return (1 / (sigma * np.sqrt(h))) * (
                np.log(w / w_prev) - (mu - 0.5 * sigma ** 2) * h
        )

    @staticmethod
    def _init_tran_prob_t(w_prev, w):
        tran_prob_t = np.zeros((w_prev.shape[0], w.shape[0]))
        return tran_prob_t

    @staticmethod
    def _get_positive_likelihood_idx(tran_prob):
        # tran_prob: ( len(w_old) x len(w)=imax )
        pos_like_idx = np.where(tran_prob.sum(axis=1) > 0)
        return pos_like_idx

    @staticmethod
    def _normalize_tran_prob(tran_prob, pos_like_idx):
        tran_prob[pos_like_idx] /= tran_prob[pos_like_idx].sum(
            axis=1, keepdims=True
        )
        return tran_prob

    @staticmethod
    def compute_minmax_port(er, cov, lb, ub, minimize):

        def f(w, er, cov, minimize):
            sign = 1 if minimize else -1
            if sign > 0:
                obj = w @ cov @ w.T
            else:
                obj = np.dot(w, er)
            return obj * sign

        cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
        w = optimize.minimize(
            fun=f,
            x0=np.ones(er.shape) / er.shape[0],
            bounds=list(zip(lb, ub)),
            args=(er, cov, minimize),
            constraints=cons,
        ).x
        return w

    @staticmethod
    def port_er(w, er):
        return w @ er

    @staticmethod
    def port_vol(w, cov):
        return np.sqrt(w @ cov @ w.T)

    @staticmethod
    def _display_evaluate_result(name, values):
        print(
            f"{name}:",
            " -> ".join(map(str, map(lambda x: round(x, 4), values))),
            end="\n\n",
        )

    @staticmethod
    def preprocess_goals(goals):
        if not isinstance(goals, Iterable):
            goals = np.array([goals])
        else:
            goals = np.array(goals)
        return goals

    def fit_distribution(self, r, dist_type):
        distfit = DistFit(r, dist_type)
        distfit.fit()
        distfit.summary()
        result = distfit.get_best('BIC')
        return result

    def _compute_tran_pdf(self, rv, distribution, dist_param):
        tran_prob = distribution(rv, *dist_param).T
        return tran_prob

    def _compute_tran_prob(self, *args):
        """
        args: idx, w, w_prev, mu, sigma
        """
        pos_idx, w, w_prev, mu, sigma, distribution, dist_param = args
        tran_prob = self._init_tran_prob_t(w_prev, w)  # (imax, imax)
        w = w.reshape(-1, 1)
        w_prev = w_prev.reshape(1, -1)
        z = self.compute_z_from_gbm(w, w_prev, mu, sigma, self.config["h"])  # (len(w)=imax, len(w_prev))
        tran_prob[pos_idx[0], :] = self._compute_tran_pdf(z, distribution, dist_param)  # (len(w_prev), len(w)=imax)
        pos_like_idx = self._get_positive_likelihood_idx(tran_prob)
        tran_prob = self._normalize_tran_prob(tran_prob, pos_like_idx)
        return tran_prob

    def compute_minmax_estimator(self, er, cov, lb=None, ub=None):
        """
        (mu, sigma) pair
        """

        if lb is None:
            lb = self.config['lb']
        else:
            pass
        if ub is None:
            ub = self.config['ub']
        else:
            pass

        w_min = self.compute_minmax_port(er, cov, lb, ub, minimize=True)
        w_max = self.compute_minmax_port(er, cov, lb, ub, minimize=False)

        min_vol = self.port_vol(w_min, cov)
        max_vol = self.port_vol(w_max, cov)

        min_er = np.dot(w_min, er)
        max_er = np.dot(w_max, er)

        return min_er, max_er, min_vol, max_vol

    def _generate_wealth_mins(self, initial_w, mu_min, vol):
        wealth_mins = [initial_w]
        for tau in np.arange(1, self.config["T"] + 1):
            v1 = self.compute_gbm(initial_w, mu_min, vol, self.config["h"] * tau, -self.config["rho"])
            v2 = self.compute_gbm(self.C[np.arange(tau)] - self.goals[tau][1][:, 0].max(),
                                  mu_min, vol, self.config["h"] * (tau - np.arange(1, tau + 1)),
                                  -self.config["rho"]).sum()
            wealth_mins.append(v1 + v2)
        wealth_mins = np.array(wealth_mins)
        wealth_mins[wealth_mins < self.config["bankrupt"]] = self.config["bankrupt"]
        return wealth_mins

    def _generate_wealth_maxs(self, initial_w, mu_max, sigma_max):
        wealth_maxs = [initial_w]
        for tau in np.arange(1, self.config["T"] + 1):
            v1 = self.compute_gbm(initial_w, mu_max, sigma_max, self.config["h"] * tau, self.config["rho"])
            v2 = self.compute_gbm(self.C[np.arange(tau)], mu_max, sigma_max,
                                  self.config["h"] * (tau - np.arange(1, tau + 1)), self.config["rho"]).sum()
            wealth_maxs.append(v1 + v2)
        wealth_maxs = np.array(wealth_maxs)
        return wealth_maxs

    def _generate_wealth_state(self, wealth_min, wealth_max):
        """
        :param wealth_min: minimum wealth at time t
        :param wealth_max: maximum wealth at time t
        """
        log_W_min = np.log(wealth_min)
        wealth_state = []
        for i in np.arange(self.config["imax"]):
            wealth_state.append(
                log_W_min + (i / (self.config["imax"] - 1)) * (np.log(wealth_max) - np.log(wealth_min))
            )
        wealth_state = np.array(wealth_state)

        # just adjustment
        wealth_state[-1] = np.log(wealth_max)
        # one of values in wealth_state must be the same as initial_W (ref. Das(2019))
        adjust_idx = np.argmin(abs(wealth_state - np.log(self.config["initial_W"])))
        adjust_value = np.log(self.config["initial_W"]) - wealth_state[adjust_idx]
        if adjust_value >= 0:
            wealth_state += adjust_value
        wealth_state = np.exp(wealth_state)
        return wealth_state

    def generate_wealth_states(self, min_er, max_er, max_vol):
        initial_w = self.config['initial_W']
        wealth_states = [np.array([initial_w])]
        wealth_mins = self._generate_wealth_mins(initial_w, min_er, max_vol)
        wealth_maxs = self._generate_wealth_maxs(initial_w, max_er, max_vol)

        for i in np.arange(self.config["T"]):
            wealth_state_new = self._generate_wealth_state(wealth_mins[i + 1], wealth_maxs[i + 1])
            wealth_states.append(wealth_state_new)

        return wealth_states

    def estimate_expected_return(self, rtns, period=252):
        model = ExpectedReturns(rtns)
        ers = model.fit(method=self.est_config['er_method']).values * period
        return ers

    def estimate_covariance(self, rtns, period=252):
        model = Covariance(rtns)
        cov = model.fit(method=self.est_config['cov_method']).values * period
        return cov

    def compute_paths(self, fit_dist=False, display_result=True):

        assert self.wealth_states is not None

        w_prob_t = np.ones(self.wealth_states[0].shape)
        wealth_probs = [w_prob_t]
        expected_wealth = [w_prob_t @ self.wealth_states[0]]
        expected_ers, expected_vols = [], []

        # Forward 방식의 evaluate
        for t in range(self.config["T"]):
            c = self.C[t]
            wealth_state_prev = self.wealth_states[t] + c
            wealth_state = self.wealth_states[t + 1]
            pos_index = np.where(wealth_state_prev > 0)
            wealth_state_prev = wealth_state_prev[pos_index]

            action = self.actions.iloc[t].copy()
            assets_invested = action[action != 0].index
            action = action[assets_invested]
            train_rtns = self.rtns.loc[:self.actions.index[t]].iloc[-self.est_config['window'] - 1:-1][
                assets_invested].dropna()

            if fit_dist:
                r = train_rtns @ action * 252
                distribution, dist_param, _ = self.fit_distribution(r, dist_type=['norm', 'skewnorm','t'])

            ers = self.estimate_expected_return(train_rtns, period=252)
            cov = self.estimate_covariance(train_rtns, period=252)

            port_er = self.port_er(action, ers)
            port_vol = self.port_vol(action, cov)

            # (len(wealth_state_prev), len(wealth_state))
            tran_prob = self._compute_tran_prob(pos_index, wealth_state, wealth_state_prev, port_er, port_vol,
                                                distribution, dist_param)
            self.tran_probs.append(tran_prob)

            expected_ers.append(w_prob_t @ (np.zeros_like(w_prob_t) + port_er))
            expected_vols.append(w_prob_t @ (np.zeros_like(w_prob_t) + port_vol))

            w_prob_t = w_prob_t @ tran_prob  # shape: imax / probability distribution of wealth at time t
            expected_wealth.append(w_prob_t @ self.wealth_states[t + 1])
            wealth_probs.append(w_prob_t)

        if display_result:
            print("*" * 20)
            print("PATHS GENERATED")
            print("*" * 20)
            self._display_evaluate_result("Expected Wealth", expected_wealth)

        datetime_index = self.actions.index.to_list() + ['END']
        self.wealth_probs = wealth_probs
        self.expected_wealth = pd.Series(expected_wealth, index=datetime_index)

    def compute_goal_probs(self, goal, end_idx):
        goal_prob = np.zeros_like(self.wealth_states[end_idx])
        goal_prob = [np.where(self.wealth_states[end_idx] >= goal, 1, goal_prob)]

        for t in range(1, end_idx + 1)[::-1]:
            gp = goal_prob[end_idx - t]
            tran_prob = self.tran_probs[t - 1]
            expected_gp = (tran_prob @ gp)
            expected_gp = np.where(np.isclose(expected_gp, 1, rtol=0, atol=1e-10), 1, expected_gp)
            goal_prob.append(expected_gp)
        goal_prob = goal_prob[::-1]
        self.goal_probs[(goal, end_idx)] = goal_prob

    def fit(self, fit_dist=True, display_result=True):

        if self.wealth_states is None:
            # Modeling Return & Risk
            rtns_ = self.rtns.dropna().copy()
            total_ers = self.estimate_expected_return(rtns_, period=252)
            total_cov = self.estimate_covariance(rtns_, period=252)

            min_er, max_er, min_vol, max_vol = self.compute_minmax_estimator(total_ers, total_cov)
            wealth_states = self.generate_wealth_states(min_er=min_er, max_er=max_er, max_vol=max_vol)
            self.wealth_states = wealth_states
            print("*" * 20)
            print("WEALTH STATES GENERATED")
            print("*" * 20)

        if self.wealth_probs is None:

            self.compute_paths(fit_dist=fit_dist, display_result=display_result)

    def evaluate(self, goals, init_wealth=None, start_idx=0, end_idx=-1, decimal=4):

        assert self.wealth_probs is not None

        goals = self.preprocess_goals(goals)

        if end_idx < 0:
            end_idx = (self.config['T'] + 1) + end_idx

        if init_wealth is None:
            init_wealth = self.config['initial_W']

        if start_idx == 0:
            if init_wealth != self.config['initial_W']:
                print(
                    f"init_wealth is set to {self.config['initial_W']}. If start_idx is set to 0, init_wealth should be 'initial_w' in config.")

            expected_prob = [
                self.wealth_probs[end_idx][self.wealth_states[end_idx] >= goal].sum()
                for goal in goals
            ]

        else:
            expected_prob = []
            for pair in product(goals, [end_idx]):
                if pair not in self.goal_probs:
                    self.compute_goal_probs(*pair)
                value = self.goal_probs[pair][start_idx][
                    abs(self.wealth_states[start_idx] - init_wealth).argmin()]
                expected_prob.append(value)

        expected_prob = np.array(expected_prob).round(decimal)
        dts = self.expected_wealth.index
        res = pd.DataFrame([
            [dts[start_idx], dts[end_idx], init_wealth, goals[i], expected_prob[i]] for i in range(len(goals))

        ], columns=['start date', 'end date', 'wealth(start)', 'wealth(end)', 'Probability'])

        return res


if __name__ == '__main__':
    pass
