import numpy as np
import pandas as pd

from portfolio.estimators.base import BaseEstimator


class ExpectedReturns(BaseEstimator):
    """
    자산 기대수익률 추정 클래스
    """

    def sample_mean(self, rtn=None):
        """
        과거 데이터에 의한 기대수익률 추정 (historical)
        """

        if rtn is None:
            mu = self.mu
        else:
            assert isinstance(rtn, np.ndarray)
            mu = rtn.mean(axis=0)
        return mu

    def exponential_mean(self, rtn=None, halflife=63) -> np.ndarray:
        """
        Exponential expected return 추정
        """

        if rtn is None:
            rtn = self.rtns
        assert isinstance(rtn, np.ndarray)
        mu = pd.DataFrame(rtn).ewm(halflife=halflife).mean().iloc[-1, :].values
        return mu

    def implied_equilibrium_mean(self, risk_aversion=5, weights=None, cov=None) -> np.ndarray:
        """
        Black-litterman 프레임워크에 나오는 implied 기대수익률 추정법
            - reverse-engineering

        :param risk_aversion: set to 5 by default (ref. Campbell and Viceira (2002))
        :param weights
            if None -> equal weights
        """
        if weights is None:
            weights = np.ones(self.N) / self.N
        if cov is None:
            cov = self.cov
        mu = risk_aversion * cov @ weights
        return mu

    def calc_target_mu(self, mu, cov, vol_weighted=False):
        """
        James-Stein의 Mu 추정에 사용되는 함수
        """
        if vol_weighted:
            inv_cov = self.inverse_matrix(cov)
            target_mu = (np.sum(inv_cov, axis=1) @ mu) / np.sum(inv_cov)
        else:
            target_mu = np.mean(mu)
        return target_mu

    def james_stein(self, rtn=None, vol_weighted=False, cov=None):
        """
        Mu 벡터의 james-stein 추정량
        https://palomar.home.ece.ust.hk/MAFS6010R_lectures/slides_shrinkage_n_BL.pdf
        """

        if rtn is None:
            rtn = self.rtns
            mu = self.mu
        else:
            assert isinstance(rtn, np.ndarray)
            mu = rtn.mean(axis=0)

        if cov is None:
            rtn_ = rtn - mu
            cov = np.cov(rtn_, rowvar=False)
        target_mu = self.calc_target_mu(mu, cov, vol_weighted)

        eigvals = np.linalg.eigvals(cov)
        T = len(rtn)

        rho = (1 / T) * (np.sum(eigvals) - 2 * np.max(eigvals)) * (1 / np.sum((mu - target_mu) ** 2))
        mu = rho * target_mu + (1 - rho) * mu
        return mu

    def bayes_stein(self, rtn=None, vol_weighted=False, cov=None):
        """
        "Bayes-Stein Estimation for Portfolio Analysis (1986)"
        https://www.globalriskguard.com/resources/assetman/bayes_0010.pdf

        Equation 17.
        """
        if rtn is not None:
            assert isinstance(rtn, np.ndarray)
            rtn_ = rtn
            mu = rtn.mean(axis=0)
        else:
            rtn_ = self.rtns
            mu = self.mu

        T = rtn_.shape[0]

        if cov is None:
            cov = self.cov
        target_mu = self.calc_target_mu(mu, cov, vol_weighted)
        inv_cov = self.inverse_matrix(cov)
        rho = (self.N + 2) / ((self.N + 2) + (mu - target_mu).T @ inv_cov @ (mu - target_mu) * T)
        mu = rho * target_mu + (1 - rho) * mu
        return mu

    def fit(self, rtn=None, method='sample', param=None):
        method = method.lower()

        if param is None:
            param = {}

        if method == 'sample':
            mu_hat = self.sample_mean(rtn)
        elif method in ['exp', 'exponential']:
            mu_hat = self.exponential_mean(rtn, **param)
        elif method == 'implied':
            mu_hat = self.implied_equilibrium_mean(**param)
        elif method == 'stein':
            mu_hat = self.james_stein(rtn, **param)
        elif method in ['bayes-stein', 'bayes_stein']:
            mu_hat = self.bayes_stein(rtn, **param)
        else:
            NotImplementedError(f"{method} is not supported")

        mu_hat = pd.Series(mu_hat, index=self.cols)
        self.mu_hat = mu_hat
        return mu_hat

    def fit_by_regime(self, probs, regime_ts, acceptable_freq=30, method='sample', param=None):
        """
        Regime-Based Strategic Asset Allocation (2024)

        input
            - probs: 국면 확률 벡터 (2개 국면이라면 사이즈 2의 벡터)
            - regime_ts: 추정에 사용하는 수익률 panel data와 같은 크기인 국면 레이블 벡터
        """
        probs, regime_ts = self._check_regime_parameter(probs, regime_ts)
        self._check_regime_freq(regime_ts, acceptable_freq)

        method = method.lower()
        if param is None:
            param = {}

        regimes = np.unique(regime_ts)

        mus = np.vstack([self.fit(self.rtns[np.where(regime_ts == r)], method, param) for r in regimes])
        mus_hat = mus.T @ probs
        mu_hat = pd.Series(mus_hat, index=self.cols)
        self.mu_hat = mu_hat

        return mus, mu_hat
