import numpy as np
import pandas as pd


class BaseEstimator:

    def __init__(self, rtns: pd.DataFrame, window=None, mu=None, cov=None):
        """
        Base Estimator

            rtns: pd.DataFrame
                return panel data for estimators

            window: int, optional
                default is None to use all

            mu: N x 1 vector, Expected Returns of assets' returns
            cov: N x N matrix, Covariance Matrix of assets' returns
                - 경우에 따라 cov추정을 위해 mu가 필요하고 mu추정을 위해 cov 추정이 필요함
          """
        T, N = rtns.shape
        window = window if window is not None else T
        rtns_ = rtns.iloc[-window:]
        self.rtns = rtns_.values
        if mu is None:
            self.mu = np.nanmean(self.rtns, axis=0)
        else:
            self.mu = mu
        if cov is None:
            self.cov = np.cov(self.rtns, rowvar=False, ddof=1)
        else:
            self.cov = cov
        self.T = window
        self.N = N
        self.cols = rtns_.columns.to_list()
        self.index = rtns_.index.to_list()
        self.cov_hat = None
        self.er_hat = None

    @staticmethod
    def inverse_matrix(mat):
        return np.linalg.pinv(mat)

    def _check_regime_parameter(self, probs, regime_ts):
        """
        regime 기반 estimation할 때, 데이터 체크
        """
        assert len(regime_ts) == len(self.rtns)
        assert isinstance(regime_ts, pd.Series) or isinstance(regime_ts, np.ndarray)
        if isinstance(regime_ts, pd.Series):
            regime_ts = regime_ts.values

        assert isinstance(probs, pd.Series) or isinstance(probs, np.ndarray)
        if isinstance(probs, pd.Series):
            probs = probs.values
        return probs, regime_ts

    @staticmethod
    def _check_regime_freq(regime_ts, acceptable_freq):
        """
        국면 별 count 체크
            - 이를 테면, 고변동 국면으로 레이블링 된 데이터가 acceptable_freq 보다 작다면 ValueError 발생
        """
        regime_freqs = pd.Series(regime_ts).value_counts()
        min_freq = regime_freqs.min()
        min_regime = regime_freqs.argmin()
        if min_freq < acceptable_freq:
            raise ValueError(f"{min_regime} 국면으로 labeling할 수 있는 시계열의 길이가 {min_freq}으로 {acceptable_freq} 보다 작습니다.")
