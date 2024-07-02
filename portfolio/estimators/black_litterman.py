import numpy as np
import pandas as pd
from portfolio.estimators.base import BaseEstimator


class BlackLitterman(BaseEstimator):
    """
    블랙리터만 프레임워크 하에서의 기대수익률/공분산 추정 클래스
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.views = []
        self.picks = []

    def estimate_posterior(self, mu, cov_tau, P, total_uncertainty, view_difference):
        """
        posterior estimates of mu and covarinace matrix
        input
            - mu: prior estimator of mu (e.g. implied return under market equilibrium)
            - cov_tau: covariance matrix muliplied by tau(=scalar)
            - total_uncertainty: P @ cov_tau @ P.T + Omega
            - view_difference: difference between investors' views and market implied return, (Q - P @ mu)
        """

        common = cov_tau @ P.T @ self.inverse_matrix(total_uncertainty)
        mu_adjust = common @ view_difference
        cov_adjust = common @ P @ cov_tau

        mu_hat = mu + mu_adjust
        cov_hat = cov_tau - cov_adjust

        return mu_hat, cov_hat

    @staticmethod
    def implied_Omega(view_confidence, cov_tau, P):
        """
        Idzoerk's method

        input
            view_confidence: 뷰에 대한 신뢰도 벡터
            cov_tau: covariance matrix multiplied by tau
            P: picking matrix (view 벡터 사이즈 x n_asset)
        output
            Omega: uncertainty matrix of the views
        """
        confs_ = np.array(view_confidence)
        confs_[confs_ == 0] = 1e-16
        alphas = 1 / confs_ - 1
        Omega = np.diag(
            np.diag(
                alphas[:, np.newaxis] * P @ cov_tau @ P.T
            )
        )
        return Omega

    def insert_view(self, x, val, y=None):
        """
        view 1개 추가 함수 (여러개를 넣고 싶으면 여러번 호출해야 함)

        기본 포맷
            if y is None:
                x의 수익률이 val일 것이다.
            else:
                x의 수익률은 y의 수익률보다 val만큼 높을 것이다.

        input
            - x: 자산 이름 (self.rtns의 칼럼과 일치해야 함)
                - list로 줘도 됨. 단, val도 리스트로 들어와야 함
            - val: 수익률 view
            - y: 비교군

        example:
            er = ExpectedReturns(rtn)
            implied_mu = er.fit('implied', param={'risk_aversion': 5, 'weights': None})
            model = BlackLitterman(rtn, mu=implied_mu.values)

            - IT 수익률이 1%일 것이다.
                - model.insert_view(x='IT', val=0.01)
            - 금융 수익률이 유틸리티보다 2% 높을 것이다.
                - model.insert_view(x='금융', val=0.02, y='유틸리티')
            - 필수소비재 수익률이 통신, 에너지보다 1% 높을 것이다.
                - model.insert_view(x='필수소비재', val=0.01, y=['통신','에너지'])
        """

        if isinstance(x, str):
            x = [x]

        picks = np.zeros(self.N)
        vx = 1 / len(x)
        picks[[self.cols.index(i) for i in x]] = vx

        if y is not None:
            if isinstance(y, str):
                y = [y]
            vy = 1 / len(y)
            picks[[self.cols.index(i) for i in y]] = -vy

        self.views.append(val)
        self.picks.append(picks)

    def fit(self, view_confidence=None, tau=None):
        """
        input
            - view_confidence: view에 대한 신뢰도 벡터
            - tau: tau는 investor's view가 posterior estimator에 주는 영향을 컨트롤 하는 하이퍼파라미터
                - tau가 작을수록 prior(e.g. market equilibrium)로 shrinke 되는 정도가 줄어들어
                 investor's veiw의 영향력이 세짐
        """
        assert len(self.views) > 0, "view가 입력되어있지 않음"
        assert len(view_confidence) == len(self.views)

        self.views = np.array(self.views)
        self.picks = np.vstack(self.picks)

        if tau is None:
            tau = 1 / self.T

        mu = self.mu
        cov = self.cov

        Q = self.views
        P = self.picks

        cov_tau = tau * cov

        if view_confidence is None:
            Omega = np.diag(np.diag(P @ cov_tau @ P.T))
        else:
            Omega = self.implied_Omega(view_confidence, cov_tau, P)

        total_uncertainty = P @ cov_tau @ P.T + Omega
        view_difference = (Q - P @ mu)
        mu_hat, cov_hat = self.estimate_posterior(mu, cov_tau, P, total_uncertainty, view_difference)
        mu_hat = pd.Series(mu_hat, index=self.cols)
        cov_hat = pd.DataFrame(cov_hat, index=self.cols, columns=self.cols)

        self.mu_hat = mu_hat
        self.cov_hat = cov_hat

        return mu_hat, cov_hat


if __name__ == '__main__':
    pass
