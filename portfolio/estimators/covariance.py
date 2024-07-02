import numpy as np
import pandas as pd
import warnings
from sklearn.covariance import ledoit_wolf
from scipy import optimize

from portfolio.estimators.base import BaseEstimator


class Covariance(BaseEstimator):
    """
    자산 수익률 공분산 추정 클래스
    """

    @staticmethod
    def _check_psd(cov):
        """
        공분산 행렬이 positive-semi definite 인지 cholesky 분해를 통해 확인
            - psd가 만족되지 않으면 eigen value가 0 아래 값이 나오는 등 공분산 행렬이 제대로 추정되지 않음
                - corner solution 원인 중 하나
        """
        if (np.array_equal(cov, cov.T)) or (np.isclose(np.abs(cov - cov.T).sum(), 0)):
            try:
                np.linalg.cholesky(cov)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False

    def _convert_to_psd(self, cov):
        """
        (준)정부호 행렬로 변환
        """
        q, V = np.linalg.eigh(cov)
        q = np.where(q >= 0, q, 0)
        recon = V @ np.diag(q) @ V.T
        if not self._check_psd(recon):
            warnings.warn("covariance matrix가 양의 (semi)정부호 행렬이 아님")
        return recon

    def sample_cov(self, rtn=None):
        """
        표본 공분산 행렬
        """
        if rtn is None:
            cov = self.cov
        else:
            assert isinstance(rtn, np.ndarray)
            cov = np.cov(rtn, rowvar=False, ddof=1)
        return cov

    def _exp_cov(self, x1, x2, halflife=63):
        """
        exponential 공분산 "값" 추정 함수
        """
        return pd.Series((x1 - x1.mean()) * (x2 - x2.mean())).ewm(halflife=halflife).mean().iloc[-1]

    def exponential_cov(self, rtn=None, halflife=63) -> np.ndarray:
        """
        exponential 공분산 "행렬" 추정 함수
             - 공분산 행렬의 대칭성을 활용한 loop 로직
        """
        if rtn is None:
            rtns = self.rtns.T
        else:
            assert isinstance(rtn, np.ndarray)
            rtns = rtn.T

        cov = np.array([
            [
                self._exp_cov(rtns[i], rtns[j], halflife=halflife) if j < i else
                self._exp_cov(rtns[i], rtns[j], halflife=halflife) / 2 if j == i else
                0
                for j in range(self.N)
            ]
            for i in range(self.N)
        ])
        cov = cov + cov.T
        return cov

    def non_market_cov(self, rtn=None) -> np.ndarray:
        """
        공분산행렬에서 시장 수익률에 해당하는 1st principal component를 제거
        """
        if rtn is None:
            rtns = self.rtns
            cov = self.cov
        else:
            assert isinstance(rtn, np.ndarray)
            rtns = rtn
            cov = np.cov(rtns, ddof=1, rowvar=False)

        corr = np.corrcoef(rtns, rowvar=False)
        eigval, eigvec = np.linalg.eigh(corr)
        max_eigval_idx = np.argmax(eigval)
        eigval[max_eigval_idx] = 0

        recon = self.reconstruct_corr(eigval, eigvec)
        recon = self.corr_to_cov(recon, np.diag(cov) ** (1 / 2))
        return recon

    def reconstruct_corr(self, eigval, eigvec):
        """
        고유값/벡터를 통해 상관계수행렬을 재복원
        """
        eigval = np.diag(eigval)
        corr_recon = eigvec @ eigval @ eigvec.T
        # set diagonal to 1
        D = np.diag(np.diag(corr_recon) ** (-1 / 2))
        recon = D @ corr_recon @ D
        return recon

    @staticmethod
    def cov_to_corr(cov):
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)  # 외적
        corr[corr < -1], corr[corr > 1] = -1, 1
        return corr

    @staticmethod
    def corr_to_cov(corr, std):
        cov = corr * np.outer(std, std)
        return cov

    @staticmethod
    def get_eigen_bound(var, q):
        lb = var * (1 - np.sqrt(q)) ** 2
        ub = var * (1 + np.sqrt(q)) ** 2
        return lb, ub

    def get_MP_pdf(self, eigvals, var, q):
        """
        Marchenko-Pastur 분포 Probability density 계산
        """
        lb, ub = self.get_eigen_bound(var, q)
        mask = (eigvals > lb) & (eigvals < ub)
        const = 1 / (2 * np.pi * q * var)
        result = np.zeros(eigvals.shape)
        result[mask] = const * np.sqrt((ub - eigvals[mask]) * (eigvals[mask] - lb)) / eigvals[mask]
        return result

    def fit_MP_pdf(self, q, eigenvalues):
        """
        Marchenko-Pastur 분포 모수 추정
        """

        def objective(var):
            return - np.sum(np.log(1e-9 + self.get_MP_pdf(eigenvalues, var, q)))

        result = optimize.minimize(objective, x0=np.array([1]),
                                   bounds=([1e-9, 2],))
        return result.x[0]

    def rmt_cov(self, rtn=None) -> np.ndarray:
        """
        Random matrix Theory의 Marchenko-Pastur 분포를 활용하여
        수익률 공분산 행렬 상 무작위행렬의 고유값에 해당하는 주성분을 스무딩하여 디노이징
        """
        if rtn is None:
            rtns = self.rtns
            mu = self.mu
        else:
            assert isinstance(rtn, np.ndarray)
            rtns = rtn
            mu = rtns.mean(axis=0)

        T, N = rtns.shape

        rtn = rtns - mu
        cov = np.cov(rtn, rowvar=False, ddof=1)
        q = N / T
        corr = np.corrcoef(rtn, rowvar=False)
        eigval, eigvec = np.linalg.eigh(corr)

        var = self.fit_MP_pdf(q, eigval)
        lb, ub = self.get_eigen_bound(var, q)

        # eigval[(eigval <= ub) & (eigval >= max(eigval))] = np.mean(e igval[eigval <= ub])
        eigval[eigval <= ub] = np.mean(eigval[eigval <= ub])

        recon = self.reconstruct_corr(eigval, eigvec)
        recon = self.corr_to_cov(recon, np.diag(cov) ** (1 / 2))

        return recon

    def epo_cov(self, rtn=None, shrinkage=0.75) -> np.ndarray:
        """
        Enhanced Portfolio Optimization (AQR, 2020)
            - 상관계수 행렬은 Identity 행렬로 shrinkage하는 것만으로도 OOS 추정에 저해가 되는
            Problematic Principal component가 디노이징 됨
        :return:
        """
        if rtn is None:
            rtns = self.rtns
        else:
            assert isinstance(rtn, np.ndarray)
            rtns = rtn
        N = rtns.shape[1]
        cov = np.cov(rtns, rowvar=False)
        corr = np.corrcoef(rtns, rowvar=False)
        corr = (1 - shrinkage) * corr + shrinkage * np.identity(N)
        cov = self.corr_to_cov(corr, np.diag(cov) ** (1 / 2))
        return cov

    def ledoit_cov(self, rtn=None) -> np.ndarray:
        """
        Ledoit-wolf의 공분산 행렬 추정
        """
        if rtn is None:
            rtns = self.rtns
        else:
            assert isinstance(rtn, np.ndarray)
            rtns = rtn

        cov = ledoit_wolf(rtns)[0]
        return cov

    def denard_cov(self, rtn=None) -> np.ndarray:
        """
        G De Nard의 공분산 행렬 추정

        cov_hat = shrinkage_intensity * Target Estimator + (1-shrinakge_intensity) * Covariance

        Target Estimator = phi_hat * Identity Matrix + nu_hat * off-diagonal of Identity matrix

        phi_hat = average of diagonal elements of sample covariance matrix
        nu_hat = average of off diagonal elements of sample covariance matrix

        shrinkage_intensity = min(max(k_hat/T, 0), 1)

        k_hat = (pi_hat - rho_hat) / gamma_hat

        pi_hat = summation of (1/T) sum_{t=1}^T {(r_it - r_bar_i.)*(r_jt - r_bar_j.) - s_ij}^2
        rho_hat = 0 (practically zero, in almost all cases)
        gamma_hat = squared frobenius norm of difference between sample covariance matrix and target estimator)
        """
        if rtn is None:
            rtns = self.rtns
            mu = self.mu
        else:
            assert isinstance(rtn, np.ndarray)
            rtns = rtn
            mu = rtns.mean(axis=0)

        T, N = rtns.shape

        rtns = (rtns - mu).T
        cov = np.cov(rtns, rowvar=True, ddof=1)

        phi_hat = np.trace(cov) / N  # (1.15a)
        nu_hat = np.mean(cov - np.diag(np.diag(cov)))  # (1.15b)
        target_estimator = np.eye(N) * phi_hat + (1 - np.eye(N)) * nu_hat  # (1.9)

        pi_hat_mat = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                arr1 = rtns[i, :]
                arr2 = rtns[j, :]
                values = []
                for t in range(T):
                    values.append(((arr1[t] - arr1.mean()) * (arr2[t] - arr2.mean()) - cov[i, j]) ** 2)
                pi_hat_mat[i, j] = np.mean(np.array(values))
                pi_hat_mat[j, i] = pi_hat_mat[i, j]
        pi_hat = pi_hat_mat.sum()
        gamma_hat = np.square(cov - target_estimator).sum()
        rho = 0
        k_hat = (pi_hat - rho) / gamma_hat
        shrinkage_estimator = min(max(k_hat / T, 0), 1)
        cov_hat = (1 - shrinkage_estimator) * cov + shrinkage_estimator * target_estimator
        return cov_hat

    def gerber_cov(self, rtn=None, threshold: float = 0.5) -> np.ndarray:
        """
        Gerber Statistic (2022 ver.) <- Positive-semi-definite
            - noise에 insensitive한 co-movement measure
            - Portfolio Optimization 목적으로 만들어낸 통계량
        ref. https://github.com/yinsenm/gerber/blob/master/src/gerber.py
        """
        assert (threshold < 1) & (threshold > 0), "threshold는 0과 1사이"

        if rtn is None:
            rtns = self.rtns
        else:
            assert isinstance(rtn, np.ndarray)
            rtns = rtn

        T, N = rtns.shape
        vol = rtns.std(axis=0, ddof=1)
        U = np.copy(rtns)
        D = np.copy(rtns)

        for i in range(N):
            U[:, i] = U[:, i] >= vol[i] * threshold
            D[:, i] = D[:, i] <= -vol[i] * threshold

        n_conc = U.T @ U + D.T @ D
        n_disc = U.T @ D + D.T @ U
        H = n_conc - n_disc
        h = np.sqrt(np.diag(H))

        h = h.reshape((N, 1))
        vol = vol.reshape((N, 1))

        corr = H / (h @ h.T)
        cov_hat = corr * (vol @ vol.T)
        return cov_hat

    def fit(self, rtn=None, method='sample', check_psd=True, param=None):

        if param is None:
            param = {}

        method = method.lower()
        if method == 'sample':
            cov_hat = self.sample_cov(rtn)
        elif method in ['exp', 'exponential']:
            cov_hat = self.exponential_cov(rtn, **param)
        elif method == 'non_market':
            cov_hat = self.non_market_cov(rtn)
        elif method == 'ledoit':
            cov_hat = self.ledoit_cov(rtn)
        elif method == 'epo':
            cov_hat = self.epo_cov(rtn, **param)
        elif method == 'rmt':
            cov_hat = self.rmt_cov(rtn)
        elif method == 'denard':
            cov_hat = self.denard_cov(rtn)
        elif method == 'gerber':
            cov_hat = self.gerber_cov(rtn, **param)
        else:
            raise NotImplementedError(f"{method} is not supported")

        if check_psd:
            if not self._check_psd(cov_hat):
                cov_hat = self._convert_to_psd(cov_hat)

        cov_hat = pd.DataFrame(cov_hat, index=self.cols, columns=self.cols)
        self.cov_hat = cov_hat

        return cov_hat

    def fit_by_regime(self, ers, probs, regime_ts, acceptable_freq=30, method='sample',
                      check_psd=True, param=None):
        """
        Regime-Based Strategic Asset Allocation (2024)

        input
            - ers: 국면별 기대수익률 행렬 (n_regime x n_asset)
            - probs: 국면 확률 벡터 (2개 국면이라면 사이즈 2의 벡터)
            - regime_ts: 추정에 사용하는 수익률 panel data와 같은 크기인 국면 레이블 벡터
        """
        if param is None:
            param = {}

        probs, regime_ts = self._check_regime_parameter(probs, regime_ts)
        self._check_regime_freq(regime_ts, acceptable_freq)

        method = method.lower()

        regimes = np.unique(regime_ts)

        covs_by_regime = np.stack(
            [self.fit(self.rtns[np.where(regime_ts == r)], method, check_psd, param) for r in regimes])
        cov_intra = covs_by_regime.T @ probs
        er = ers.T @ probs
        cov_inter = np.array(
            [(ers[i] - er)[:, np.newaxis] @ (ers[i] - er)[:, np.newaxis].T for i in range(len(probs))]).T @ probs
        cov_hat = cov_intra + cov_inter
        if check_psd:
            if not self._check_psd(cov_hat):
                cov_hat = self._convert_to_psd(cov_hat)

        cov_hat = pd.DataFrame(cov_hat, index=self.cols, columns=self.cols)
        self.cov_hat = cov_hat
        return cov_hat
