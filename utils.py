import numpy as np
import pandas as pd
from scipy import stats


def get_eom_dates(df, start='2008', end=None):
    df_ = df.copy()
    df_.loc[:, 'month'] = df_.index.month
    df_.loc[:, 'year'] = df_.index.year
    dts = df_.drop_duplicates(subset=['year', 'month'], keep='last').index
    if end is None:
        end = df_.index[-1]
    return dts[(dts >= start) & (dts <= end)]


class DistFit:

    def __init__(self, data, dist_type):
        self.data = data
        raw1d = np.array(data).flatten()
        self.data1d = raw1d[~np.isnan(raw1d)]
        self.dist_type = dist_type
        self.params = {}
        self.pdfs = {}
        self.aic = {}
        self.bic = {}

    @staticmethod
    def _preprocess_data(data):
        raw1d = np.array(data).flatten()
        return raw1d[~np.isnan(raw1d)]

    def _compute_data_pdf(self):
        y, x = np.histogram(self.data1d, bins=100, density=True)
        x = [(j + x[i + 1]) / 2.0 for i, j in enumerate(x[:-1])]
        self.y = y
        self.x = x

    def _compute_err(self, pdf):
        return np.sum((pdf - self.y) ** 2)

    @staticmethod
    def _compute_aic(log_likelihood, k):
        return 2 * k - 2 * log_likelihood

    def _compute_bic(self, err, k):
        n = len(self.data1d)
        return n * np.log(err / n) + k * np.log(n)

    @staticmethod
    def _match_str_to_object(dist):
        if dist.lower() == 'norm':
            Dist = stats.norm
        elif dist.lower() == 't':
            Dist = stats.t
        elif dist.lower() == 'skewnorm':
            Dist = stats.skewnorm
        else:
            Dist = stats.laplace
        return Dist

    def _fit(self, dist: str = 'norm'):
        assert dist.lower() in ['norm', 't', 'skewnorm', 'laplace']

        Dist = self._match_str_to_object(dist)

        parameter = Dist.fit(self.data1d)
        pdf = Dist.pdf(self.x, *parameter)
        ll = np.sum(Dist.logpdf(self.x, *parameter))

        self.params[dist] = parameter
        self.pdfs[dist] = pdf

        err = self._compute_err(pdf)
        aic = self._compute_aic(ll, len(parameter))
        bic = self._compute_bic(err, len(parameter))

        self.aic[dist] = aic
        self.bic[dist] = bic

    def fit(self):
        self._compute_data_pdf()
        for dist in self.dist_type:
            self._fit(dist)

    def summary(self):
        result = pd.DataFrame([self.aic.values(), self.bic.values()], index=['AIC', 'BIC'],
                              columns=self.dist_type).T
        self.result = result
        return result

    def get_best(self, by):
        result = self.result.sort_values(by=[by])
        best = result.index[0]
        score = result.loc[best, by]
        Dist = self._match_str_to_object(best)
        pdf = Dist.pdf
        return pdf, self.params[best], score
