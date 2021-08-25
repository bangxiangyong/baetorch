from scipy.stats import gamma, beta, lognorm, uniform, expon, norm

from statsmodels.distributions import ECDF
import numpy as np

from uncertainty_ood_v2.util.get_predictions import flatten_nll


class ECDF_Wrapper(ECDF):
    def fit(self, x):
        return super(ECDF_Wrapper, self).__init__(x)

    def predict(self, x):
        return self(x)

    def cdf(self, x):
        return self(x)


class Outlier_CDF:
    """
    Class to get outlier probabilities by fitting distributions to outlier scores.

    Available distributions are gamma, beta, lognorm, norm, uniform, exponential and ECDF.
    Also normalisation scaling is available as option based on:
     https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2011/SDM11-outlier-preprint.pdf
    """

    dist_dict = {
        "gamma": gamma,
        "beta": beta,
        "lognorm": lognorm,
        "norm": norm,
        "uniform": uniform,
        "expon": expon,
        "ecdf": ECDF_Wrapper,
    }

    def __init__(self, dist_type="gamma", norm_scaling=None, norm_scaling_level="mean"):
        if norm_scaling is None:
            norm_scaling = False if dist_type == "uniform" else True
        self.norm_scaling = norm_scaling
        self.norm_scaling_level = norm_scaling_level
        self.norm_scaling_param = None

        self.dist_type = dist_type
        self.dist_class = self.dist_dict[dist_type]

    def fit(self, outlier_score):
        # fit a selected distribution to outlier scores
        if self.dist_type != "ecdf":
            self.dist_ = self.dist_class(*self.dist_class.fit(outlier_score))
        else:
            self.dist_ = self.dist_class(outlier_score)

        # if norm_scaling is required
        if self.norm_scaling:
            self.fit_norm_scaler(self.predict(outlier_score, norm_scaling=False))

        return self

    def fit_norm_scaler(self, outlier_score):
        self.norm_scaling_param = self.get_norm_scaling_level(
            outlier_score, min_level=self.norm_scaling_level
        )

    def predict(self, outlier_score, norm_scaling=None):
        if norm_scaling is None:
            norm_scaling = self.norm_scaling

        outlier_proba = self.dist_.cdf(outlier_score)

        # if norm_scaling is enabled
        if norm_scaling:
            if self.norm_scaling_param is None:
                raise ValueError(
                    "Please fit the CDF first. Norm scaler is found to be None."
                )

            outlier_proba = np.clip(
                (outlier_proba - self.norm_scaling_param)
                / (1 - self.norm_scaling_param),
                0,
                1,
            )

        return outlier_proba

    def get_norm_scaling_level(self, nll, min_level="quartile"):
        """
        Return level of NLL scores for cdf scaling.
        """
        if min_level == "quartile":
            return np.percentile(nll, 75)
        elif min_level == "median":
            return np.percentile(nll, 50)
        elif min_level == "mean":
            return np.mean(nll)
        else:
            raise NotImplemented


class BAE_Outlier_Proba:
    def __init__(
        self,
        dist_type="ecdf",
        norm_scaling=None,
        norm_scaling_level="mean",
        fit_per_bae_sample=True,
    ):
        self.dist_type = dist_type
        if norm_scaling is None:
            norm_scaling = False if dist_type == "uniform" else True
        self.norm_scaling = norm_scaling
        self.norm_scaling_level = norm_scaling_level
        self.fit_per_bae_sample = fit_per_bae_sample

    def fit(self, bae_nll_samples, dist_type=None, norm_scaling=None):
        """
        Fits a single or multiple distributions on BAE NLL scores of training data.
        """
        # House keeping on handling default dist_type and default norm_scaling
        # If None is supplied, resort to internal saved param
        # Otherwise, overrides the internal saved param.

        if dist_type is None:
            dist_type = self.dist_type
        else:
            self.dist_type = dist_type
        if norm_scaling is None:
            norm_scaling = self.norm_scaling
        else:
            self.norm_scaling = norm_scaling
        dist_ = []
        # fit a cdf on every BAE model's prediction
        # resulting in ensemble of cdfs
        if self.fit_per_bae_sample:
            for bae_pred_sample in flatten_nll(bae_nll_samples):
                dist_.append(
                    Outlier_CDF(
                        dist_type=dist_type,
                        norm_scaling=norm_scaling,
                        norm_scaling_level=self.norm_scaling_level,
                    ).fit(bae_pred_sample)
                )

        # fit a cdf on the mean of BAE models' prediction
        else:
            dist_ = Outlier_CDF(dist_type=dist_type, norm_scaling=norm_scaling).fit(
                flatten_nll(bae_nll_samples).mean(0)
            )
        self.dist_ = dist_
        return self

    def predict_proba_samples(self, bae_nll_samples, norm_scaling=None):
        """
        Predicts the outlier probability with option to turn on/off normalisation.
        """
        if self.fit_per_bae_sample:
            outlier_probas = np.array(
                [
                    dist_.predict(bae_nll_sample, norm_scaling=norm_scaling)
                    for dist_, bae_nll_sample in zip(
                        self.dist_, flatten_nll(bae_nll_samples)
                    )
                ]
            )
        else:
            outlier_probas = np.array(
                [
                    self.dist_.predict(bae_nll_sample, norm_scaling=norm_scaling)
                    for bae_nll_sample in flatten_nll(bae_nll_samples)
                ]
            )
        return outlier_probas

    def calc_ood_unc(self, prob):
        """
        Calculates uncertainty of being an outlier given the probability samples.
        """
        unc = prob * (1 - prob)
        epi = prob.var(0)
        alea = unc.mean(0)
        return {"epi": epi, "alea": alea, "total": epi + alea}

    def calc_ood_unc(self, prob):
        """
        Calculates uncertainty of being an outlier given the probability samples.
        """
        unc = prob * (1 - prob)
        epi = prob.var(0)
        alea = unc.mean(0)
        return {"epi": epi, "alea": alea, "total": epi + alea}

    def predict(self, bae_nll_samples, norm_scaling=None):
        """
        Given BAE NLL samples, return both the OOD proba mean and proba uncertainty (dict).
        """
        proba_samples = self.predict_proba_samples(
            bae_nll_samples, norm_scaling=norm_scaling
        )
        proba_mean = proba_samples.mean(0)
        proba_unc = self.calc_ood_unc(proba_samples)  # dict with keys of epi/alea/total
        return proba_mean, proba_unc
