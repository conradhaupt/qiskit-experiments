# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Discriminators that wrap SKLearn."""

from typing import Any, List, Dict, Union

from qiskit_experiments.data_processing.discriminator import BaseDiscriminator
from qiskit_experiments.data_processing.exceptions import DataProcessorError

try:
    from sklearn.base import ClassifierMixin
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.utils.validation import check_is_fitted

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class SkLDA(BaseDiscriminator):
    """A wrapper for the SKlearn linear discriminant analysis."""

    def __init__(self, lda: "LinearDiscriminantAnalysis"):
        """
        Args:
            lda: The sklearn linear discriminant analysis. This may be a trained or an
                untrained discriminator.

        Raises:
            DataProcessorError: if SKlearn could not be imported.
        """
        if not HAS_SKLEARN:
            raise DataProcessorError(
                f"SKlearn is needed to initialize an {self.__class__.__name__}."
            )

        self._lda = lda
        self.attributes = [
            "coef_",
            "intercept_",
            "covariance_",
            "explained_variance_ratio_",
            "means_",
            "priors_",
            "scalings_",
            "xbar_",
            "classes_",
            "n_features_in_",
            "feature_names_in_",
        ]

    @property
    def discriminator(self) -> Any:
        """Return then SKLearn object."""
        return self._lda

    def is_trained(self) -> bool:
        """Return True if the discriminator has been trained on data."""
        return not getattr(self._lda, "classes_", None) is None

    def predict(self, data: List):
        """Wrap the predict method of the LDA."""
        return self._lda.predict(data)

    def fit(self, data: List, labels: List):
        """Fit the LDA.

        Args:
            data: The independent data.
            labels: The labels corresponding to data.
        """
        self._lda.fit(data, labels)

    def config(self) -> Dict[str, Any]:
        """Return the configuration of the LDA."""
        attr_conf = {attr: getattr(self._lda, attr, None) for attr in self.attributes}
        return {"params": self._lda.get_params(), "attributes": attr_conf}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SkLDA":
        """Deserialize from an object."""

        if not HAS_SKLEARN:
            raise DataProcessorError(f"SKlearn is needed to initialize an {cls.__name__}.")

        lda = LinearDiscriminantAnalysis()
        lda.set_params(**config["params"])

        for name, value in config["attributes"].items():
            if value is not None:
                setattr(lda, name, value)

        return SkLDA(lda)


class SkCLF(BaseDiscriminator):
    """A wrapper for the SKlearn classfier Pipeline."""

    def __init__(self, cls: Union[ClassifierMixin, Pipeline]):
        """
        Args:
            cls: The classifier.

        Raises:
            DataProcessorError: if SKlearn could not be imported.
        """
        if not HAS_SKLEARN:
            raise DataProcessorError(
                f"SKlearn is needed to initialize an {self.__class__.__name__}."
            )

        if isinstance(cls, Pipeline):
            self._clf = cls
        else:
            self._clf = make_pipeline(StandardScaler(), cls)
        self.attributes = [
            "named_steps",
            "classes_",
            "n_features_in_",
            "features_names_in_",
        ]

    @property
    def discriminator(self) -> Any:
        """Return the SKLearn object."""
        return self._clf

    def is_trained(self) -> bool:
        """Return True if the discriminator has been trained on data."""
        return not getattr(self._clf, "classes_", None) is None

    def predict(self, data: List):
        """Wrap the predict method of the LDA."""
        return self._clf.predict(data)

    def fit(self, data: List, labels: List):
        """Fit the classifier.

        Args:
            data: The independent data.
            labels: The labels corresponding to data.
        """
        self._clf.fit(data, labels)
        self._sklearn_is_fitted__ = check_is_fitted(self._clf)

    def config(self) -> Dict[str, Any]:
        """Return the configuration of the classifier."""
        attr_conf = {attr: getattr(self._clf, attr, None) for attr in self.attributes}
        return {"params": self._clf.get_params(), "attributes": attr_conf}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SkLDA":
        """Deserialize from an object."""

        if not HAS_SKLEARN:
            raise DataProcessorError(f"SKlearn is needed to initialize an {cls.__name__}.")

        sgdc = Pipeline(config['params']['steps'])
        sgdc.set_params(**config["params"])

        skclf = SkCLF(sgdc)
        for name, value in config["attributes"].items():
            if value is not None:
                setattr(skclf, name, value)

        return skclf
