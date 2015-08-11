import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline


class DenseTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


# TODO: Figure out how to blend s.t. I get better performance than before I blended
# TODO: Possible idea for blending: instead of only using results as inputs, also put in word vector results. May work better. Find an optimal weighing of these.
## Going along with ^, PCA ingredients + trained results, then GB those
class Blender(BaseEstimator, ClassifierMixin):
    def __init__(self, trained_clfs):
        self.clfs = trained_clfs
        # self.classifier = make_pipeline(OneHotEncoder(), DenseTransformer(),
        #                                 GradientBoostingClassifier())
        self.classifier = GradientBoostingClassifier()
        # self.classifier = make_pipeline(
        #     OneHotEncoder(), LogisticRegression(class_weight='auto'))

    def fit(self, data, target):
        # self.enc = LabelEncoder().fit(target)
        probs = self.transform_input(data)
        # self.classifier.fit(predictions, target)
        self.classifier.fit(probs, target)

    def predict(self, data):
        predictions = self.transform_input(data)
        return self.classifier.predict(predictions)

    def transform_input(self, data):
        probabilities = [clf.predict_proba(data) for clf in self.clfs]

        probabilities = np.array(probabilities)
        # features, samples = probabilities.shape
        n_clfs, samples, features = probabilities.shape
        probabilities = np.reshape(probabilities, (samples, n_clfs * features))
        probabilities[np.isnan(probabilities)] = 0
        return probabilities
    # def transform_input(self, data):
    #     predictions = [clf.predict(data) for clf in self.clfs]
    #     predictions = np.array([list(self.enc.transform(pred))
    #                             for pred in predictions])
    #     features, samples = predictions.shape
    #     return np.reshape(predictions, (samples, features))
