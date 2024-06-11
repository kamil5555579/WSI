import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import sklearn.datasets

class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.parameters = {}
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.parameters[c] = {
                "mean": X_c.mean(axis=0), # mu_y for each feature
                "var": X_c.var(axis=0) + 1e-4, # sigma^2_y for each feature
                "prior": X_c.shape[0] / X.shape[0], # P(y)
            }

    def _pdf(self, X, c):
        mean = self.parameters[c]["mean"]
        var = self.parameters[c]["var"]
        numerator = np.exp(-((X - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator # Gaussian P(x|y) for each feature and observation

    def predict(self, X):
        posteriors = np.zeros((X.shape[0], len(self.classes)))
        for i, c in enumerate(self.classes):
            prior = np.log(self.parameters[c]["prior"]) # log(P(y))
            likelihood = np.log(self._pdf(X, c)).sum(axis=1) # log(P(x|y)) for each observation
            posteriors[:, i] = prior + likelihood
        return self.classes[np.argmax(posteriors, axis=1)]
    

if __name__ == "__main__":
    data = sklearn.datasets.load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    le = LabelEncoder()
    y = le.fit_transform(y)

    accuracies = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        nb = NaiveBayes()

        nb.fit(X_train.values, y_train)

        y_pred = nb.predict(X_test.values)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    print("Mean Accuracy:", np.mean(accuracies))
    print("Std Accuracy:", np.std(accuracies))