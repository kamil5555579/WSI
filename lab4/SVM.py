import numpy as np
import cvxopt
from cvxopt import solvers

class SVM:
    def __init__(self, kernel='linear', C=1.0, gamma='auto'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Calculate kernel matrix
        if self.kernel == 'linear':
            K = np.dot(X, X.T)
        elif self.kernel == 'rbf':
            if self.gamma == 'auto':
                self.gamma = 1 / n_features
            K = self.rbf_kernel(X)
        elif self.kernel == 'poly':
            K = (1 + np.dot(X, X.T)) ** 2
        else:
            raise ValueError("Invalid kernel type.")

        # Quadratic Programming setup
        P = cvxopt.matrix(np.outer(y, y) * K)  # Hessian matrix
        q = cvxopt.matrix(-np.ones(n_samples))  # Linear term
        G = cvxopt.matrix(np.vstack([-np.eye(n_samples), np.eye(n_samples)]))  # Inequality constraints
        h = cvxopt.matrix(np.hstack([np.zeros(n_samples), self.C * np.ones(n_samples)]))  # Inequality bounds
        A = cvxopt.matrix(y.astype(float), (1, n_samples))  # Equality constraint
        b = cvxopt.matrix(0.0)  # Equality constraint

        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)

        # Get Lagrange multipliers
        self.alpha = np.array(solution['x']).flatten()

        # Compute support vectors and bias
        sv_indices = np.where(self.alpha > 1e-5)[0]
        self.support_vectors = X[sv_indices]
        self.alpha = self.alpha[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.w = np.dot(self.alpha * y[sv_indices], self.support_vectors)
        self.b = np.mean(y[sv_indices] - np.dot(K[sv_indices][:, sv_indices], self.alpha * y[sv_indices]))

    def predict(self, X):
        if self.kernel == 'linear':
            K = np.dot(X, self.support_vectors.T)
            f = np.dot(K, self.alpha * self.support_vector_labels) + self.b
            return np.sign(f)
        elif self.kernel == 'rbf':
            K = self.rbf_kernel(X, self.support_vectors)
            f = np.dot(K, self.alpha * self.support_vector_labels) + self.b
            return np.sign(f)
        elif self.kernel == 'poly':
            K = (1 + np.dot(X, self.support_vectors.T)) ** 2
            f = np.dot(K, self.alpha * self.support_vector_labels) + self.b
            return np.sign(f)
        else:
            raise ValueError("Invalid kernel type.")

    def rbf_kernel(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        return np.exp(-self.gamma * np.linalg.norm(X1[:, np.newaxis] - X2, axis=2) ** 2)
