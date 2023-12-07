import numpy as np
from cvxopt import matrix, solvers
import numpy as np
import time
import os

def rbf_kernel(X, gamma):
    # compute the squared norm of each row in X
    row_norms = np.sum(X**2, axis=1).reshape(-1, 1)

    # use broadcasting to find pairwise squared differences
    # (row_norms - 2 * X.dot(X.T) + row_norms.T) computes the squared Euclidean distance
    squared_differences = row_norms - 2 * X.dot(X.T) + row_norms.T

    K = np.exp(-gamma * squared_differences)
    return K

class SVM:
    def __init__(self, correct_label=5):
        self.w = None
        self.b = None
        self.correct_label = correct_label

    def fit(self, X, y, C, gamma, kernel, dataset, sv_threshold, save_dir):
        
        start_time = time.perf_counter()

        n_samples = X.shape[0]  # X is of shape (n_samples, n_features), features = pixels
        y = np.where(y == self.correct_label, 1.0, -1.0)  # label 5
        
        if kernel == 'rbf':
            K = rbf_kernel(X, gamma)
            P = matrix(np.outer(y, y) * K)
        
        elif kernel == 'linear':
            P = matrix(np.outer(y, y) * np.dot(X, X.T)) # P_ij = y_i * y_j * dot(X_i, X_j) 
            
        q = matrix(np.ones(n_samples) * -1)    # used for -sum(alphas) term of objective function
        G = matrix(np.vstack((np.diag(np.ones(n_samples) * -1), np.identity(n_samples)))) # 0 <= alpha_i <= C constraint
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * C))) # same constraint for G
        A = matrix(y, (1, n_samples))   # Convert y labels to a 1xm matrix
        b = matrix(0.0)

        sol = solvers.qp(P, q, G, h, A, b)
        
        
        # Extract the Lagrange multipliers
        alphas = np.array(sol['x']).flatten()
        
        # Identify the indices of the support vectors
        sv_indices = np.where(alphas > sv_threshold)[0]

        # Extract support vector coordinates
        support_vectors = X[sv_indices]
        
        print(f'# of support vectors: {len(support_vectors)}')

        # calculate weight vector w
        self.w = np.sum(alphas[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)


        # Calculate bias term b - any SV will do
        sv = support_vectors[0]
        self.b = y[sv_indices[0]] - np.dot(self.w, sv)
        
        end_time = time.perf_counter()

        # save weights
        if save_dir:
            path = os.path.join(save_dir, f'wb{self.correct_label}.npz')
            np.savez(path, weights=self.w, bias=self.b)

        return end_time - start_time
        
    def predict(self, X):
        linear_output = np.matmul(X, self.w) + self.b
        return linear_output

    
    def test(self, X_test, y_test):
        predictions = self.predict(X_test)
        # assuming y_test isn't already binary
        y_test_binary = np.where(y_test == self.correct_label, 1, -1) 
        correct_predictions = np.sum(predictions == y_test_binary)
        accuracy = correct_predictions.item() / y_test.shape[0]
        return accuracy

    def load_weights(self, weights, bias):
        self.w = weights
        self.b = bias


class OneVsRestSVM:
    def __init__(self, n_classes, save_dir=None):
        self.classifiers = [SVM(correct_label=i) for i in range(n_classes)]
        self.save_dir = save_dir

    def fit(self, X, y, C=1.0, gamma=1.0, kernel='linear', dataset='cifar', sv_threshold=1e-5):
        for i, svm in enumerate(self.classifiers):
            total_time = svm.fit(X, y, C=C, gamma=gamma, kernel=kernel, dataset=dataset, sv_threshold=sv_threshold, save_dir=self.save_dir)
            print(f"Classifier {i} took {total_time:.2f} seconds to fit. ({total_time/60:.2f} minutes)")

    def predict(self, X):
        decision_scores = np.stack([classifier.predict(X) for classifier in self.classifiers])
        predictions = np.argmax(decision_scores, axis=0)
        return predictions

    def test(self, X_test, y_test):
        predictions = self.predict(X_test)
        correct_predictions = np.sum(predictions == y_test)
        accuracy = correct_predictions.item() / y_test.shape[0]
        return accuracy
       
        
    def init_weights(self, dataset, kernel):
        for i, svm in enumerate(self.classifiers):
            root = f'saves/{dataset}/{kernel}/wb{i}.npz'
            data = np.load(root)
            svm.load_weights(data['weights'], data['bias'])
