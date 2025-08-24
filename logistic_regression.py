"""
Logistic regression classifier using gradient descent with backtracking line search.
"""

import numpy as np

class LogisticRegression():
    """
    Logistic Regression classifier using gradient descent with backtracking line search.

    Parameters
    ----------
    intercept : bool, default=True
        If True, an intercept term is added to the model (bias).
    max_iterations : int, default=100
        Maximum number of iterations for the gradient descent optimization.
    l0 : float, default=1.0
        Initial step size (learning rate) used in the line search procedure.

    Attributes
    ----------
    intercept : bool
        Whether or not an intercept is included in the model.
    max_iterations : int
        Maximum number of iterations allowed during fitting.
    l0 : float
        Initial learning rate for the optimization procedure.

    Methods
    -------
    sigmoid_transform(x)
        Compute the sigmoid transformation.
    fit(x, y, method='OLS')
        Fit the logistic regression model using gradient descent with backtracking line search.
    predict(x, beta)
        Predict probabilities for the input samples.
    """
    def __init__(self, intercept=True, max_iterations=100, l0=1.0):
        self.intercept = intercept
        self.max_iterations = max_iterations
        self.l0 = l0

    def sigmoid_transform(self, x):
        """
        Compute the sigmoid transformation.

        Parameters
        ----------
        x : array-like of shape (n_samples,) or (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray
            The element-wise sigmoid transformation of the input.
        """
        return 1 / (1 + np.exp(-x))

    def _prepare_data(self, x):
        """
        Prepare the input data by adding an intercept term if required.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        ndarray of shape (n_samples, n_features + 1) if intercept=True
            The modified feature matrix with an intercept term prepended.

        Raises
        ------
        ValueError
            If 'intercept' is not a boolean or if `x` is not a valid array.
        """
        try:
            if self.intercept:
                x = np.insert(x, 0, 1, axis=1)
            elif self.intercept not in (True, False):
                raise ValueError("Variable 'intercept' must be equal to True or False")
        except Exception as exc:
            raise ValueError("X must be at least a one-dimensional array.") from exc

        return x

    def _logistic_loss(self, x, y, beta):
        """
        Compute the logistic regression loss (negative log-likelihood).

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Input feature matrix.
        y : ndarray of shape (n_samples, 1)
            True binary labels (0 or 1).
        beta : ndarray of shape (n_features, 1)
            Current parameter vector.

        Returns
        -------
        float
            Logistic loss value.
        """
        return np.sum(-y * np.dot(x, beta) + np.log(1 + np.exp(np.dot(x, beta))))

    def _logistic_gradient(self, x, y, beta):
        """
        Compute the gradient of the logistic loss function.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Input feature matrix.
        y : ndarray of shape (n_samples, 1)
            True binary labels (0 or 1).
        beta : ndarray of shape (n_features, 1)
            Current parameter vector.

        Returns
        -------
        ndarray of shape (n_features, 1)
            Gradient vector.
        """
        return np.dot(x.T, self.sigmoid_transform(np.dot(x, beta)) - y)

    def fit(self, x, y):
        """
        Fit the logistic regression model using gradient descent with backtracking line search.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Input feature matrix.
        y : ndarray of shape (n_samples, 1)
            True binary labels (0 or 1).

        Returns
        -------
        beta : ndarray of shape (n_features, 1)
            Estimated model parameters.
        cost_function : float
            Final cost function value. 
        residuals : ndarray of shape (n_samples, 1)
            Residuals (observed - predicted).
        """
        l = self.l0
        x = self._prepare_data(x)

        beta = np.zeros((x.shape[1], 1))
        beta_next = beta - (1/l) * self._logistic_gradient(x, y, beta)
        for _ in range(self.max_iterations):
            grad = self._logistic_gradient(x, y, beta)
            beta_next = beta - (1/l) * grad

            while (self._logistic_loss(x, y, beta_next)
                   > self._logistic_loss(x, y, beta) - (1 / (2*l)) * np.sum(grad**2)):
                l *= 2
                beta_next = beta - (1 / l) * grad

                if l > 1e10:
                    break

            beta = beta_next
        cost_function = self._logistic_loss(x, y, beta)
        residuals = y - self.sigmoid_transform(np.dot(x, beta))

        return beta, cost_function, residuals

    def predict(self, x, beta):
        """
        Predict probabilities for the input samples.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Input feature matrix.
        beta : ndarray of shape (n_features, 1)
            Model parameter vector.

        Returns
        -------
        ndarray of shape (n_samples, 1)
            Predicted probabilities for each input sample belonging to class 1.
        """
        x = self._prepare_data(x)
        y_hat = self.sigmoid_transform(np.dot(x, beta))
        return y_hat
