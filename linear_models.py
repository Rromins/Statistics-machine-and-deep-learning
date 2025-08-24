"""
Linear Regression with gradient descent and OLS
"""

import math
import numpy as np

class LinearRegression():
    """
    Ordinary Least Squares (OLS) regression and Gradient Descent optimization
    for multiple explanatory variables.

    This class fits a linear model of the form:
        Y = beta_0 + beta_1 X_1 + beta_2 X_2 + ... + beta_n X_n + epsilon

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for the gradient descent algorithm.
    iterations : int, default=1000
        Number of iterations to run gradient descent.
    intercept : bool, default=True
        Whether to add an intercept column.

    Methods
    -------
    fit(x, y, method='Gradient Descent')
        Estimate regression coefficients using either OLS or Gradient Descent.
    predit(x, beta)
        Predict target values using the linear regression model.
    """
    def __init__(self, intercept=True, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.intercept = intercept

    def _prepare_data(self, x):
        """
        Prepare the design matrix for regression.

        Adds an intercept column of ones to `x` if `intercept=True`.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Input data matrix.

        Returns
        -------
        x : ndarray of shape (n_samples, n_features) or (n_samples, n_features+1)
            Prepared design matrix.

        Raises
        ------
        ValueError
            If `intercept` is not boolean or if `x` is not at least 1D.
        """
        try:
            if self.intercept:
                x = np.insert(x, 0, 1, axis=1)
            elif self.intercept not in (True, False):
                raise ValueError("Variable 'intercept' must be equal to True or False")
        except Exception as exc:
            raise ValueError("X must be at least a one-dimensional array.") from exc

        return x

    def _cost_function(self, x, beta, y):
        """
        Compute the Mean Squared Error (MSE) cost function.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Design matrix.
        beta : ndarray of shape (n_features, 1)
            Regression coefficients.
        y : ndarray of shape (n_samples, 1)
            Target values.

        Returns
        -------
        float
            Value of the cost function (MSE).
        """
        n = x.shape[0]
        return (1/n) * np.sum((y - np.dot(x, beta))**2)

    def _compute_gradient(self, x, beta, y):
        """
        Compute the gradient of the cost function with respect to beta.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Design matrix.
        beta : ndarray of shape (n_features, 1)
            Regression coefficients.
        y : ndarray of shape (n_samples, 1)
            Target values.

        Returns
        -------
        ndarray of shape (n_features, 1)
            Gradient vector.
        """
        n = x.shape[0]
        return -(2/n) * np.dot(x.T, (y - np.dot(x, beta)))

    def _ols(self, x, y):
        """
        Estimate regression coefficients using the Normal Equations.

        Solves:
            beta = (X^T*X)^(-1) * X^T*y

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Design matrix.
        y : ndarray of shape (n_samples, 1)
            Target values.

        Returns
        -------
        beta : ndarray of shape (n_features, 1)
            Estimated coefficients.
        cost_function : float
            Value of the cost function at the solution.
        residuals : ndarray of shape (n_samples, 1)
            Differences between observed and predicted values.
        """
        # calculate matrices
        xt = x.T
        xtx_inv = np.linalg.inv(np.dot(xt, x))
        xty = np.dot(xt, y)

        # compute beta, residuals and the cost function value
        beta = np.dot(xtx_inv, xty)
        residuals = y - np.dot(x, beta)
        cost_function = self._cost_function(x, beta, y)

        return beta, cost_function, residuals

    def _gradient_descent(self, x, y):
        """
        Estimate regression coefficients using Gradient Descent.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Design matrix.
        y : ndarray of shape (n_samples, 1)
            Target values.

        Returns
        -------
        beta : ndarray of shape (n_features, 1)
            Estimated coefficients.
        cost_function : list of float
            History of the cost function values across iterations.
        residuals : ndarray of shape (n_samples, 1)
            Differences between observed and predicted values.
        """
        # choose the starting point randomly
        limit = 1 / math.sqrt(x.shape[0])
        beta = np.random.uniform(-limit, limit, (x.shape[1], 1))

        # initiate the cost function history
        cost_function = []

        # run the gradient descent process
        for _ in range(self.iterations):
            gradient = self._compute_gradient(x, beta, y)
            beta = beta - self.learning_rate * gradient
            cost_function.append(self._cost_function(x, beta, y))
        residuals = y - np.dot(x, beta)

        return beta, cost_function, residuals

    def fit(self, x, y, method='Gradient Descent'):
        """
        Fit a linear regression model to the data.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Input data matrix.
        y : array-like of shape (n_samples,)
            Target values.
        method : {'OLS', 'Gradient Descent'}, default='Gradient Descent'
            Estimation method.

        Returns
        -------
        beta : ndarray of shape (n_features,) or (n_features+1,)
            Estimated regression coefficients.
        cost_function : float or list of float
            Cost function value(s). A float for OLS, a list for Gradient Descent.
        residuals : ndarray of shape (n_samples,)
            Differences between observed and predicted values.

        Raises
        ------
        ValueError
            If `method` is not 'OLS' or 'Gradient Descent'.
        """
        x = self._prepare_data(x=x)
        y = np.array(y).reshape(-1, 1)

        if method == 'OLS':
            beta, cost_function, residuals = self._ols(x, y)

        elif method == 'Gradient Descent':
            beta, cost_function, residuals = self._gradient_descent(x, y)

        else:
            raise ValueError("'method' need to be equal to 'OLS' or 'Gradient Descent'")

        return beta.reshape(-1), cost_function, residuals.reshape(-1)

    def predict(self, x, beta):
        """
        Predict target values using the linear regression model.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Input data matrix for which predictions are to be made.
        beta : ndarray of shape (n_features,) or (n_features+1,)
            Regression coefficients previously estimated by `fit`.
        
        Returns
        -------
        y_hat : ndarray of shape (n_samples,)
            Predicted values corresponding to the input data `x`.

        Raises
        ------
        ValueError
            If the design matrix `x` cannot be prepared (e.g., the model
            has not been fitted yet, or input dimensions are inconsistent).
        """
        x = self._prepare_data(x=x)
        y_hat = np.dot(x, beta)
        return y_hat
