"""
Polynomial regression using gradient descent or OLS
"""

import math
from itertools import combinations_with_replacement
import numpy as np

class PolynomialRegression():
    """
    Polynomial Regression model supporting both Ordinary Least Squares (OLS) 
    and Gradient Descent estimation methods, with optional L2 regularization (ridge).

    Parameters
    ----------
    degree : int, default=2
        Degree of the polynomial features.
    learning_rate : float, default=0.01
        Learning rate used in Gradient Descent.
    iterations : int, default=1000
        Maximum number of iterations for Gradient Descent.
    lam : float, default=1e-3
        Regularization strength (ridge penalty). If 0, no regularization is applied.

    Methods
    -------
    polynomial_transform(x, intercept=True)
        Expand the input data with polynomial features up to the specified degree.
    fit(x, y, method='OLS')
        Estimate regression coefficients using either OLS or Gradient Descent.
    predict(x, beta)
        Predict target values using the linear regression model.

    Notes
    -----
    - Input features are expanded to polynomial terms using combinations with replacement.
    - Intercept (bias term) can be added explicitly in the transformed design matrix.
    """
    def __init__(self, intercept=True, degree=2, learning_rate=0.01, iterations=1000, lam=1e-3):
        self.intercept = intercept
        self.degree = degree
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lam = lam

    def polynomial_transform(self, x):
        """
        Expand the input data with polynomial features up to the specified degree.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Original input feature matrix.
        intercept : bool
            If True, a column of ones is prepended to represent the intercept term.

        Returns
        -------
        x_poly : ndarray of shape (n_samples, n_output_features)
            Transformed feature matrix including polynomial terms (and intercept if requested).

        Raises
        ------
        ValueError
            If `intercept` is not a boolean.
        """
        n_samples, n_features = x.shape

        # generate all possible combinations for a given degree
        combinations = [combinations_with_replacement(range(n_features), i)
                        for i in range(1, self.degree+1)]
        all_combinations = [item for sublist in combinations for item in sublist]
        output_features = len(all_combinations)

        # compute the new data
        x_poly = np.empty((n_samples, output_features))
        for i, index in enumerate(all_combinations):
            x_poly[:, i] = np.prod(x[:, index], axis=1)

        if self.intercept:
            x_poly = np.insert(x_poly, 0, 1, axis=1)
        elif self.intercept not in (True, False):
            raise ValueError("Variable 'intercept' must be equal to True or False")

        return x_poly

    def _cost_function(self, x, beta, y):
        """
        Compute the Ridge-regularized Mean Squared Error (MSE).

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
            Regularized MSE value.
        """
        n = x.shape[0]
        mse = (1/n) * np.sum((y - np.dot(x, beta))**2)
        ridge = (self.lam/n) * np.sum(beta**2)
        return mse + ridge

    def _compute_gradient(self, x, beta, y):
        """
        Compute the gradient of the regularized cost function.

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
        grad : ndarray of shape (n_features, 1)
            Gradient vector with respect to `beta`.
        """
        n = x.shape[0]
        return -(2/n) * np.dot(x.T, (y - np.dot(x, beta))) + (2*self.lam/n) * beta

    def _ols(self, x, y):
        """
        Estimate coefficients using the Ordinary Least Squares (Normal Equation).

        Solves:
            beta = (X^T X)^(-1) X^T y

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Design matrix after polynomial expansion.
        y : ndarray of shape (n_samples, 1)
            Target values.

        Returns
        -------
        beta : ndarray of shape (n_features, 1)
            Estimated regression coefficients.
        cost_function : float
            Final value of the cost function.
        residuals : ndarray of shape (n_samples, 1)
            Residuals (observed - predicted).
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
        Estimate coefficients using Batch Gradient Descent.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Design matrix after polynomial expansion.
        y : ndarray of shape (n_samples, 1)
            Target values.

        Returns
        -------
        beta : ndarray of shape (n_features, 1)
            Estimated regression coefficients.
        cost_function : list of float
            Cost function values over iterations.
        residuals : ndarray of shape (n_samples, 1)
            Residuals (observed - predicted).
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

    def fit(self, x, y, method='OLS'):
        """
        Fit the polynomial regression model to data.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Input features.
        y : array-like of shape (n_samples,)
            Target values.
        intercept : bool, default=True
            Whether to include an intercept term.
        method : {'OLS', 'gradient descent'}, default='OLS'
            Estimation method to use.

        Returns
        -------
        beta : ndarray of shape (n_features,) or (n_features+1,)
            Estimated coefficients.
        cost_function : float or list of float
            Cost function value(s). Float for OLS, list for Gradient Descent.
        residuals : ndarray of shape (n_samples,)
            Residuals (observed - predicted).

        Raises
        ------
        ValueError
            If `method` is not recognized.
        """
        x_poly = self.polynomial_transform(x=x)
        y = np.array(y).reshape(-1, 1)

        if method == 'OLS':
            beta, cost_function, residuals = self._ols(x_poly, y)

        elif method == 'gradient descent':
            beta, cost_function, residuals = self._gradient_descent(x_poly, y)

        else:
            raise ValueError("'method' need to be equal to 'OLS' or 'gradient descent'")

        return beta.reshape(-1), cost_function, residuals.reshape(-1)

    def predict(self, x, beta):
        """
        Predict target values for new input data.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Input feature matrix.
        beta : ndarray of shape (n_features,) or (n_features+1,)
            Coefficients obtained from `fit`.
        intercept : bool, default=True
            Whether the intercept column should be added.

        Returns
        -------
        y_hat : ndarray of shape (n_samples,)
            Predicted target values.
        """
        x = self.polynomial_transform(x=x)
        y_hat = np.dot(x, beta)

        return y_hat
