import numpy as np
import matplotlib.pyplot as plt

class UnivariateLinearRegression():
    """
    Ordinary Least Squares (OLS) regression for a single explanatory variable.

    Parameters --- 
        X : array-like of shape (n_samples,)
            Input variable.
        Y : array-like of shape (n_samples,)
            Target variable.

    Methods ---
        fit(intercept=True)
            Estimate regression coefficients.
        plot(beta)
            Display data points and fitted regression line.
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def _prepare_data(self, intercept):
        """
        Prepare the data matrix for regression.

        If `intercept` is True, adds a column of ones to the design matrix.
        Transforms X and Y into the correct 2D shapes for computations.

        Parameters ---
            intercept : bool
                Whether to add an intercept term to the regression.

        Raises ---
            ValueError
                If `intercept` is not boolean, or if `X` is not one-dimensional.
        """
        try:
            if intercept == True:
                self.df = np.vstack([np.ones(len(self.X)), self.X]).T
            elif intercept == False:
                self.df = np.array(self.X).reshape(-1, 1)
            else:
                raise ValueError("Variable 'intercept' must be equal to True or False.")
        except:
            raise ValueError("Variable X must be a one-dimensional array.")
        self.Y = np.array(self.Y).reshape(-1, 1)
    
    def fit(self, intercept=True):
        """
        Estimate regression coefficients using the normal equations.

        Solves the system:
            Beta = (X^T X)^(-1) X^T Y

        Parameters ---
            intercept : bool, default=True
                If True, add an intercept term to the regression.

        Returns ---
            beta : ndarray of shape (n_features,) or (n_features+1,)
                Estimated regression coefficients.
            residuals : ndarray of shape (n_samples,)
                Differences between observed and predicted values.
        """
        self._prepare_data(intercept=intercept)
        
        # calculate matrices
        xT = self.df.T
        xTx_inv = np.linalg.inv(np.dot(xT, self.df))
        xTy = np.dot(xT, self.Y)
        beta = np.dot(xTx_inv, xTy)
        Xbeta = np.dot(self.df, beta)
        residuals = self.Y - Xbeta
        return beta.reshape(-1), residuals.reshape(-1)

    def plot(self, beta):
        """
        Plot the data points and the fitted regression line.

        Parameters ---
            beta : ndarray of shape (n_features,)
                Estimated regression coefficients, typically obtained from `fit`.

        Raises ---
            ValueError
                If the model has not been fitted before plotting.
        """
        try:
            plt.scatter(self.X, self.Y, s=5)
            plt.plot(self.X, np.dot(self.df, beta), label='Univariate Linear Regression line', c='red')
            plt.legend()
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Univariate Linear Regression')
            plt.show()
        except:
            raise ValueError("Data need to be fit before plotting.")


class MultivariateLinearRegression():
    """
    Ordinary Least Squares (OLS) regression for multiple explanatory variables.

    This class fits a linear model of the form:
        Y = beta_0 + beta_1 X_1 + beta_2 X_2 + ... + beta_n X_n + epsilon

    Parameters ---
        X : array-like of shape (n_samples, n_features)
            Input data matrix.
        Y : array-like of shape (n_samples,)
            Target variable.

    Methods ---
        fit(intercept=True)
            Estimate regression coefficients using the normal equations.
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def _prepare_data(self, intercept):
        """
        Prepare the design matrix for regression.

        If `intercept` is True, adds a column of ones to X.
        Transforms Y into the correct 2D shape.

        Parameters ---
            intercept : bool
                Whether to add an intercept term to the regression.

        Raises ---
            ValueError
                If `X` does not have the correct shape (n_samples, n_features).
        """
        try:
            if intercept == True:
                self.df = np.hstack((self.X, np.ones(len(self.X))[:, None]))
            elif intercept == False:
                self.df = self.X
            else:
                raise ValueError("Variable 'intercept' must be equal to True or False")
        except:
            raise ValueError("Matrice X must be of the shape (n_samples, n_features)")
        self.Y = np.array(self.Y).reshape(-1, 1)
    
    def fit(self, intercept=True):
        """
        Estimate regression coefficients using the normal equations.

        Solves the system:
            Beta = (X^T X)^(-1) X^T Y

        Parameters ---
            intercept : bool, default=True
                If True, add an intercept term to the regression.

        Returns ---
            beta : ndarray of shape (n_features,) or (n_features+1,)
                Estimated regression coefficients.
            residuals : ndarray of shape (n_samples,)
                Differences between observed and predicted values.

        Raises ---
            ValueError
                If the input matrix X does not have the correct shape.
        """
        self._prepare_data(intercept=intercept)
        
        # calculate matrices
        try:
            xT = self.df.T
            xTx_inv = np.linalg.inv(np.dot(xT, self.df))
            xTy = np.dot(xT, self.Y)
            beta = np.dot(xTx_inv, xTy)
            Xbeta = np.dot(self.df, beta)
            residuals = self.Y - Xbeta
        except:
            raise ValueError("Matrice X must be of the shape (n_samples, n_features)")
        
        return beta.reshape(-1), residuals.reshape(-1)