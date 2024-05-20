+++
authors = ["John C Hale"]
title = "Regularized Linear Regression"
date = "2024-05-10"
description = "n overview of how regularized linear regression works"
math = true
draft = false
tags = [
"ml",
"data science",
"machine learning",
"regression",
]
categories = [
"Machine Learning Walkthrough",
]
series = ["ML Walkthrough"]
+++
## Purpose
The goal of this article is first to develop an understanding of overfitting 
and regularization. After which, we will discuss the intuition, math, and code 
of the two primary methods of regularizing linear regression; ridge regression, 
and LASSO regression.

Below were defining each of our classes. We are going to keep them barebones for this exercise, so we can highlight the important parts. Notice that the `predict()` function is the exact same for each class. That is because the underlying model is the same for all of these variations. What is different, is how we define the loss function.

## Overfitting
Overfitting usually occurs when a model is overly complex for a given problem 
or given dataset, and thus able to memorize the training set. This leads to 
excellent performance in training, but less so when testing. This is also known as having high variance.

## Regular Linear Regression Recap


```python
import pandas as pd
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(suppress=True, precision=6)
```



The `fit()` function is where we optimize our parameters, $\beta$. 

Our loss function for regular linear regression looks like this:
$$\vec{\hat{\beta}}=\min_{\vec{\hat{\beta}}} L(D, \vec{\beta}) =\min_{\vec{\hat{\beta}}} \sum_{i=1}^{n}{(\hat{\beta} .\vec{x_i} - y_i)^2}$$

For basic linear regression below, we have a closed form solution, and we can find our optimal $\beta$  to minimize our loss function as follows:
$$\hat{\beta} = (\vec{X}^{T} \vec{X})^{-1} \vec{X}^{T} \vec{y}$$



```python
class linear_regression:
    def __init__(self, X, y):
        self.betas = self.fit(X, y)

    def fit(self,X, y):
        X=np.append(X, np.ones(X.shape[1]).reshape(1,-1), 0).T
        betas = np.linalg.pinv(X.T @ X) @ X.T @ y
        return betas

    def predict(self, X):
        X=np.append(X, np.ones(X.shape[1]).reshape(1,-1), 0).T
        return X.T @ self.betas  
```

## Regularization
Regularization encompasses any method that acts to “punish” complexity within a
model in an effort to prevent overfitting. This can involve adding penalties for 
heavily weighted model, but there are other methods we will discuss in a future
article.

Two common tools in regularization of machine learning algorithms are the L1-Norm
and L2-Norm. The L1-Norm equates to Manhattan distance, and the L2-Norm is 
equivalent to Euclidean distance. 

## Ridge Regression
Ridge regularization (L2-Regularization) effectively works by taking the square
of the coefficients, summing them together, and then taking the square
root. The calculation is appended to the cost function discussed in the linear 
regression post. Doing this effectively punishes coefficients for having large 
magnitudes. It pushes the coefficents *towards* zero.

works as follows.


$$
\vec{\hat{\beta}}=\min_{\vec{\hat{\beta}}} L(D, \vec{\beta}) =\min_{\vec{\hat{\beta}}} \sum_{i=1}^{n}{(\hat{\beta} .x_i - y_i)^2}+\lambda\hat{\beta_i}^2
$$

$$
L(D,\vec{\beta})=||X\vec{\beta} - Y||^2 + \lambda \bf{I} ||\vec{\beta}||^2
$$

$$
=(X\vec{\beta}-y)^T(X\vec{\beta}-Y)+\lambda \bf{I} \vec{\beta}^T \vec{\beta}
$$

$$
=Y^TY-Y^TX\vec{\beta}-\vec{\beta}^TX^TY+\vec{\beta}^TX^TX\vec{\beta} + \lambda \bf{I} \vec{\beta}^T \vec{\beta}
$$

Get gradient w.r.t. $\vec{\beta}$

$$
\frac{\partial{L(D,\vec{\beta})}}{\partial{\vec{\beta}}} = \frac{\partial{(Y^TY-Y^TX\vec{\beta}-\vec{\beta}^TX^TY+\vec{\beta}^TX^TX\vec{\beta}+\lambda \bf{I} \vec{\beta}^T \vec{\beta}})}{\partial{\vec{\beta}}}
$$

$$
= -2Y^TX+2\vec{\beta}^TX^TX + 2 \lambda \bf{I} \vec{\beta}^T
$$

Set gradient to zero

$$
=-2Y^TX+2\vec{\beta}^TX^TX + \lambda \bf{I} \vec{\beta}^T=0
$$

$$
Y^TX=\vec{\beta}^TX^TX + \lambda \bf{I} \vec{\beta}^T
$$

$$
Y^TX=\vec{\beta}^T(X^TX + \lambda \bf{I})
$$

$$
X^TY=(X^TX + \lambda \bf{I})^T\vec{\beta}
$$

$$
\vec{\beta}=(X^TX + \lambda \bf{I} )^{-1}X^TY
$$

The beauty of ridge regression, is that the solution is still closed form and thus 
always solvable.

By pushing feature values towards zero, it also helps to prevent the model from
over relying on a small subset of features.

As discussed above, ridge regression also has a closed form solution, thus we calculate the $\beta$ parameters in the `fit()` function as follows:

$$\vec{\beta}=(X^TX + \lambda \bf{I} )^{-1}X^TY$$


```python
class ridge_regression:
    def __init__(self, X, y, λ):
        self.betas = self.fit(X, y, λ)

    def fit(self,X, y, λ):
        X=np.append(X, np.ones(X.shape[1]).reshape(1,-1), 0).T
        I = np.identity(X.shape[1])

        betas = np.linalg.pinv(X.T @ X + λ * I) @ X.T @ y
        return betas

    def predict(self, X):
        X=np.append(X, np.ones(X.shape[1]).reshape(1,-1), 0).T
        return X.T @ self.betas  
```

## LASSO Regression
LASSO regression uses L1-Normalization. So rather than taking the sum of squares of
the coefficients and adding that to the loss the function, we take the sum of absolute
values of the coefficients.

By doing this, the penalty for coefficient magnitude is linear, versus exponential
like it was for ridge regression. This has the effect of not just pushing coefficients
to below 1, but all the way to 0. If the coefficient has a value of 0.5, then it deals
a penalty of 0.5 with LASSO regression. Whereas with ridge regression, the smaller the coefficient,
then the penalty gets exponentially smaller. Ie. a coefficient of 0.5 would have a penalty
of $0.5^2 = 0.25$. 

With that being said, LASSO regression thus works as a method of automatic feature selection,
since it will push coefficients towards 0, as well as a regularizaiton method!

The downside is that LASSO regression does NOT have a closed form solution, and thus we
don't have guarantee that we ever find the optimal solution. Since it doesn't have a closed
form solution, an iterative optimization algorithm is typically ran on the loss function
to find the values of $\beta$.

The loss function being minimized is
$$\vec{\hat{\beta}}=
\min_{\vec{\hat{\beta}}} L(D, \vec{\beta}) =\min_{\vec{\hat{\beta}}}
\sum_{i=1}^{n}{(\hat{\beta} .\vec{x_i} - y_i)^2}+\lambda |\hat{\beta}_i|
$$

Something to note, if you set $\lambda$ to 0, our loss function is identical to regular linear regression's loss function.


```python
class lasso_regression:
    def __init__(self, X, y, λ):
        self.betas = self.fit(X, y, λ)

    def fit(self,X, y, λ):
        # Add Bias
        X=np.append(X, np.ones(X.shape[1]).reshape(1,-1), 0).T

        # Define loss function
        loss_function = lambda betas, X, y, λ: np.sum(((X @ betas) - y)**2) +  λ * np.abs(betas).sum()
        # Initialize parameters
        betas = np.random.normal(0,.001,X.shape[1])
        # Minimize loss function by adjusting parameters
        res = minimize(loss_function, x0=betas, args=(X, y, λ))
        #Select optimized parameters
        betas = res.x        
        # Return optimized parameters
        return betas

    def predict(self, X):
        X=np.append(X, np.ones(X.shape[1]).reshape(1,-1), 0).T
        return X.T @ self.betas 
```

## Elastic Net Regression

Finally we have elastic net regression. It is a combination of both LASSO and ridge regression. It has two hyper parameter terms now $\lambda_1$ and $\lambda_2$. $\lambda_1$ controls extent of L2-regularization, and $\lambda_2$ controls the extent of L1-regularization. If either of the two parameters are set to 0, then that effectively removes that regularization function. Below is the loss function.

$$\vec{\hat{\beta}}=
\min_{\vec{\hat{\beta}}} L(D, \vec{\beta}) =\min_{\vec{\hat{\beta}}}
\sum_{i=1}^{n}{(\hat{\beta} .\vec{x_i} - y_i)^2}+\lambda_1 \hat{\beta}_i^2 + \lambda_2 |\hat{\beta}_i|
$$

Elastic Net regularization is essentially a generalizaiton of ridge and LASSO regularization.

Elasticnet regression does not have a closed form solution either, and thus the `fit()` function looks very similar to lasso regression's. The difference being the loss function. So the `fit()` function is miinimizeing the loss shown above.

Note that if you set $\lambda 1$ to 0, you have lASSO regression, and if you set $\lambda 2$ to 0, you have ridge regression. If you set both $\lambda 1$ and $\lambda 2$ hyperparameters to 0, you get regular linear regression. 


```python
class elasticnet_regression:
    def __init__(self, X, y, λ1, λ2):
        self.betas = self.fit(X, y, λ1, λ2)

    def fit(self,X, y, λ1, λ2):
        # Add Bias
        X=np.append(X, np.ones(X.shape[1]).reshape(1,-1), 0).T
        # Define loss function
        loss_function = lambda betas, X, y, λ1, λ2: np.sum((y - (X @ betas))**2) + λ1 * np.abs(betas).sum() + λ2 * np.square(betas).sum()
        # Initialize parameters
        betas = np.random.normal(0,.001,X.shape[1])
        # Minimize loss function by adjusting parameters
        res = minimize(loss_function, x0=betas, args=(X, y,  λ1,  λ2))
        #Select optimized parameters
        betas = res.x        
        # Return optimized parameters
        return betas

    def predict(self, X):
        X=np.append(X, np.ones(X.shape[1]).reshape(1,-1), 0).T
        return X.T @ self.betas  
```

## Other Applications

Ridge (L2) and LASSO (L1) regularizaiton is used in quite a few other models as well. Neural networks, logistic regression, boosted trees, etc... all have variations that include either L1 or L2 regularization.


## Conclusion

In this post we discussed some regularized variants of regular linear regression. Specifically we covered ridge regression, which uses L2 regularization. Lasso regression, which uses L1 regularization, and finally elasticnet, which uses a combination of L1 and L2 regularization. We went over the intuition behind the different methods and implemented them in code. There was no example in this article, but I may add one in a future rendition. It is admittedly difficult to illustrate the need for regularization with a linear model on 2D data :). 
