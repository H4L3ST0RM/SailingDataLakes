+++
authors = ["John C Hale"]
title = "Regularized Linear Regression"
date = "2024-03-12"
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

## Overfitting
Overfitting usually occurs when a model is overly complex for a given problem 
or given dataset, and thus able to memorize the training set. This leads to 
excellent performance in training, but less so when testing.

[IMG EXAMPLE]




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
\vec{\hat{\beta}}=\min_{\vec{\hat{\beta}}} L(D, \vec{\beta}) =\min_{\vec{\hat{\beta}}} \sum_{i=1}^{n}{(\hat{\beta} .\vec{x_i} - y_i)^2}+\lambda\hat{\beta_i}^2
\vec{\hat{\beta}}=\min_{\vec{\hat{\beta}}} L(D, \vec{\beta}) =\min_{\vec{\hat{\beta}}} \sum_{i=1}^{n}{(\hat{\beta} .\vec{x_i} - y_i)^2}+\lambda\hat{\beta_i}^2
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

## Example

## Metrics

## Conclusion
