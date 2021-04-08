# The Stein Variational Gradient Descent Method
`svgd` is the abbreviation for Stein Variational Gradient Descent, and it is a General Purpose Bayesian Inference Algorithm. In this package, we use svgd to implement a Byesian Neural Network, and provides the main function `SVGD_bayesian_nn` and the prediction function `SVGD_bayesian_nn_predict` to the users. 

## installation

Install our `svgd` package from github:
```
devtools::install_github("BangyaoZhao/svgd")
```

Load package by:
```
library(svgd)
```

## usage

One simple example can be found here:

```
library(MASS)
library(ggplot2)
library(dplyr)
df = Boston
X = as.matrix(df[, 1:12])
y = as.matrix(df[, 13:14])


SVGD = SVGD_bayesian_nn(
  X_train = X,
  y_train = y,
  X_test = X,
  y_test = y,
  M = 20,
  batch_size = 100,
  max_iter = 500,
  num_nodes = c(50, 2),
  master_stepsize = 1e-3,
  method = 'adagrad'
)

y_hat = SVGD_bayesian_nn_predict(X, SVGD$theta, c(50, 2), SVGD$scaling_coef)
rownames(y_hat) = colnames(y)
rbind(data.frame(t(y_hat), type = 'y_hat'), data.frame(y, type = 'y')) %>%
  ggplot(aes(lstat, medv, color = type)) +
  geom_point() +
  theme_bw()
```
