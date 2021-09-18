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
##########################################
##             Boston Example           ##
##########################################
devtools::load_all()
library(MASS)
library(ggplot2)
library(dplyr)
# library(autodiffr)
# ad_setup()

df = Boston
X = as.matrix(df[, 1:12])
y = as.matrix(df[, 13:14])
SVGD = SVGD_bayesian_nn(X_train = X,
                        y_train = y,
                        M = 20,
                        a0 = 1,
                        b0 = 0.1
                        )
SVGD = optimizer(SVGD,
                 max_iter = 1000,
                 batch_size = 100,
                 tol = 1e-6,
                 check_freq = Inf)
evaluation(SVGD, SVGD$X_dev, SVGD$y_dev)
SVGD = optimizer(SVGD,
                 max_iter = 1000,
                 batch_size = 100,
                 tol = 1e-6,
                 check_freq = Inf)


SVGD = development(SVGD)
evaluation(SVGD, X, y)

# plot
y_hat=SVGD_bayesian_nn_predict(SVGD,X)
y_hat%>%
  t()%>%
  as.data.frame()%>%
  rename(lstat=V1,medv=V2)%>%
  mutate(type = 'predict')->
  data1
y%>%
  as.data.frame()%>%
  mutate(type = 'true')->
  data2

data = rbind(data1,data2)
data%>%
  ggplot(aes(lstat,medv,color = type))+
  geom_point()+
  theme_bw()
```
