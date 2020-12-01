#' Prediction Function
#'
#' @param X The covariate matrix to be predicted.
#' @param theta The estimated parameters in the vector format.
#' @param num_nodes The number of nodes in each hidden layer.
#' @param scaling_coef scaling coef.
#' @examples
#' library(MASS)
#' df = Boston
#' X = as.matrix(Boston[, 1:13])
#' y = Boston$medv
#' SVGD = SVGD_bayesian_nn(
#'   X_train = X,
#'   y_train = y,
#'   X_test = X,
#'   y_test = y,
#'   M = 20,
#'   batch_size = 100,
#'   max_iter = 100,
#'   num_nodes = c(50),
#'   master_stepsize = 1e-3,
#'   method = 'adagrad'
#' )
#' @return A vector.
#' @export

SVGD_bayesian_nn_predict <- function(X, theta, num_nodes, scaling_coef) {
  M <- dim(theta)[1]
  d <- ncol(X)
  n <- nrow(X)
  mean_X_train <- scaling_coef[[1]]
  sd_X_train <- scaling_coef[[2]]
  mean_y_train <- scaling_coef[[3]]
  sd_y_train <- scaling_coef[[4]]

  X <- t(apply(X, 1, function(x) {(x-mean_X_train)/sd_X_train}))

  pred_y <- matrix(rep(0, times = M * n), nrow = M)
  for (i in 1:M) {
    para_list <- unpack_parameters(theta[i, ], d, num_nodes)
    loggamma <- para_list$loggamma
    pred_y[i, ] <- forward_probagation(t(X), para_list, 'relu')$ZL * sd_y_train + mean_y_train
  }
  pred <- colMeans(pred_y)

  return(pred)
}
