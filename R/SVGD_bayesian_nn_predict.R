#' Prediction Function
#'
#' @param X The covariate matrix to be predicted.
#' @param theta The estimated parameters in the vector format.
#' @param num_nodes The number of nodes in each hidden layer.
#' @param scaling_coef scaling coef.
#' @return A vector of predicted outcomes.
#' @export

SVGD_bayesian_nn_predict <- function(svgd_bnn, X) {
  theta = svgd_bnn$theta
  num_nodes = svgd_bnn$num_nodes
  scaling_coef = svgd_bnn$scaling_coef

  M <- dim(theta)[1]
  d <- ncol(X)
  n <- nrow(X)
  mean_X_train <- scaling_coef[[1]]
  sd_X_train <- scaling_coef[[2]]
  mean_y_train <- scaling_coef[[3]]
  sd_y_train <- scaling_coef[[4]]

  X <- t(apply(X, 1, function(x) {(x-mean_X_train)/sd_X_train}))
  output_dim <- num_nodes[length(num_nodes)]

  pred_y <- array(0, dim = c(M, output_dim, n))
  for (i in 1:M) {
    para_list <- unpack_parameters(theta[i, ], d, num_nodes)
    loggamma <- para_list$loggamma
    pred_y[i, , ] <- forward_probagation(t(X), para_list, 'relu')$ZL * sd_y_train + mean_y_train
  }
  pred <- apply(pred_y, c(2, 3), mean)

  return(pred)
}
