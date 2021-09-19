#' @export evaluation

evaluation <- function(svgd_bnn, X_test, y_test) {
  eigenMat = svgd_bnn$eigenMat
  theta = svgd_bnn$theta
  num_nodes = svgd_bnn$num_nodes
  scaling_coef = svgd_bnn$scaling_coef

  M <- dim(theta)[1]
  mean_X_train <- scaling_coef[[1]]
  sd_X_train <- scaling_coef[[2]]
  mean_y_train <- scaling_coef[[3]]
  sd_y_train <- scaling_coef[[4]]
  y_test <- t(y_test)

  output_dim <- dim(y_test)[1]
  log_det_eigenMat <- sum(log(diag(eigenMat)))
  inv_eigenMat <- diag(1 / diag(eigenMat))

  X_test <-
    t(apply(X_test, 1, function(x) {
      (x - mean_X_train) / sd_X_train
    }))

  d <- ncol(X_test)

  pred_y_test <- array(0, dim = c(M, output_dim, dim(y_test)[2]))
  prob <-
    matrix(rep(0, times = M * length(y_test)),
           nrow = M,
           ncol = dim(X_test)[1])

  for (i in 1:M) {
    para_list <- unpack_parameters(theta[i,], d, num_nodes)
    loggamma <- para_list$loggamma
    pred_y_test[i, ,] <-
      forward_probagation(t(X_test), para_list, 'relu')$ZL * sd_y_train + mean_y_train
    prob[i,] <-
      (exp(loggamma)) ^ (output_dim / 2) / ((2 * pi) ^ (output_dim / 2)) *
      exp(-exp(loggamma) / 2 * colSums((pred_y_test[i, ,] - y_test) ^ 2 * diag(inv_eigenMat)))
  }
  pred <- apply(pred_y_test, c(2, 3), mean)


  svgd_rmse <- rmse(pred, y_test)
  svgd_Rsq <- Rsq(pred, y_test)
  svgd_ll <- mean(log(colMeans(prob))) - 0.5 * log_det_eigenMat
  return(list(
    svgd_rmse = svgd_rmse,
    svgd_Rsq = svgd_Rsq,
    svgd_ll = svgd_ll
  ))
}


rmse <- function(pred, y) {
  return(sqrt(mean((pred - y) ^ 2)))
}


Rsq <- function(pred, y) {
  SSR = sum((y - pred) ^ 2)
  SST = sum((y - mean(y)) ^ 2)
  return(1 - SSR / SST)
}
