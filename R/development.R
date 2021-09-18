#' @export development

development = function(svgd_bnn) {
  X_train = svgd_bnn$X_train_scaled
  y_train = svgd_bnn$y_train_scaled
  X_train_unscaled = svgd_bnn$X_train_unscaled
  y_train_unscaled = svgd_bnn$y_train_unscaled
  X_dev = svgd_bnn$X_dev
  y_dev = svgd_bnn$y_dev
  scaling_coef = svgd_bnn$scaling_coef
  eigenMat = svgd_bnn$eigenMat
  eigenMat_inv = svgd_bnn$eigenMat_inv
  scaled_eigenMat = svgd_bnn$scaled_eigenMat
  scaled_eigenMat_inv = svgd_bnn$scaled_eigenMat_inv
  num_vars = svgd_bnn$num_vars
  d = svgd_bnn$d
  M = svgd_bnn$M
  num_nodes = svgd_bnn$num_nodes
  theta = svgd_bnn$theta
  mean_X_train = scaling_coef$mean_X_train
  sd_X_train = scaling_coef$sd_X_train
  mean_y_train = scaling_coef$mean_y_train
  sd_y_train = scaling_coef$sd_y_train

  X_dev <-
    t(apply(X_dev, 1, function(x) {
      (x - mean_X_train) / sd_X_train
    }))

  y_dev <- t(y_dev)

  f_log_lk <- function(loggamma) {
    output_dim <- dim(y_train)[1]
    log_det_eigenMat <- sum(log(diag(eigenMat)))
    return(sum(log((exp(loggamma) ^ (output_dim / 2)) / ((2 * pi) ^ (output_dim /
                                                                       2)) * exp(-exp(loggamma) / 2
                                                                                 * colSums((pred_y_dev - y_dev) ^
                                                                                             2 * diag(eigenMat_inv)
                                                                                 ))
    ) - 0.5 * log_det_eigenMat))
  }

  for (i in 1:M) {
    para_list <- unpack_parameters(theta[i,], d, num_nodes)
    pred_y_dev <-
      forward_probagation(t(X_dev), para_list, 'relu')$ZL * sd_y_train + mean_y_train
    lik1 <- f_log_lk(para_list$loggamma)
    loggamma2 <-
      mean(log(diag(eigenMat)) - log(rowMeans((pred_y_dev - y_dev)^2)))
    lik2 <- f_log_lk(loggamma2)

    if (lik2 > lik1) {
      para_list$loggamma <- loggamma2
      theta[i,] <- pack_parameters(para_list)
    }
  }
  svgd_bnn$theta = theta
  return(svgd_bnn)
}
