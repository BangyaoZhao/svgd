##########################################
##             Main Function            ##
##########################################
#' Main Function
#'
#' @param X_train The training dataset variables, a matrix with rows representing observations and columns representing covariates.
#' @param y_train The training dataset outcomes, a vector with the length same as the number of rows of `X_train`.
#' @param X_test The testing data set variables, a matrix with the same number of columns as `X_train`.
#' @param y_test The testing dataset outcomes, a vector with the length same as the number of rows of `X_test`.
#' @param batch_size The batch size.
#' @param max_iter The maximum number of iterations.
#' @param M The number of particles.
#' @param num_nodes The number of nodes in each hidden layer (does not include the last layer, because the node in the last layer is always 1).
#' @param a0 a0, for the prior distribution of lambda and gamma.
#' @param b0 b0, for the prior distribution of lambda and gamma.
#' @param master_stepsize The master stepsize, which is needed to adjust convergence if using adagrad for optimization of the NN.
#' @param auto_corr The auto correlation, which is needed to adjust convergence if using adagrad for optimization of the NN.
#' @param method The optimization method to be used.
#' @param eigenMat the variance matrix of the outcome
#' @return A list containing:
#' \itemize{
#'   \item{theta: }{The estimated parameters in the vector format.}
#'   \item{scaling_coef: }{The scaling coefficient}
#'   \item{svgd_rmse: }{The RMSE on the training data}
#'   \item{svgd_11: }{log likelihood}
#' }
#' @importFrom stats dist median rgamma rnorm
#' @importFrom utils tail
#' @export

SVGD_bayesian_nn <-
  function(X_train,
           y_train,
           eigenMat = diag(dim(y_train)[2]),
           X_test = NULL,
           y_test = NULL,
           batch_size = 100,
           max_iter = 1000,
           M = 20,
           num_nodes = c(20, 1),
           a0 = 1,
           b0 = 0.1,
           master_stepsize = 1e-3,
           auto_corr = 0.9,
           method = 'adam') {
    n_layers <- length(num_nodes) + 1
    d <- ncol(X_train)
    n_data <- nrow(X_train)

    eigenMat_inv <- diag(1 / diag(eigenMat))

    para_cumsum <- parameter_cumsum(d, num_nodes)$para_cumsum
    num_vars <- tail(para_cumsum, 1)

    # Keep the last 10% (max 500) of training data points for model developing
    size_dev = min(round(0.1 * n_data), 500)
    X_dev <- X_train[-(1:(n_data - size_dev)),]
    y_dev <- y_train[-(1:(n_data - size_dev)),]
    X_train <- X_train[(1:(n_data - size_dev)),]
    y_train <- y_train[(1:(n_data - size_dev)),]

    # Normalize the data set
    X_train <- scale(X_train)
    y_train <- scale(y_train)
    mean_X_train <- attr(X_train, 'scaled:center')
    sd_X_train <- attr(X_train, 'scaled:scale')
    mean_y_train <- attr(y_train, 'scaled:center')
    sd_y_train <- attr(y_train, 'scaled:scale')
    scaling_coef <-
      list(mean_X_train, sd_X_train, mean_y_train, sd_y_train)

    y_train <- t(y_train)
    # Get the number of data points
    N0 <- nrow(X_train)
    # Initialize the parameters
    theta <- matrix(rep(0, num_vars * M), nrow = M)
    for (i in 1:M) {
      theta_i <- initialization(d, num_nodes, a0, b0)

      # A better initialization for gamma
      ridx <- sample(N0, min(N0, 1000), replace = F)
      y_hat <-
        forward_probagation(t(X_train[ridx,]), theta_i, 'relu')$ZL
      loggamma <-
        mean(log(diag(eigenMat)) - log(rowMeans((y_hat - y_train[, ridx]) ^ 2)))
      theta_i$loggamma <- loggamma

      theta[i,] <- pack_parameters(theta_i)
    }

    # Call the optimizer
    theta <-
      optimizer(
        X_train,
        y_train,
        eigenMat_inv,
        master_stepsize,
        auto_corr,
        max_iter,
        batch_size,
        M,
        num_vars,
        d,
        num_nodes,
        N0,
        theta,
        a0,
        b0,
        method
      )

    # Tuning for a better gamma
    X_dev <-
      t(apply(X_dev, 1, function(x) {
        (x - mean_X_train) / sd_X_train
      }))
    y_dev <- t(y_dev)

    f_log_lk <- function(loggamma) {
      output_dim <- dim(y_train)[1]
      det_eigenMat <- cumprod(diag(eigenMat))[output_dim]
      return(sum(log((exp(loggamma) ^ (output_dim / 2)) / ((2 * pi) ^ (output_dim /
                                                                         2) * sqrt(det_eigenMat)) * exp(-exp(loggamma) / 2 * colSums((pred_y_dev - y_dev) ^
                                                                                                                                       2 * diag(eigenMat_inv)
                                                                         ))
      )))
    }
    for (i in 1:M) {
      para_list <- unpack_parameters(theta[i,], d, num_nodes)
      pred_y_dev <-
        forward_probagation(t(X_dev), para_list, 'relu')$ZL * sd_y_train + mean_y_train
      lik1 <- f_log_lk(para_list$loggamma)
      loggamma2 <-
        mean(log(diag(eigenMat)) - log(rowMeans((y_hat - y_train[, ridx]) ^ 2)))
      lik2 <- f_log_lk(loggamma2)
      if (lik2 > lik1) {
        para_list$loggamma <- loggamma2
        theta[i,] <- pack_parameters(para_list)
      }
    }

    if (!is.null(X_test)) {
      metrics <-
        evaluation(X_test, y_test, eigenMat, theta, num_nodes, scaling_coef)
      return(
        list(
          theta = theta,
          scaling_coef = scaling_coef,
          svgd_rmse = metrics$svgd_rmse,
          svgd_11 = metrics$svgd_11
        )
      )
    } else {
      return(list(theta = theta, scaling_coef = scaling_coef))
    }
  }
