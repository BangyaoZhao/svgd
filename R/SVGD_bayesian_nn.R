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
#' @param use_autodiff Whether to use autodiffr, default to FALSE.
#' @return A list containing:
#' \itemize{
#'   \item{theta: }{The estimated parameters in the vector format.}
#'   \item{scaling_coef: }{The scaling coefficient}
#'   \item{svgd_rmse: }{The RMSE on the training data}
#'   \item{svgd_11: }{log likelihood}
#' }
#' @importFrom stats dist median rgamma rnorm
#' @importFrom utils tail
#' @export SVGD_bayesian_nn

SVGD_bayesian_nn <-
  function(X_train,
           y_train,
           eigenMat = diag(x = apply(y_train, 2, var)),
           X_test = X_train,
           y_test = y_train,
           dev_split = 0.1,
           M = 20,
           num_nodes = c(ncol(X_train), ncol(y_train)),
           a0 = c(1, 1),
           b0 = c(0.1, 0.1),
           initial_values = FALSE) {
    n_layers <- length(num_nodes) + 1
    d <- ncol(X_train)
    n_data <- nrow(X_train)

    if (dim(eigenMat)[1] == 1) {
      eigenMat_inv <- as.matrix(1 / eigenMat)
    } else {
      eigenMat_inv <- diag(1 / diag(eigenMat))
    }

    para_cumsum <- parameter_cumsum(d, num_nodes)$para_cumsum
    num_vars <- tail(para_cumsum, 1)

    # Keep the last 10% (max 500) of training data points for model developing
    size_dev = min(round(dev_split * n_data), 500)
    X_dev <- X_train[-(1:(n_data - size_dev)), ]
    y_dev <- y_train[-(1:(n_data - size_dev)), ]
    X_train <- X_train[(1:(n_data - size_dev)), ]
    y_train <- y_train[(1:(n_data - size_dev)), ]

    X_train_unscaled = X_train
    y_train_unscaled = y_train

    # Normalize the data set
    X_train <- scale(X_train)
    y_train <- scale(y_train)
    mean_X_train <- attr(X_train, 'scaled:center')
    sd_X_train <- attr(X_train, 'scaled:scale')
    mean_y_train <- attr(y_train, 'scaled:center')
    sd_y_train <- attr(y_train, 'scaled:scale')
    scaling_coef <-
      list(
        mean_X_train = mean_X_train,
        sd_X_train = sd_X_train,
        mean_y_train = mean_y_train,
        sd_y_train = sd_y_train
      )

    scaled_eigenMat <- eigenMat / sd_y_train ^ 2
    if (dim(eigenMat)[1] == 1) {
      scaled_eigenMat_inv = as.matrix(1 / scaled_eigenMat)
    } else {
      scaled_eigenMat_inv <- diag(1 / diag(scaled_eigenMat))
    }

    y_train <- t(y_train)
    # Get the number of data points
    N0 <- nrow(X_train)

    if (is.matrix(initial_values)) {
      theta = initial_values
    } else {
      # Initialize the parameters
      theta <- matrix(rep(0, num_vars * M), nrow = M)
      for (i in 1:M) {
        theta_i <- initialization(d, num_nodes, a0, b0)

        # A better initialization for gamma
        ridx <- sample(N0, min(N0, 1000), replace = F)
        y_hat <-
          forward_probagation(t(X_train[ridx, ]), theta_i, 'relu')$ZL
        loggamma <-
          mean(log(diag(scaled_eigenMat)) - log(rowMeans((
            y_hat - y_train[, ridx]
          ) ^
            2)))
        theta_i$loggamma <- loggamma

        theta[i, ] <- pack_parameters(theta_i)
      }
    }

    return(
      list(
        X_train_scaled = X_train,
        y_train_scaled = y_train,
        X_train_unscaled = X_train_unscaled,
        y_train_unscaled = y_train_unscaled,
        X_dev = X_dev,
        y_dev = y_dev,
        X_test = X_test,
        y_test = y_test,
        scaling_coef = scaling_coef,
        eigenMat = eigenMat,
        eigenMat_inv = eigenMat_inv,
        scaled_eigenMat = scaled_eigenMat,
        scaled_eigenMat_inv = scaled_eigenMat_inv,
        num_vars = num_vars,
        d = d,
        N0 = N0,
        M = M,
        num_nodes = num_nodes,
        a0 = a0,
        b0 = b0,
        theta = theta
      )
    )
  }
