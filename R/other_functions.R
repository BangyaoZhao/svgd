##########################################
##   Neural Network Utility Functions   ##
##########################################

relu <- function(A) {
  Z <- pmax(A, 0)
  cache <- A
  return(list(Z = Z, cache = cache))
}

identity <- function(A) {
  Z <- A
  cache <- A
  return(list(Z = Z, cache = cache))
}

relu_backward <- function(dZ, cache) {
  A <- cache
  dA <- dZ
  dA[A <= 0] <- 0
  return(dA)
}

identity_backward <- function(dZ, cache) {
  A <- cache
  dA <- dZ
  return(dA)
}

log_posterior_backward <-
  function(ZL, Y, eigenMat_inv, parameters) {
    loggamma <- parameters$loggamma
    dA <- exp(loggamma) * eigenMat_inv %*% (Y - ZL)
    return(dA)
  }

##########################################
##      Forward Propagation Module      ##
##########################################

linear_forward <- function(Z, W, b) {
  A <- W %*% Z + b
  cache <- list(Z = Z, W = W, b = b)
  return(list(A = A, cache = cache))
}

one_step_forward <- function(Z_prev, W, b, activation = 'relu') {
  tmp1 <- linear_forward(Z_prev, W, b)
  A <- tmp1$A
  linear_cache <- tmp1$cache

  if (activation == 'relu') {
    tmp2 <- relu(A)
    Z <- tmp2$Z
    activation_cache <- tmp2$cache
  } else {
    tmp3 <- identity(A)
    Z <- tmp3$Z
    activation_cache <- tmp3$cache
  }

  cache <-
    list(linear_cache = linear_cache, activation_cache = activation_cache)

  return(list(Z = Z, cache = cache))
}

forward_probagation <- function(X, parameters, activation) {
  caches <- list()

  Z <- X
  L <- (length(parameters) - 2) / 2

  for (l in 1:(L - 1)) {
    Z_prev <- Z
    tmp1 <-
      one_step_forward(Z_prev, parameters[[sprintf('W%s', l)]], parameters[[sprintf('b%s', l)]], activation)
    Z <- tmp1$Z
    caches[[sprintf('L%s', l)]] <- tmp1$cache
  }

  tmp2 <-
    one_step_forward(Z, parameters[[sprintf('W%s', L)]], parameters[[sprintf('b%s', L)]], 'identity')
  ZL <- tmp2$Z
  caches[[sprintf('L%s', L)]] <- tmp2$cache

  return(list(ZL = ZL, caches = caches))
}

##########################################
##      Backward Propagation Module     ##
##########################################

linear_backward <- function(dA, cache) {
  Z_prev <- cache$Z
  W <- cache$W
  b <- cache$b

  dW <- dA %*% t(Z_prev)
  db <- rowSums(dA)
  dZ_prev <- t(W) %*% dA

  return(list(dZ_prev = dZ_prev, dW = dW, db = db))
}

one_step_backward <-
  function(dZ,
           cache,
           activation,
           parameters = NULL,
           Y = NULL,
           ZL = NULL,
           eigenMat_inv = NULL) {
    linear_cache <- cache$linear_cache
    activation_cache <- cache$activation_cache

    if (activation == 'relu') {
      dA <- relu_backward(dZ, activation_cache)
    } else {
      dA <- log_posterior_backward(ZL, Y, eigenMat_inv, parameters)
    }

    tmp <- linear_backward(dA, linear_cache)
    dZ_prev <- tmp$dZ_prev
    dW <- tmp$dW
    db <- tmp$db

    return(list(dZ_prev = dZ_prev, dW = dW, db = db))
  }

backward_probagation <-
  function(ZL, Y, eigenMat_inv, caches, parameters) {
    grads <- list()
    L <- (length(parameters) - 2) / 2

    current_cache <- caches[[sprintf('L%s', L)]]
    tmp1 <-
      one_step_backward(
        cache = current_cache,
        activation = 'identity',
        Y = Y,
        ZL = ZL,
        eigenMat_inv = eigenMat_inv,
        parameters = parameters,
        dZ = NULL
      )
    grads[[sprintf('dZ%s', L - 1)]] <- tmp1$dZ_prev
    grads[[sprintf('dW%s', L)]] <- tmp1$dW
    grads[[sprintf('db%s', L)]] <- tmp1$db

    for (l in (L - 1):1) {
      current_cache <- caches[[sprintf('L%s', l)]]
      tmp2 <-
        one_step_backward(dZ = grads[[sprintf('dZ%s', l)]],
                          cache = current_cache,
                          activation = 'relu')
      grads[[sprintf('dZ%s', l - 1)]] <- tmp2$dZ_prev
      grads[[sprintf('dW%s', l)]] <- tmp2$dW
      grads[[sprintf('db%s', l)]] <- tmp2$db
    }

    return(grads)
  }

##########################################
##           Compute Gradients          ##
##########################################

gradient <-
  function(X_batch,
           y_batch,
           eigenMat_inv,
           parameters,
           N,
           a0,
           b0) {
    N0 <- dim(X_batch)[2]
    L <- (length(parameters) - 2) / 2
    output_dim <- dim(y_batch)[1]

    para_vector <- pack_parameters(parameters)
    num_vars <- length(para_vector)

    tmp = forward_probagation(X_batch, parameters, activation = 'relu')
    ZL <- tmp$ZL
    caches <- tmp$caches

    raw_grads <-
      backward_probagation(ZL, y_batch, eigenMat_inv, caches, parameters)

    sum_of_square = sum(para_vector[1:(num_vars - 2)] ^ 2)

    # for (l in 1:L) {
    #   sum_of_square = sum_of_square + sum(parameters[[sprintf('W%s', l)]]^2) + sum(parameters[[sprintf('b%s', l)]]^2)
    # }


    loggamma <- parameters$loggamma
    d_loggamma <-
      N * output_dim / 2 - exp(loggamma) / 2 * sum((ZL - y_batch) ^ 2 * diag(eigenMat_inv)) * N / N0 + a0 - b0 * exp(loggamma)

    loglambda <- parameters$loglambda
    d_loglambda <-
      (num_vars - 2) / 2 - exp(loglambda) * sum_of_square / 2 + a0 - b0 * exp(loglambda)

    grads <- list()
    for (l in 1:L) {
      grads[[sprintf('W%s', l)]] <-
        raw_grads[[sprintf('dW%s', l)]] * N / N0 - exp(loglambda) * parameters[[sprintf('W%s', l)]]
      grads[[sprintf('b%s', l)]] <-
        raw_grads[[sprintf('db%s', l)]] * N / N0 - exp(loglambda) * parameters[[sprintf('b%s', l)]]
    }

    grads$loggamma <- d_loggamma
    grads$loglambda <- d_loglambda

    d_para_vector <- pack_parameters(grads)
    return(d_para_vector)
  }

gradient_auto <-
  function(X_batch,
           y_batch,
           theta_mat,
           eigenMat_inv,
           parameters,
           N,
           a0,
           b0) {
    d = nrow(X_batch)
    N0 = ncol(X_batch)

    theta = pack_parameters(parameters)
    num_nodes = unlist(lapply(parameters, nrow))
    names(num_nodes) = NULL

    L = length(num_nodes)
    num_nodes_previous = c(d, num_nodes)

    last = parameter_cumsum(d, num_nodes)$para_cumsum
    first = c(0, last) + 1
    n_params = length(theta)
    n_last_layer = num_nodes[length(num_nodes)]

    log_p = function(theta) {
      loglambda = theta[n_params]
      loggamma = theta[n_params - 1]

      t = theta[c(n_params - 1, n_params)]
      l = sum(t * a0 - exp(t) * b0)
      l = l + loglambda * (n_params - 2) / 2 - sum(theta[1:(n_params - 2)] ^
                                                     2) * exp(loglambda) / 2

      y_hat_batch = X_batch
      #browser()
      for (i in 1:L) {
        W <-
          array(theta[first[2 * i - 1]:last[2 * i - 1]], c(num_nodes[i], num_nodes_previous[i]))
        b <-
          array(theta[rep(first[2 * i]:last[2 * i], N0)], c(num_nodes[i], N0))
        y_hat_batch = W %m% y_hat_batch + b
        if (i < L) {
          y_hat_batch = (abs(y_hat_batch) + y_hat_batch) / 2
        }
      }


      l1 = N * n_last_layer * loggamma / 2 - N / N0 * sum(eigenMat_inv %m% ((y_hat_batch -
                                                                               y_batch) ^ 2)) * exp(loggamma) / 2#
      l = l + l1
      return(l)
    }
    #browser()
    log_p <-
      ad_variant(log_p,
                 silent = TRUE,
                 checkArgs = list(theta))

    dlog_p = makeGradFunc(
      log_p,
      x = theta,
      debug = FALSE,
      use_tape = TRUE,
      compiled = TRUE
    )

    #browser()

    return(t(apply(theta_mat, 1, dlog_p)))#,
  }
##########################################
##        SVGD Utility Functions        ##
##########################################

svgd_kernel <- function(theta, h = -1) {
  # theta: M by p matrix, M is the number of particles and p is the number of total parameters.
  pairwise_distance <-
    as.matrix(dist(theta, upper = TRUE, diag = TRUE))
  pairwise_distance <- pairwise_distance ^ 2
  if (h < 0) {
    # if h < 0, using median trick
    h <- median(pairwise_distance)
    h <- sqrt(0.5 * h / log(dim(theta)[1] + 1))
  }

  if (h == 0) {
    h <- 1
  }
  # compute the rbf kernel
  Kxy <- exp(-pairwise_distance / h ** 2 / 2)

  dxkxy <- -Kxy %*% theta
  sumkxy <- rowSums(Kxy)
  for (i in 1:dim(theta)[2]) {
    dxkxy[, i] <- dxkxy[, i] + theta[, i] * sumkxy
  }
  dxkxy <- dxkxy / (h ** 2)
  return(list(Kxy = Kxy, dxkxy = dxkxy))
}

parameter_cumsum <- function(d, num_nodes) {
  # d: integer, number of features in X
  # num_nodes: vector of integers, number of nodes in each hidden layer
  num_of_para <- c()
  for (i in 1:length(num_nodes)) {
    if (i == 1) {
      num_of_para <-
        c(num_of_para, num_nodes[1] * d, num_nodes[1])  # weight + bias
    } else {
      num_of_para <-
        c(num_of_para, num_nodes[i] * num_nodes[i - 1], num_nodes[i])
    }
  }
  num_of_para <- c(num_of_para, 1, 1)
  return(list(
    num_of_para = num_of_para,
    para_cumsum = cumsum(num_of_para)
  ))
}

pack_parameters <- function(para_list) {
  # pack parameters to a long vector
  para_vector <- c()
  for (para in para_list) {
    para_vector <- c(para_vector, c(para))
  }
  return(para_vector)
}

unpack_parameters <- function(para_vector, d, num_nodes) {
  # restore parameters from a long vector
  # weigth matrix has the shape n_l by n_(l-1)
  para_cumsum <- parameter_cumsum(d, num_nodes)$para_cumsum
  para_cumsum <- c(0, para_cumsum)
  num_layers <- length(num_nodes)
  para_list <- list()
  for (i in 1:(2 * num_layers)) {
    if (i %% 2 == 1) {
      begin <- para_cumsum[i] + 1
      end <- para_cumsum[i + 1]
      layer <- ceiling(i / 2)
      para_list[[sprintf('W%s', layer)]] <-
        matrix(para_vector[begin:end], nrow = num_nodes[layer])
    } else {
      begin <- para_cumsum[i] + 1
      end <- para_cumsum[i + 1]
      layer <- ceiling(i / 2)
      para_list[[sprintf('b%s', layer)]] <- para_vector[begin:end]
    }
  }
  para_list$loggamma <- para_vector[length(para_vector) - 1]
  para_list$loglambda <- para_vector[length(para_vector)]
  return(para_list)
}

initialization <- function(d, num_nodes, a0, b0) {
  # parameter initialization
  # d: integer, number of features in X
  # num_nodes: vector of integers, number of nodes in hidden layers
  # a0, b0: shape and scale parameters in gamma distribution, respectively
  # In this function, we use Xavier initialization for weights in the neural network;
  # bias terms are set to be 0; gamma and lambda are drawn from the prior Gamma(a0, b0)
  num_of_para <- parameter_cumsum(d, num_nodes)$num_of_para
  num_layers <- length(num_nodes)
  para_list <- list()

  for (i in 1:num_layers) {
    if (i == 1) {
      para_list[[sprintf('W%s', 1)]] <-
        1 / sqrt(d + 1) * matrix(rnorm(d * num_nodes[1]), nrow = num_nodes[1])
      para_list[[sprintf('b%s', 1)]] <- rep(0, num_nodes[1])
    } else {
      para_list[[sprintf('W%s', i)]] <-
        1 / sqrt(num_nodes[i - 1] + 1) * matrix(rnorm(num_nodes[i - 1] * num_nodes[i]), nrow = num_nodes[i])
      para_list[[sprintf('b%s', i)]] <- rep(0, num_nodes[i])
    }

  }
  para_list$loggamma = log(rgamma(1, shape = a0, scale = b0))
  para_list$loglambda = log(rgamma(1, shape = a0, scale = b0))
  return(para_list)
}

##########################################
##      Train, Predict and Evaluate     ##
##########################################

optimizer <-
  function(X_train,
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
           method,
           use_autodiff) {
    grad_theta <-
      matrix(rep(0, times = M * num_vars), nrow = M, ncol = num_vars)

    fudge_factor <- 1e-6

    beta_1 <- 0.9
    beta_2 <- 0.999
    epsilon <- 1e-8
    m_t <- 0
    v_t <- 0
    historical_grad <- 0

    for (i in 0:(max_iter - 1)) {
      # Sub-sampling step
      batch <- ((i * batch_size):((i + 1) * batch_size - 1)) %% N0
      batch <- batch + 1

      if (use_autodiff) {
        para_list <- unpack_parameters(theta[1,], d, num_nodes)
        grad_theta = gradient_auto(
          t(X_train[batch, ]),
          matrix(y_train[, batch], ncol = batch_size),
          theta,
          eigenMat_inv,
          para_list,
          N0,
          a0,
          b0
        )
      } else {
        for (j in 1:M) {
          para_list <- unpack_parameters(theta[j,], d, num_nodes)
          grad_theta[j,] <-
            gradient(
              t(X_train[batch,]),
              matrix(y_train[, batch], ncol = batch_size),
              eigenMat_inv,
              para_list,
              N0,
              a0,
              b0
            )
        }
      }
      #browser()

      # Calculate the kernel matrix
      kernel_list <- svgd_kernel(theta = theta)
      grad_theta <-
        (kernel_list$Kxy %*% grad_theta + kernel_list$dxkxy) / M

      if (method == 'adagrad') {
        if (i == 0) {
          historical_grad <- historical_grad + grad_theta ^ 2
        } else {
          historical_grad <-
            auto_corr * historical_grad + (1 - auto_corr) * grad_theta ^ 2
        }
        adj_grad <-
          grad_theta / (fudge_factor + sqrt(historical_grad))
        theta_prev = theta
        theta <- theta + master_stepsize * adj_grad
      } else {
        t <- i + 1
        g_t <- grad_theta
        m_t <- beta_1 * m_t + (1 - beta_1) * g_t
        v_t <- beta_2 * v_t + (1 - beta_2) * (g_t ^ 2)
        m_cap <- m_t / (1 - (beta_1 ** t))
        v_cap <- v_t / (1 - (beta_2 ** t))
        theta_prev = theta
        theta <-
          theta + (master_stepsize * m_cap) / (sqrt(v_cap) + epsilon)
      }
      if (((i + 1) %% 50 == 0) &
          (mean((theta - theta_prev) ^ 2) < 1e-10)) {
        cat('early stopping at iter', i + 1)
        break
      }
    }
    return(theta)
  }

evaluation <-
  function(X_test,
           y_test,
           eigenMat,
           theta,
           num_nodes,
           scaling_coef) {
    M <- dim(theta)[1]
    mean_X_train <- scaling_coef[[1]]
    sd_X_train <- scaling_coef[[2]]
    mean_y_train <- scaling_coef[[3]]
    sd_y_train <- scaling_coef[[4]]
    y_test <- t(y_test)

    output_dim <- dim(y_test)[1]
    det_eigenMat <- cumprod(diag(eigenMat))[output_dim]
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
        (exp(loggamma)) ^ (output_dim / 2) / ((2 * pi) ^ (output_dim / 2) * sqrt(det_eigenMat)) * exp(-exp(loggamma) / 2 * colSums((pred_y_test[i, ,] - y_test) ^
                                                                                                                                     2 * diag(inv_eigenMat)))
    }
    pred <- apply(pred_y_test, c(2, 3), mean)

    svgd_rmse <- sqrt(mean((pred - y_test) ^ 2))
    svgd_ll <- mean(log(colMeans(prob)))
    return(list(svgd_rmse = svgd_rmse, svgd_ll = svgd_ll))
  }
