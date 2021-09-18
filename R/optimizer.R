#' @export optimizer

optimizer <-
  function(svgd_bnn,
           master_stepsize = 0.001,
           auto_corr = 0.9,
           max_iter = 1000,
           batch_size = 100,
           method = 'adam',
           use_autodiff = F,
           monitor_metric = 'Rsq',
           tol = 1e-4,
           check_freq = 50
  ) {
    X_train = svgd_bnn$X_train_scaled
    y_train = svgd_bnn$y_train_scaled
    X_train_unscaled = svgd_bnn$X_train_unscaled
    y_train_unscaled = svgd_bnn$y_train_unscaled
    X_test = svgd_bnn$X_test
    y_test = svgd_bnn$y_test
    scaling_coef = svgd_bnn$scaling_coef
    eigenMat_inv = svgd_bnn$eigenMat_inv
    N0 = svgd_bnn$N0
    d = svgd_bnn$d
    M = svgd_bnn$M
    num_vars = svgd_bnn$num_vars
    num_nodes = svgd_bnn$num_nodes
    theta = svgd_bnn$theta
    a0 = svgd_bnn$a0
    b0 = svgd_bnn$b0

    grad_theta <-
      matrix(rep(0, times = M * num_vars), nrow = M, ncol = num_vars)

    fudge_factor <- 1e-6

    beta_1 <- 0.9
    beta_2 <- 0.999
    epsilon <- 1e-8
    m_t <- 0
    v_t <- 0
    historical_grad <- 0

    metric = c()
    val_metric = c()
    pb <- txtProgressBar(min = 0, max = max_iter, style = 3)
    for (i in 0:(max_iter - 1)) {
      # Sub-sampling step
      batch <- ((i * batch_size):((i + 1) * batch_size - 1)) %% N0
      batch <- batch + 1

      for (j in 1:M) {
        para_list <- unpack_parameters(theta[j, ], d, num_nodes)
        grad_theta[j, ] <-
          gradient(
            t(X_train[batch, ]),
            matrix(y_train[, batch], ncol = batch_size),
            eigenMat_inv,
            para_list,
            N0,
            a0,
            b0
          )
      }

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
      svgd_bnn$theta = theta
      #cat(i + 1, ' ')


      if ((i+1) %% check_freq == 0) {
        svgd_bnn_prev = svgd_bnn
        svgd_bnn_prev$theta = theta_prev
        metric_prev = evaluation(svgd_bnn_prev, X_train_unscaled, y_train_unscaled)[[sprintf('svgd_%s', monitor_metric)]]
        metric_new = evaluation(svgd_bnn, X_train_unscaled, y_train_unscaled)[[sprintf('svgd_%s', monitor_metric)]]

        val_metric_prev = evaluation(svgd_bnn_prev, X_test, y_test)[[sprintf('svgd_%s', monitor_metric)]]
        val_metric_new = evaluation(svgd_bnn, X_test, y_test)[[sprintf('svgd_%s', monitor_metric)]]

        metric = c(metric, metric_new)
        val_metric = c(val_metric, val_metric_new)

        plot(metric, type = 'l', col = 'blue', xaxt = 'n', xlab = 'iter', ylab = monitor_metric)
        lines(val_metric, col = 'red')
        legend(x = 'topleft', legend = c("train", "validation"), col = c('blue', 'red'), lty = 1)


        axis(1, at = 1:length(metric), label = check_freq*(1:length(metric)))
        if (abs( (metric_new - metric_prev) / metric_prev) < tol) {
          cat('\n', 'early stopping at iter', i+1)

          return(svgd_bnn)
        }
      }
      setTxtProgressBar(pb, i + 1)
    }
    close(pb)
    return(svgd_bnn)
  }
