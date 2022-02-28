# This code has been tested with R version 3.3.2
library(mvtnorm) # Tested with version 1.0-5
library(crossEstimation) # Tested with version 1.0

calc_mahal <- function(x, c_ind, n, vi) {
  # Function to calculate Mahalanobis distance
  if (dim(x)[2] == 1) {
    diff_x <- mean(x[t_ind]) - mean(x[c_ind])
    m <- t(diff_x) %*% VI %*% diff_x
  } else {
    diff_x <- colMeans(x[t_ind, ]) - colMeans(x[c_ind, ])
    m <- t(diff_x) %*% VI %*% diff_x
  }

  return((n / 4 * m)[[1]])
}

rlkjcorr <- function ( n , K , eta = 1 ) {
  # Draw correlated random variables, see code at
  # https://github.com/brockk/trialr/blob/master/R/rlkjcorr.R

  stopifnot(is.numeric(K), K >= 2, K == as.integer(K))
  stopifnot(eta > 0)
  #if (K == 1) return(matrix(1, 1, 1))

  f <- function() {
    alpha <- eta + (K - 2)/2
    r12 <- 2 * rbeta(1, alpha, alpha) - 1
    R <- matrix(0, K, K) # upper triangular Cholesky factor until return()
    R[1,1] <- 1
    R[1,2] <- r12
    R[2,2] <- sqrt(1 - r12^2)
    if(K > 2) for (m in 2:(K - 1)) {
      alpha <- alpha - 0.5
      y <- rbeta(1, m / 2, alpha)

      # Draw uniformally on a hypersphere
      z <- rnorm(m, 0, 1)
      z <- z / sqrt(crossprod(z)[1])

      R[1:m,m+1] <- sqrt(y) * z
      R[m+1,m+1] <- sqrt(1 - y)
    }
    return(crossprod(R))
  }
  R <- replicate( n , f() )
  if ( dim(R)[3]==1 ) {
    R <- R[,,1]
  } else {
    # need to move 3rd dimension to front, so conforms to array structure that Stan uses
    R <- aperm(R,c(3,1,2))
  }
  return(R)
}

n <- 50
n1 <- 25

REPS <- 1000
samples <- 100

# The code is run for all relevant combinations of tau, K, corr and hetero.
tau <- 0
K <- 2
corr <- "false"
hetero <- "false"

FILENAME <- "sim_tibs.csv"

beta <- rep(1/sqrt(K), K)

df_all <- data.frame()

for (j in 1:REPS){

  if (corr == "false" || K == 1) {
    Z <- rmvnorm(n=n, mean=rep(0, K), sigma=diag(K))
  } else {
    Z <- rmvnorm(n=n, mean=rep(0, K), sigma=rlkjcorr(1, K, 1))
  }

  u <- rnorm(n, 0, 1)

  if (hetero == "false") {
    alpha <- 0
  } else if (hetero == "true") {
    alpha <- rnorm(n, 0, 1)
  }

  Y0 <- Z%*%beta + u
  Y1 <- tau + Y0 + alpha
  SATE <- mean(Y1) - mean(Y0)
  VI <- solve(cov(Z))

  OUT <- array(numeric(),c(samples,4))

  for (i in 1:samples) {

    t_ind <- sort(sample(1:n, n1, replace=FALSE))
    c_ind <- setdiff(1:n, t_ind)

    Yobs <- Y0
    Yobs[t_ind] <- Y1[t_ind]

    W = rep(0, n)
    W[t_ind] = 1

    m <- calc_mahal(Z, c_ind, n, VI)

    q <- ate.glmnet(Z, Yobs, W, alpha = 1, nfolds = 10, conf.level = 0.95,
                    method = "joint", lambda.choice = "lambda.min")
    if (hetero == "false") {
      p_tibs <- (1 - pnorm(abs(q$tau/ sqrt(q$var)))) * 2
    } else if (hetero == "true" && tau == 0) {
      p_tibs <- (1 - pnorm(abs((q$tau - SATE) / sqrt(q$var)))) * 2
    } else if (hetero == "true" && tau != 0) {
      p_tibs <- (1 - pnorm(abs(q$tau/ sqrt(q$var)))) * 2
    }

    s_tibs <- p_tibs <= 0.05
    OUT[i, ] <- c(q$tau, p_tibs, s_tibs, m)

  }

  colnames(OUT) <- c("TIBS_est","p_TIBS", "s_TIBS", "M")
  df <- data.frame(OUT)
  df$quantile <- rank(df$M)
  df$SATE <- SATE
  df$TIBS_mse <- (df$TIBS_est - df$SATE)^2

  df_all <- rbind(df_all, df)

  print(sprintf("K: %s, iter: %s", K, j))

}

df_agg <- aggregate(df_all, by=list(Category=df_all$quantile), FUN=mean)

col_order <- c("quantile", "TIBS_est", "p_TIBS", "s_TIBS", "M", "SATE",
               "TIBS_mse")
df_agg <- df_agg[, col_order]
df_agg$corr <- corr
df_agg$hetero <- hetero
df_agg$reps_in_sample <- samples
df_agg$reps <- REPS
df_agg$n <- n
df_agg$n0 <- n - n1
df_agg$k <- K
df_agg$tau <- tau

write.table(df_agg, FILENAME, sep = ",", row.names=FALSE,
            append=TRUE, col.names = !file.exists(FILENAME))
