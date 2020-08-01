library("cmdstanr")

# using data from R iris dataset that comes with R
data(iris)

K <- 3 #outcomes
D <- 1 #number_predictors
N <- 150 #number of data points 
Beta_1 <- mean(iris$Sepal.Length[iris$Species=='setosa'])
sd_1 <- sd(iris$Sepal.Length[iris$Species=='setosa'])
Beta_2 <- mean(iris$Sepal.Length[iris$Species=='versicolor'])
sd_2 <- sd(iris$Sepal.Length[iris$Species=='versicolor'])
Beta_3 <- mean(iris$Sepal.Length[iris$Species=='virginica'])
sd_3 <- sd(iris$Sepal.Length[iris$Species=='virginica'])

x_1 <- rnorm(50,Beta_1,sd_1)
x_2 <- rnorm(50,Beta_2,sd_2)
x_3 <- rnorm(50,Beta_3,sd_3)

y <- c(rep(1,50),rep(2,50),rep(3,50))
x <- c(x_1,x_2,x_3)

x_matrix_implicit_intercept <- matrix(nrow=N,ncol=d+1)
x_matrix_implicit_intercept[,2] <- x
x_matrix_implicit_intercept[,1] <- rep(1,N)


stan_data_matrix <- list(N=N, K=k, D=d+1, x=x_matrix_implicit_intercept, y=y)
model <- cmdstan_model("multi_logit_regression.stan")
fit <- model$sample(data = stan_data_matrix,
                    output_dir = "output",
                    validate_csv = FALSE)
print(rstan::read_stan_csv(fit$output_files()))

sepal_length <- 5
setosa_v <- 15.43 + -2.84 * sepal_length
versicolor_v <- -3.82 + .7 * sepal_length
virginica_v <- -11.64 + 1.97*sepal_length

numerator <- exp(versicolor_v) + exp(virginica_v) + exp(setosa_v)
setosa_p <- exp(setosa_v)/numerator
versicolor_p <- exp(versicolor_v)/numerator
virginica_p <- exp(virginica_v)/numerator
print(sprintf("setosa=%.2f, versicolor_p=%.2f, virginica_p=%.2f",
              setosa_p, versicolor_p, virginica_p))

stan_data_matrix <- list(N=N, K=k, D=d+1, x=x_matrix_implicit_intercept, y=y)
model <- cmdstan_model("multi_logit_regression_dummy_i_i2.stan")
fit <- model$sample(data = stan_data_matrix,
                        output_dir = "output",
                    validate_csv = FALSE)
print(rstan::read_stan_csv(fit$output_files()))

print(paste("ran stan executable: ", model$exe_file()))

print(fit$summary())

#=======

D_i_i <- d+1
x_i_i <- matrix(ncol=D_i_i,nrow=N)
x_i_i[,2] <- x
x_i_i[,1] <- rep(1,N)

data_i_i <- list(D=D_i_i, N=N, K=k, x=x_i_i, y=y)

model_2_1 <- cmdstan_model("multi_logit_regression_dummy_i_i.stan")
fit_2_1 <- model_2_1$sample(data = data_i_i,
                            output_dir = "output",validate_csv = FALSE)
//print(fit_1$summary())
print(rstan::read_stan_csv(fit_2_1$output_files()))


============
  
data_3 <- list(D=d, N=N, K=k, x=x, y=y)

model_3 <- cmdstan_model("multi_logit_regression_centered.stan")
fit_3 <- model_3$sample(data = data_i_i,
                            output_dir = "output",validate_csv = FALSE)
//print(fit_1$summary())
print(rstan::read_stan_csv(fit_3$output_files()))



# covariate matrix
mX = matrix(rnorm(1000), 200, 5)

# coefficients for each choice
vCoef1 = rep(0, 5)
vCoef2 = rnorm(5)
vCoef3 = rnorm(5)

# vector of probabilities
vProb = cbind(exp(mX%*%vCoef1), exp(mX%*%vCoef2), exp(mX%*%vCoef3))

# multinomial draws
mChoices = t(apply(vProb, 1, rmultinom, n = 1, size = 1))
dfM = cbind.data.frame(y = apply(mChoices, 1, function(x) which(x==1)), mX)
library(nnet)
m <- multinom(y ~ ., data = dfM[,-2])
summary(m)



# simulate data
n <- 1000
k <- 3 #number of outcomes
category_1 <- rnorm(n,2,1)
category_2 <- rnorm(n,5,1)
category_3 <- rnorm(n,0,1)

rmultinom(n, size = 3, prob = c(0,1,1))
x <- runif(n, 0, 10)
y <- rnorm(n, alpha_s + beta_s * x, sigma_s)

print(sprintf(paste("simulation parameters are: alpha_s=%.1f",
              "beta_s=%.1f, sigma_s=%.1f, n=%d"),
              alpha_s, beta_s, sigma_s, n))


k <- 1 # 1 column for coefficient on x
x_matrix <- matrix(ncol = 1, nrow = n)
x_matrix[, k] <- x
stan_data_matrix <- list(N = n, K = k, x = x_matrix, y = y)
model_3 <- cmdstan_model("regression_matrix.stan")
fit_3 <- model_3$sample(data = stan_data_matrix, num_chains = 4,
                      output_dir = "output")
print(paste("ran stan executable: ", model_3$exe_file()))
print(fit_3$summary())

#=============runs matrix version with integrated intercept==========
k_2 <- 2
x_matrix <- matrix(ncol = k_2, nrow = n)
x_matrix[, 1] <- rep(1, n) # intercept is always 1
x_matrix[, 2] <- x # predictor values
model_4 <- cmdstan_model("regression_matrix_w_intercept.stan")
stan_data_matrix_w_intercept <- list(N = n, K = k_2, x = x_matrix, y = y)
fit_4 <- model_4$sample(data = stan_data_matrix_w_intercept,
                      num_chains = 4, output_dir = "output")
print(paste("ran stan executable: ", model_4$exe_file()))
print(fit_4$summary())

#=============runs QR version==========

model_5 <- cmdstan_model("regression_QR.stan")
fit_5 <- model_5$sample(data = stan_data_matrix_w_intercept,
                      num_chains = 4, output_dir = "output")
print(paste("ran stan executable: ", model_5$exe_file()))
print(fit_5$summary())
