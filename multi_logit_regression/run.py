""" Example program that generates data, compiles and runs regression.stan"""
from cmdstanpy import CmdStanModel
from numpy.random import uniform, normal
from numpy import exp, concatenate, array
from scipy.stats import binom

# Run from command line: Python regression_1.py
N = 150
D = 1
K = 3
setosa_sepal_mean = 5.006
setosa_sepal_sd = 0.3524897
versicolor_sepal_mean = 5.936
versicolor_sepal_sd = 0.5161711
virginica_sepal_mean = 6.588
virginica_sepal_sd = 0.6358796

setosa_x = normal(size=50, loc=setosa_sepal_mean, scale=setosa_sepal_sd)
versicolor_x = normal(size=50, loc=versicolor_sepal_mean, scale=versicolor_sepal_sd)
virginica_x = normal(size=50, loc=virginica_sepal_mean, scale=virginica_sepal_sd)

x = concatenate((setosa_x,versicolor_x,virginica_x))
x_intercept = [1] * N
y = [1] * 50 + [2] * 50 + [3] * 50
x_matrix = array([x_intercept,x]).T

#output = "generating parameters are: setosa_sepal_mean={:.1f}, beta_truth={:.1f}, n={:d}"
#print(output.format(alpha_truth, beta_truth, n))

stan_data = {'N': N, 'K': K, 'D': D+1, 'x': x_matrix, 'y': y}

stan_program = CmdStanModel(stan_file='multi_logit_regression.stan')
stan_program.compile()
fit = stan_program.sample(data=stan_data, output_dir='output')
print("running stan executable: ", stan_program.exe_file)
print(fit.summary())

# lp__      -103.353000  0.053929  1.72234  ...  1019.99  6.74140  1.00469
# beta[1,1]   14.771200  0.092153  3.52694  ...  1464.78  9.68112  1.00292
# beta[1,2]   -1.852440  0.085656  3.19734  ...  1393.37  9.20917  1.00342
# beta[1,3]  -13.075700  0.087151  3.36450  ...  1490.36  9.85022  1.00178
# beta[2,1]   -2.485960  0.081389  2.85278  ...  1228.58  8.12003  1.00227
# beta[2,2]    0.588428  0.081007  2.84162  ...  1230.50  8.13273  1.00227
# beta[2,3]    2.402020  0.080873  2.85609  ...  1247.21  8.24315  1.00202

from math import exp

sepal_length = 5
setosa_v = 14.771200 + -2.485960 * sepal_length
versicolor_v = -1.852440 + 0.588428 * sepal_length
virginica_v = -13.075700 + 2.402020 *sepal_length

numerator = exp(versicolor_v) + exp(virginica_v) + exp(setosa_v)
setosa_p = exp(setosa_v)/numerator
versicolor_p = exp(versicolor_v)/numerator
virginica_p = exp(virginica_v)/numerator
#output = "generating parameters are: setosa_sepal_mean={:.1f}, beta_truth={:.1f}, n={:d}"
#print(output.format(alpha_truth, beta_truth, n))


output = "setosa={:.2f}, versicolor_p={:.2f}, virginica_p={:.2f}"
print(output.format(setosa_p, versicolor_p, virginica_p))

# dummy encoding
stan_program = CmdStanModel(stan_file="multi_logit_regression_dummy_i_i.stan")
stan_program.compile()
fit = stan_program.sample(data=stan_data, output_dir='output')
print("running stan executable: ", stan_program.exe_file)
print(fit.summary())

# name                                         ...                             
# lp__          -100.49900  0.047733  1.45537  ...   929.615  15.59230  1.00114
# beta_raw[1,1]  -34.52250  0.285684  6.23211  ...   475.882   7.98190  1.00781
# beta_raw[1,2]  -42.36710  0.295120  6.58277  ...   497.529   8.34499  1.00614
# beta_raw[2,1]    6.35112  0.053149  1.15514  ...   472.372   7.92304  1.00782
# beta_raw[2,2]    7.62143  0.054725  1.20601  ...   485.650   8.14575  1.00632
# beta[1,1]        0.00000  0.000000  0.00000  ...  2000.000  33.54570      NaN
# beta[1,2]      -34.52250  0.285684  6.23211  ...   475.882   7.98190  1.00781
# beta[1,3]      -42.36710  0.295120  6.58277  ...   497.529   8.34499  1.00614
# beta[2,1]        0.00000  0.000000  0.00000  ...  2000.000  33.54570      NaN
# beta[2,2]        6.35112  0.053149  1.15514  ...   472.372   7.92304  1.00782
# beta[2,3]        7.62143  0.054725  1.20601  ...   485.650   8.14575  1.00632





# 
# D_i_i <- d+1
# x_i_i <- matrix(ncol=D_i_i,nrow=N)
# x_i_i[,2] <- x
# x_i_i[,1] <- rep(1,N)
# 
# data_i_i <- list(D=D_i_i, N=N, K=k, x=x_i_i, y=y)
# 
# model_2_1 <- cmdstan_model("multi_logit_regression_dummy_i_i.stan")
# fit_2_1 <- model_2_1$sample(data = data_i_i,
#                             output_dir = "output",validate_csv = FALSE)
# //print(fit_1$summary())
# print(rstan::read_stan_csv(fit_2_1$output_files()))
# 
# 
# ============
#   
# data_3 <- list(D=d, N=N, K=k, x=x, y=y)
# 
# model_3 <- cmdstan_model("multi_logit_regression_centered.stan")
# fit_3 <- model_3$sample(data = data_i_i,
#                             output_dir = "output",validate_csv = FALSE)
# //print(fit_1$summary())
# print(rstan::read_stan_csv(fit_3$output_files()))
# 
# 
# 
# # covariate matrix
# mX = matrix(rnorm(1000), 200, 5)
# 
# # coefficients for each choice
# vCoef1 = rep(0, 5)
# vCoef2 = rnorm(5)
# vCoef3 = rnorm(5)
# 
# # vector of probabilities
# vProb = cbind(exp(mX%*%vCoef1), exp(mX%*%vCoef2), exp(mX%*%vCoef3))
# 
# # multinomial draws
# mChoices = t(apply(vProb, 1, rmultinom, n = 1, size = 1))
# dfM = cbind.data.frame(y = apply(mChoices, 1, function(x) which(x==1)), mX)
# library(nnet)
# m <- multinom(y ~ ., data = dfM[,-2])
# summary(m)
# 
# 
# 
# # simulate data
# n <- 1000
# k <- 3 #number of outcomes
# category_1 <- rnorm(n,2,1)
# category_2 <- rnorm(n,5,1)
# category_3 <- rnorm(n,0,1)
# 
# rmultinom(n, size = 3, prob = c(0,1,1))
# x <- runif(n, 0, 10)
# y <- rnorm(n, alpha_s + beta_s * x, sigma_s)
# 
# print(sprintf(paste("simulation parameters are: alpha_s=%.1f",
#               "beta_s=%.1f, sigma_s=%.1f, n=%d"),
#               alpha_s, beta_s, sigma_s, n))
# 
# 
# k <- 1 # 1 column for coefficient on x
# x_matrix <- matrix(ncol = 1, nrow = n)
# x_matrix[, k] <- x
# stan_data_matrix <- list(N = n, K = k, x = x_matrix, y = y)
# model_3 <- cmdstan_model("regression_matrix.stan")
# fit_3 <- model_3$sample(data = stan_data_matrix, num_chains = 4,
#                       output_dir = "output")
# print(paste("ran stan executable: ", model_3$exe_file()))
# print(fit_3$summary())
# 
# #=============runs matrix version with integrated intercept==========
# k_2 <- 2
# x_matrix <- matrix(ncol = k_2, nrow = n)
# x_matrix[, 1] <- rep(1, n) # intercept is always 1
# x_matrix[, 2] <- x # predictor values
# model_4 <- cmdstan_model("regression_matrix_w_intercept.stan")
# stan_data_matrix_w_intercept <- list(N = n, K = k_2, x = x_matrix, y = y)
# fit_4 <- model_4$sample(data = stan_data_matrix_w_intercept,
#                       num_chains = 4, output_dir = "output")
# print(paste("ran stan executable: ", model_4$exe_file()))
# print(fit_4$summary())
# 
# #=============runs QR version==========
# 
# model_5 <- cmdstan_model("regression_QR.stan")
# fit_5 <- model_5$sample(data = stan_data_matrix_w_intercept,
#                       num_chains = 4, output_dir = "output")
# print(paste("ran stan executable: ", model_5$exe_file()))
# print(fit_5$summary())
# 
