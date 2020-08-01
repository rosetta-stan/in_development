library("cmdstanr")

N <- 10000
u <- rlogis(N)
x <- rnorm(N,mean=2,sd=1)
ys <- x + u
mu <- c(-Inf,-1,0,1, Inf)
y <- cut(ys, mu)
plot(y,ys)
df <- data.frame(y,x)

library(MASS)
fit <- polr(y  ~ x, method = "logistic", data = df)
summary(fit)
#=========
x = rnorm(500,0)

Beta1 = .1
Beta2 = .5
Denominator= 1+exp(Beta1*x)+exp(Beta2*x)
vProb = cbind(1/Denominator, exp(x*Beta1)/Denominator, exp(x*Beta2)/Denominator )
mChoices = t(apply(vProb, 1, rmultinom, n = 1, size = 1))
dfM = cbind.data.frame(y = apply(mChoices, 1, function(x) which(x==1)), x)
library(nnet)
fit<-(multinom(y ~ x + 0, dfM))
summary(fit)

Beta0 = .1
Beta1 = .4
Beta2 = .8

x0 <- rnorm(100,Beta0,Beta0/.5)
x1 <- rnorm(100,Beta1,Beta1/.5)
x2 <- rnorm(100,Beta2,Beta2/.5)

y <- c(rep(1,100),rep(2,100),rep(3,100))
x <- c(x0,x1,x2)
library("nnet")
data.df <- data.frame(x=x,y=y)
fit3<-multinom(y ~ x,data.df)
summary(fit3)

Denominator2= exp(Beta0*x0)+exp(Beta1*x1)+exp(Beta2*x2)
vProb2 = cbind(exp(x0)/Denominator2, exp(x1)/Denominator2, exp(x2)/Denominator2)

mChoices2 = t(apply(vProb2, 1, rmultinom, n = 1, size = 1))
dfM2 = cbind.data.frame(y = apply(mChoices2, 1, function(x) which(x==1)), x)
library("nnet")
#We want zero intercept hence x+0 hence the foumula of regression as below
fit2<-(multinom(y ~ x + 0, dfM2))
summary(fit2)
#=========
k = 3 # num outcomes
d = 2 # two coeffs, intercept and Beta1
n = 500
x = rnorm(n,0)
Beta1 = 2
#Beta2 = .5
Denominator = 1+exp(Beta1*x) #+exp(Beta2*x)
vProb = cbind(1/Denominator, exp(x*Beta1)/Denominator) #, exp(x*Beta2)/Denominator )

mChoices = t(apply(vProb, 1, rmultinom, n = 1, size = 2))
dfM = cbind.data.frame(y = apply(mChoices, 1, function(x) which(x==1)), x)
library("nnet")
#We want zero intercept hence x+0 hence the foumula of regression as below
fit<-(multinom(y ~ x + 0, dfM))
summary(fit)

stan_data_matrix <- list(N = n, K = k, x = x_matrix, y = y)
model_3 <- cmdstan_model("regression_matrix.stan")
fit_3 <- model_3$sample(data = stan_data_matrix, num_chains = 4,
                        output_dir = "output")
print(paste("ran stan executable: ", model_3$exe_file()))
print(fit_3$summary())



#=======




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


===============
  
n <- 1000
df1 <- data.frame(x1=runif(n,0,100),
                  x2=runif(n,0,100))
df1 <- transform(df1,
                 y=1+ifelse(100 - x1 - x2 + rnorm(n,sd=10) < 0, 0,
                            ifelse(100 - 2*x2 + rnorm(n,sd=10) < 0, 1, 2)),
                 set="Original")


set.seed(123)
n <- 1000
x <- runif(n,0,10)
c_intercept_true <- 2
c1b_true <- .1
c2b_true <- .7
c3b_true <- 1.4
linear_pred_1 <- c1a_true + c1b_true * x
p <- plogis(linear_pred_1)
y <- rbern(n, p)

print(sprintf(paste("simulation parameters are: alpha_true=%.1f",
                    "beta_true=%.1f, n=%d"),
              alpha_true, beta_true, n))

stan_data <- list(N = n, x = x, y = y)



========== from https://www.r-bloggers.com/prototyping-multinomial-logit-with-r/
  
data(iris)

### method 1: nnet package ###
library(cmdstanr)
library(nnet)
mdl1 <- multinom(Species ~ Sepal.Length, data = iris, model = TRUE)
summary(mdl1)

#iris$y <- ifelse(iris$Species == 'setosa', 0, 1)

n <- nrow(iris)
d <- 3
#x_vect <- matrix(ncol=1, nrow=n)
#x_matrix[, 1] <- rep(1, n) # intercept is always 1
x_matrix[, 1] <- iris$Sepal.Length # predictor values

stan_data_matrix <- list(N=n, K=3, D=d, x=iris$Sepal.Length, y=iris$Species)

model <- cmdstan_model("multi_logit_regression_explicit.stan")
fit <- model$sample(data = stan_data_matrix,
                        output_dir = "output")
print(fit$summary())



library(brms)
library(rstan)
rstan_options (auto_write=TRUE)
options (mc.cores=parallel::detectCores ()) # Run on multiple cores

set.seed (3875)

ir <- data.frame (scale (iris[, -5]), Species=iris[, 5])
b2 <- brm (Species ~ Petal.Length + Petal.Width + Sepal.Length + Sepal.Width, data=ir,
             family="categorical", n.chains=3, n.iter=3000, n.warmup=600,
             prior=c(set_prior ("normal (0, 8)")))


===============run independent binary logistic regression========
  
data(iris)

setosa_v_versicolor <- subset(iris,(iris$Species != 'versicolor'))
setosa_v_versicolor[,'y'] <- ifelse(setosa_v_versicolor$Species=='setosa',0,1)

setosa_v_virginica <- subset(iris,(iris$Species != 'virginica'))
setosa_v_virginica[,'y'] <- ifelse(setosa_v_virginica$Species=='setosa',0,1)

data_setosa_v_versicolor <- list(N=length(setosa_v_versicolor$Species),
                                 y=setosa_v_versicolor$y,
                                 x=setosa_v_versicolor$Sepal.Length)

model_setosa_v_versicolor <- cmdstan_model("logistic_regression.stan")
fit_setosa_v_versicolor <- model_setosa_v_versicolor$sample(data = data_setosa_v_versicolor, 
                                                            output_dir = "output")
print(paste("ran stan executable: ", model_setosa_v_versicolor$exe_file()))
print(fit_setosa_v_versicolor$summary())

library(rstanarm)
stan_glm(y ~ Sepal.Length, family=binomial(link="logit"), 
         data=setosa_v_versicolor)
(Intercept)  -32.5    6.5 
Sepal.Length   5.7    1.2 

library(nnet)
mdl1 <- multinom(Species ~ Sepal.Length, data=iris, model = TRUE)
summary(mdl1)


mdl2 <- multinom(y ~ Sepal.Length, data=setosa_v_versicolor, model = TRUE)
summary(mdl2)


mdl3 <- multinom(y ~ Sepal.Length, data=setosa_v_virginica, model = TRUE)
summary(mdl3)


