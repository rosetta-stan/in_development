data {
  int N; //num data points
  int y[N];
  real x[N];
}

transformed data {
  int dummies[N,2];
  for (n in 1:N) {
    if (y[n] == 0) {
      dummies[n,1] = 0;
      dummies[n,2] = 0;
    }
    else if (y[n] == 1) {
      dummies[n,1] = 1;
      dummies[n,2] = 0;
    }
    else {
      dummies[n,1] = 0;
      dummies[n,2] = 1;
    }
  }
}

parameters {
  real alpha[2];
  real beta[2];
}

model {
  alpha ~ normal(-40, 10);
  beta ~ normal(0, 10); 
  for (n in 1:N) {
    if (y[n] == 0) {
      0 ~ bernoulli_logit(alpha[1] + beta[1] * x[n]);
      0 ~ bernoulli_logit(alpha[2] + beta[2] * x[n]);
    }
    else  {
      1 ~ bernoulli_logit((alpha[1] + beta[1] * x[n]) * dummies[n][1] +
                          (alpha[2] + beta[2] * x[n]) * dummies[n][2]);
    }
  }
}
