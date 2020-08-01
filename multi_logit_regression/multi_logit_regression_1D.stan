data {
  int K;
  int N;
  int y[N];
  real x[N];
}

parameters {
  real beta[K];
}
model {
  vector[3] value = to_vector([2.1,3.5,5.5]);
//  beta ~ normal(0, 5);
  for (n in 1:N) {
    //real betas[3] = [beta[1]*x[n],beta[2]*x[n],beta[3]*x[n]];
    y[n] ~ categorical_logit(value);
  }
}

