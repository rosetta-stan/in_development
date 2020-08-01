data {
  int K;
  int N;
  int D;
  int y[N];
  matrix[N,D] x;
}

parameters {
  vector[K - 1] beta_raw;
  vector[K] alpha;
}

transformed parameters {
  vector[K] beta = append_row(beta_raw, -sum(beta_raw));
}

model {
  matrix[N, K] x_beta = x * beta;
 
  for (n in 1:N) {
    y[n] ~ categorical_logit(x_beta[n]');
  }
}

