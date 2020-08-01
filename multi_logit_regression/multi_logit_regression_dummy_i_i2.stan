data {
  int K;
  int N;
  int D;
  int y[N];
  matrix[N, D] x;
}

transformed data {
  vector[D] zeros = rep_vector(0, D);
}

parameters {
  matrix[D, K - 1] beta_raw;
}

transformed parameters {
  matrix[D, K] beta = append_col(zeros,beta_raw);
}

model {
  matrix[N, K] x_beta = x * beta;
 
  for (n in 1:N) {
    y[n] ~ categorical_logit(x_beta[n]');
  }
}

