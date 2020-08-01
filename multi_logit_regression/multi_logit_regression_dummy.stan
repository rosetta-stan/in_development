data {
  int K;
  int N;
  int D;
  int y[N];
  matrix[N, D] x;
}

transformed data {
  row_vector[D] zeros = rep_row_vector(0, D);
}

parameters {
  matrix[D, K-1] b_r;
}

transformed parameters {
  matrix[D,K] beta = append_col(zeros',b_r);
}

model {
  matrix[N, K] x_beta = x * beta;
  //to_vector(beta) ~ normal(0,10);
  for (n in 1:N) {
    #print("x[n]=",x[n]);
    #print("beta=",beta);
    #print("x_beta[n]=",x_beta[n]);
    y[n] ~ categorical_logit(x_beta[n]');
  }
}


