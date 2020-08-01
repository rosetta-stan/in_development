data {
  int K;
  int N;
  int D;
  int y[N];
  matrix[N, D] x;
}

transformed data {
  vector[D] zeros = rep_vector(0, D);
  print("x=",x);
}

parameters {
  matrix[D, K - 1] beta_raw;
}

transformed parameters {
  matrix[D, K] beta = append_col(zeros,beta_raw);
}

model {
  matrix[N, K] x_beta = x * beta;
  print("x=",x);
  print("beta=",beta);
  print("x_beta=",x_beta);
  1 ~ bernoulli(.5);
}
generated quantites {
  real beta_virginica = beta_raw[1] 
}
