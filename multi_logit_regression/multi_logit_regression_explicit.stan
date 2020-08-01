data {
  int K; //num categories
  int N;
  int y_1_v_0[N_1_0];
  int y_2_v_0[N_2_0];
  matrix[] x1_v_0;
  matrix[N] x2_v_0;
}

transformed data {
  print("K=",K);
  print("N=",N);
  print("y=",y);
  print("x=",x);
}

parameters {
  vector[K] beta;
  real alpha;
}

model {
  print("K=", K);
  print("N=", N);
  print("D=", D);
  print("y=", y);
  print("x=", x);
  print("alpha=", alpha);
  print("beta=",beta);
  matrix[N, K] x_beta = x * beta;
  print("x_beta=",x_beta);
  //beta[,1] ~ uniform(-100,100);
  //beta[,2] ~ normal(5,3);
  to_vector(beta) ~ uniform(-100,100); //normal(0, 5);
  for (n in 1:N) {
    y[n] ~ categorical_logit(x_beta[n]');
  }
}
