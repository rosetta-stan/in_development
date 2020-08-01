data {
  int K; //num outcomes
  int N; //num data points
  int y[N];
  //matrix[N, D] x;
  int dummies[N,2];
  real x[N];
}

parameters {
  real alpha[2];
  real numeric_coeff[2];
  real beta[2];
}

model {
  beta ~ normal(0, 5);
  numeric_coeff ~ normal(0, 5);
  alpha ~ normal(0, 5);
  for (n in 1:N) {
    vector[2] value;
    for (d in 1:2) {
        value[d] = alpha[d] + 
                      beta[d] * dummies[n,1] +
                      beta[d] * dummies[n,2] +
                      numeric_coeff[d] * x[n];
          print("d=",d," alpha[d]=", alpha[d], 
              " beta[d]=", beta[d],
              " dummies[n,1]=", dummies[n,1], 
              " dummies[n,2]=", dummies[n,2], 
              " numeric_coeff[d]=", numeric_coeff[d],
              " x[n]=", x[n],
              " value[d]=", value[d]);
    }
    print("y[n]=",y[n]," value=",value);
    y[n] ~ categorical_logit(value);
  }
}
