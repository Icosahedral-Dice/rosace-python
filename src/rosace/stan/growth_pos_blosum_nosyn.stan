data {
  int<lower=0> T; // # of time points
  int<lower=0> V; // # of variants
  int<lower=0> P; // # of positions
  int<lower=0> M; // # of mean group
  int<lower=0> B; // # of blosum group
  array[V] int vMAPp;
  array[V] int vMAPm;
  array[V] int vMAPb;
  vector[T] t; // time
  array[V] vector[T] m; // normalized count
  vector[B] blosum_count; // Count of each BLOSUM group (no SYN in this model)
}
transformed data {
  vector[B] diagon = rep_vector(0, B);
  diagon[1] = 1.0;
  vector[B - 1] off_diagonal = rep_vector(0, B - 1);
  real<lower=0> off_diag_ss = 0.;
  for (i in 1 : B - 1) {
    off_diagonal[i] = (-1.0 / (B - 1) - off_diag_ss) / diagon[i];
    off_diag_ss = off_diag_ss + off_diagonal[i] * off_diagonal[i];
    diagon[i + 1] = sqrt(1.0 - off_diag_ss);
  }
  matrix[B, B - 1] nu_multiplier = rep_matrix(0.0, B, B - 1);
  for (i in 1 : B - 1) {
    nu_multiplier[i][i] = diagon[i];
    for (j in i : B - 1) {
      nu_multiplier[j + 1][i] = off_diagonal[i];
    }
  }
  vector[B] w = blosum_count[ : B];
  real count_mean = mean(w);
  for (i in 1 : B) {
    w[i] = count_mean / w[i];
  }
  matrix[B, B] weight_multiplier = diag_matrix(w);
  nu_multiplier = weight_multiplier * nu_multiplier;
}
parameters {
  vector[P] phi; // slope per position
  vector[B - 1] nu_raw;
  vector<lower=0>[P] sigma2;
  vector<lower=0>[M] epsilon2;
  vector[V] eta2; // std_normal for beta
  vector[V] b; // intercept
}
transformed parameters {
  vector[B] nu = nu_multiplier * nu_raw;
  vector[V] beta; // slope per variants
  for (v in 1 : V) {
    beta[v] = phi[vMAPp[v]] + nu[vMAPb[v]] + eta2[v] * sqrt(sigma2[vMAPp[v]]);
  }
}
model {
  phi ~ normal(0, 1);
  nu_raw ~ normal(0, 0.5);
  sigma2 ~ inv_gamma(1, 1);
  eta2 ~ normal(0, 1);
  epsilon2 ~ inv_gamma(1, 1);
  b ~ normal(0, 0.25);
  for (v in 1 : V) {
    m[v] ~ normal(b[v] + beta[v] * t, sqrt(epsilon2[vMAPm[v]]));
  }
}
