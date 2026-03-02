"""Stan model code strings for rosace growth models.

Four models of increasing complexity:
- growth_nopos: No position-level shrinkage.
- growth_pos: Position-level shrinkage (non-centred parameterisation).
- growth_pos_blosum: + BLOSUM amino acid grouping.
- growth_pos_blosum_act: + rho activation scaling.
"""

GROWTH_NOPOS = r"""
data {
  int<lower=0> T; // # of time points
  int<lower=0> V; // # of variants
  int<lower=0> M; // # of mean group
  array[V] int vMAPm;
  vector[T] t; // time
  array[V] vector[T] m; // normalized count
}
parameters {
  vector[V] beta;
  vector<lower=0>[M] epsilon2;
  vector[V] b; // intercept
}
model {
  beta ~ normal(0, 1);
  epsilon2 ~ inv_gamma(1, 1);
  b ~ normal(0, 0.25);
  for (v in 1:V) {
    m[v] ~ normal(b[v] + beta[v] * t, sqrt(epsilon2[vMAPm[v]]));
  }
}
"""

GROWTH_POS = r"""
data {
  int<lower=0> T;
  int<lower=0> V;
  int<lower=0> P;
  int<lower=0> M;
  array[V] int vMAPp;
  array[V] int vMAPm;
  vector[T] t;
  array[V] vector[T] m;
}
parameters {
  vector[P] phi;
  vector<lower=0>[P] sigma2;
  vector<lower=0>[M] epsilon2;
  vector[V] eta2;
  vector[V] b;
}
transformed parameters {
  vector[V] beta;
  for (v in 1:V) {
    beta[v] = phi[vMAPp[v]] + eta2[v] * sqrt(sigma2[vMAPp[v]]);
  }
}
model {
  phi ~ normal(0, 1);
  sigma2 ~ inv_gamma(1, 1);
  eta2 ~ normal(0, 1);
  epsilon2 ~ inv_gamma(1, 1);
  b ~ normal(0, 0.25);
  for (v in 1:V) {
    m[v] ~ normal(b[v] + beta[v] * t, sqrt(epsilon2[vMAPm[v]]));
  }
}
"""

GROWTH_POS_BLOSUM = r"""
data {
  int<lower=0> T;
  int<lower=0> V;
  int<lower=0> P;
  int<lower=0> M;
  int<lower=0> B;  // # of BLOSUM groups
  array[V] int vMAPp;
  array[V] int vMAPm;
  array[V] int vMAPb;
  vector[T] t;
  array[V] vector[T] m;
  vector[B] blosum_count; // count of variants per BLOSUM group
}
transformed data {
  // Weighted Helmert-like sum-to-zero contrast matrix for B groups
  // Mirrors R growth_pos_blosum_nosyn
  vector[B] diagon = rep_vector(0, B);
  diagon[1] = 1.0;
  vector[B-1] off_diagonal = rep_vector(0, B-1);
  real<lower=0> off_diag_ss = 0.;
  for (i in 1:(B-1)) {
    off_diagonal[i] = (-1.0 / (B - 1) - off_diag_ss) / diagon[i];
    off_diag_ss = off_diag_ss + off_diagonal[i] * off_diagonal[i];
    diagon[i+1] = sqrt(1.0 - off_diag_ss);
  }
  matrix[B, B-1] nu_multiplier = rep_matrix(0.0, B, B-1);
  for (i in 1:(B-1)) {
    nu_multiplier[i][i] = diagon[i];
    for (j in i:(B-1)) {
      nu_multiplier[j+1][i] = off_diagonal[i];
    }
  }
  vector[B] w = blosum_count[:B];
  real count_mean = mean(w);
  for (i in 1:B) {
    w[i] = count_mean / w[i];
  }
  matrix[B, B] weight_multiplier = diag_matrix(w);
  nu_multiplier = weight_multiplier * nu_multiplier;
}
parameters {
  vector[P] phi;
  vector[B-1] nu_raw;
  vector<lower=0>[P] sigma2;
  vector<lower=0>[M] epsilon2;
  vector[V] eta2;
  vector[V] b;
}
transformed parameters {
  vector[B] nu = nu_multiplier * nu_raw;
  vector[V] beta;
  for (v in 1:V) {
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
  for (v in 1:V) {
    m[v] ~ normal(b[v] + beta[v] * t, sqrt(epsilon2[vMAPm[v]]));
  }
}
"""

GROWTH_POS_BLOSUM_ACT = r"""
data {
  int<lower=0> T;
  int<lower=0> V;
  int<lower=0> P;
  int<lower=0> M;
  int<lower=0> B;  // # of BLOSUM groups
  array[V] int vMAPp;
  array[V] int vMAPm;
  array[V] int vMAPb;
  vector[T] t;
  array[V] vector[T] m;
  vector[B] blosum_count; // count of variants per BLOSUM group
}
transformed data {
  // Weighted Helmert-like sum-to-zero contrast matrix for B groups
  // Mirrors R growth_pos_blosum_act_nosyn
  vector[B] diagon = rep_vector(0, B);
  diagon[1] = 1.0;
  vector[B-1] off_diagonal = rep_vector(0, B-1);
  real<lower=0> off_diag_ss = 0.;
  for (i in 1:(B-1)) {
    off_diagonal[i] = (-1.0 / (B - 1) - off_diag_ss) / diagon[i];
    off_diag_ss = off_diag_ss + off_diagonal[i] * off_diagonal[i];
    diagon[i+1] = sqrt(1.0 - off_diag_ss);
  }
  matrix[B, B-1] nu_multiplier = rep_matrix(0.0, B, B-1);
  for (i in 1:(B-1)) {
    nu_multiplier[i][i] = diagon[i];
    for (j in i:(B-1)) {
      nu_multiplier[j+1][i] = off_diagonal[i];
    }
  }
  vector[B] w = blosum_count[:B];
  real count_mean = mean(w);
  for (i in 1:B) {
    w[i] = count_mean / w[i];
  }
  matrix[B, B] weight_multiplier = diag_matrix(w);
  nu_multiplier = weight_multiplier * nu_multiplier;
}
parameters {
  vector[P] phi;
  vector<lower=0>[P] sigma2;
  vector<lower=0,upper=1>[P] rho;  // per-position activation fraction
  vector[B-1] nu_raw;
  vector<lower=0>[M] epsilon2;
  vector[V] eta2;
  vector[V] b;
}
transformed parameters {
  vector[V] beta;
  vector[B] nu = nu_multiplier * nu_raw;
  for (v in 1:V) {
    beta[v] = phi[vMAPp[v]] + rho[vMAPp[v]] * nu[vMAPb[v]] + eta2[v] * sqrt(sigma2[vMAPp[v]]);
  }
}
model {
  phi ~ normal(0, 1);
  nu_raw ~ normal(0, 0.5);
  rho ~ beta(1.5, 1.5);
  sigma2 ~ inv_gamma(1, 1);
  eta2 ~ normal(0, 1);
  epsilon2 ~ inv_gamma(1, 1);
  b ~ normal(0, 0.25);
  for (v in 1:V) {
    m[v] ~ normal(b[v] + beta[v] * t, sqrt(epsilon2[vMAPm[v]]));
  }
}
"""

# Map method names to model strings
STAN_MODELS = {
    "ROSACE0": GROWTH_NOPOS,
    "ROSACE1": GROWTH_POS,
    "ROSACE2": GROWTH_POS_BLOSUM,
    "ROSACE3": GROWTH_POS_BLOSUM_ACT,
}
