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
}
parameters {
  vector[P] phi;
  vector<lower=0>[P] sigma2;
  vector<lower=0>[M] epsilon2;
  vector[B] psi;          // BLOSUM group effect
  vector<lower=0>[B] tau2; // BLOSUM group variance
  vector[V] eta2;
  vector[V] eta3;
  vector[V] b;
}
transformed parameters {
  vector[V] beta;
  for (v in 1:V) {
    beta[v] = phi[vMAPp[v]] + psi[vMAPb[v]]
              + eta2[v] * sqrt(sigma2[vMAPp[v]])
              + eta3[v] * sqrt(tau2[vMAPb[v]]);
  }
}
model {
  phi ~ normal(0, 1);
  sigma2 ~ inv_gamma(1, 1);
  psi ~ normal(0, 1);
  tau2 ~ inv_gamma(1, 1);
  eta2 ~ normal(0, 1);
  eta3 ~ normal(0, 1);
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
}
parameters {
  vector[P] phi;
  vector<lower=0>[P] sigma2;
  vector<lower=0>[M] epsilon2;
  vector[B] psi;
  vector<lower=0>[B] tau2;
  vector[V] eta2;
  vector[V] eta3;
  vector[V] b;
  real<lower=0, upper=1> rho;  // activation fraction
}
transformed parameters {
  vector[V] beta;
  for (v in 1:V) {
    beta[v] = rho * (phi[vMAPp[v]] + psi[vMAPb[v]]
              + eta2[v] * sqrt(sigma2[vMAPp[v]])
              + eta3[v] * sqrt(tau2[vMAPb[v]]));
  }
}
model {
  phi ~ normal(0, 1);
  sigma2 ~ inv_gamma(1, 1);
  psi ~ normal(0, 1);
  tau2 ~ inv_gamma(1, 1);
  eta2 ~ normal(0, 1);
  eta3 ~ normal(0, 1);
  epsilon2 ~ inv_gamma(1, 1);
  b ~ normal(0, 0.25);
  rho ~ beta(2, 2);
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
