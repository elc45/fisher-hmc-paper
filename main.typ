#import "@preview/arkheion:0.1.0": arkheion, arkheion-appendices
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm

#show: set text(size: 12pt)

#show: arkheion.with(
  title: [
    If Only My Posterior Were Normal:\
    Introducing Fisher HMC
  ],
  authors: (
    (
      name: "Adrian Seyboldt",
      email: "adrian.seyboldt@gmail.com",
      orcid: "0000-0002-4239-4541",
      affiliation: "PyMC Labs",
    ),
  ),
  abstract: [
    Hamiltonian Monte Carlo (HMC) is a powerful tool for Bayesian inference, as
    it can explore complex and high-dimensional parameter spaces. But HMC's
    performance is highly sensitive to the geometry of the posterior
    distribution, which is often poorly approximated by traditional mass matrix
    adaptations, especially in cases of non-normal or correlated posteriors. We
    propose Fisher HMC, an adaptive framework that uses the Fisher divergence to
    guide transformations of the parameter space. It generalizes mass matrix
    adaptation from affine functions to arbitrary diffeomorphisms. By aligning
    the score function of the transformed posterior with those of a standard
    normal distribution, our method identifies transformations that adapt to the
    posterior's scale and shape. We develop theoretical foundations efficient
    implementation strategies, and demonstrate significant sampling
    improvements. Our implementation, nutpie, integrates with PyMC and Stan and
    delivers better efficiency compared to existing samplers.
  ],
  keywords: (
    "Bayesian Inference",
    "Hamiltonian Monte Carlo",
    "Mass Matrix Adaptation",
    "Normalizing Flows",
    "Fisher Divergence",
  ),
  date: datetime.today().display(),
)

#let appendix(body) = {
  set heading(numbering: "A", supplement: [Appendix])
  counter(heading).update(0)
  body
}

#set cite(style: "chicago-author-date")
#show link: underline

#show math.equation: set text(weight: 500)
#show math.equation: set block(spacing: 0.65em)

#show raw: set text(font: "Fira Code", size: 10pt)

#show math.equation: set text(
  font: "New Computer Modern Math",
  size: 11pt,
)

#let ex(x) = $op("E", limits: #false)[#x]$
#let var(x, idx: none) = $op("Var"_#idx, limits: #false)[#x]$
#let cov(x) = $op("Cov", limits: #false)[#x]$
#let diag(x) = $op("diag", limits: #false)(#x)$
#let dist(x, y) = $op("dist", limits: #false)(#x, #y)$
#let inner(x, y) = $lr(angle.l #x, #y angle.r)$

= Introduction
<introduction>

Hamiltonian Monte Carlo (HMC) is a powerful Markov Chain Monte Carlo (MCMC)
method widely used in Bayesian inference for sampling from complex posterior
distributions. HMC can explore high-dimensional parameter spaces more
efficiently than traditional MCMC techniques, which makes it popular in
probabilistic programming libraries like Stan and PyMC. However, the performance
of HMC depends critically on the parameterization of the posterior space. Modern
samplers automate a part of these reparametrizations by adapting a "mass matrix"
in the warmup phase of sampling. A common approach in HMC is to estimate a mass 
matrix based on the inverse of the posterior covariance, typically in a diagonalized 
form, to adjust for differences in scale across dimensions. We can think of this as a
reparametrization that simply rescales the parameters such that they have a
posterior variance of one. It is not obvious, however, that this is the best
rescaling that can be done. Morever, even a well-tuned mass
matrix can not do much to help us when sampling from more challenging posterior
distributions such as those with strong correlations in high
dimensions, or with funnel-like pathologies. In most cases, these problems can 
be overcome by careful, manual reparametrization of models, but this requires expertise and time.
For researchers working with multilevel hierarchical models with correlated group-level parameters, manually
rescaling and rotating the parameter space to improve sampling efficiency
requires deep statistical expertise and can be time-consuming. 
In many cases, good reparametrizations are also data-dependent, which makes it
difficult to write a model once, and apply it to a wide range of individual
datasets.

To address these limitations, we propose an adaptive HMC framework that extends
beyond the traditional concept of a mass matrix: instead of just rescaling
variables, we allow for arbitrary diffeomorphisms that dynamically transform the
parameter space. We use the Fisher divergence as a criterion to choose between
different transformations, which allows us to adapt the geometry of the
posterior space in a way that optimizes HMC's efficiency. By aligning the scores
(derivatives of the log-density) of the transformed posterior with those of a
standard normal distribution, we approximate an idealized parameterization that
facilitates efficient sampling.


= Fisher HMC: Motivation and Theory

== Motivation: Example with Normal Posterior
<motivation-gaussian>

HMC is a gradient-based method, meaning that the algorithm computes the 
derivatives of the log posterior density (the scores). While these gradients 
contain significant information about the target density, traditional methods of 
mass matrix adaptation ignore them. To illustrate how useful the scores can be, 
consider a standard normal posterior $N (mu , sigma^2)$ with density $p(x) prop exp(-(x - mu)^2/sigma^2)$.
Let's assume we have two posterior draws $x_1$ and $x_2$, together
with the covector of scores
$
  alpha_i = frac(diff, diff x_i) log p(x_i) = sigma^(-2) (mu - x_i).
$

Based on this information alone, we can directly compute $mu$ and $sigma$ to identify
the exact posterior. Solving for $mu$ and $sigma$, we get

$
  mu = dash(x) + sigma^2 dash(alpha) quad "and" quad sigma^2 = var(x_i)^(1/2) var(alpha_i)^(-1/2),
$
where $dash(x)$ and $dash(alpha)$ are the sample means of $x_i$ and $alpha_i$,
respectively. If we take advantage of the scores, we can compute the exact
posterior and thus an optimal mass matrix with no sample variance, based on just two draws!

This generalizes directly to multivariate normal posteriors $N(mu, Sigma)$, where we can leverage
the elegant fact that the scores are normally distributed with covariance $Sigma^(-1)$.
Assume we have $N + 1$ linearly independent draws $x_i in RR^N$ with scores 
$alpha_i = Sigma^(-1) (x_i - mu)$. The mean of these equations gives us
$mu = dash(x) - Sigma dash(alpha)$. It follows that $Sigma^(-1) X = S$, where
the i-th column of $S$ is $alpha_i - dash(alpha)$, and the i-th column of $X$ is
$x_i - dash(x)$. Finally, we have 
$
  S S^T = cov(alpha_i) = Sigma^(- 1) X X^T Sigma^(-1) = Sigma^(-1) cov(x_i) Sigma^(- 1)
$
and we can recover $Sigma$ as the
geometric mean of the positive symmetric matrices $cov(x_i)$ and
$cov(s_i)^(-1)$:

$
  Sigma = cov(x_i)^(-1/2)(cov(x_i)^(1/2)cov(alpha_i)cov(x_i)^(1/2))^(1/2)cov(x_i)^(-1/2)
$

In this way we can compute the parameters of the normal distribution
exactly. Of course, most posterior distributions of interest are not multivariate normal, and if they were, we 
would not have to run MCMC in the first place. But it is common in Bayesian inference for the posterior
to approximate a normal distribution reasonably well, which suggests that the
scores contain useful information that is ignored in standard methods.

== Transformed HMC
<transformed-hmc>
When we manually reparameterize a model to make HMC more efficient, we try to
find a transformation (or diffeomorphism) of our posterior such that HMC works better. Formally, 
if our posterior $mu$ is defined on a space $M$, we try to find a
diffeomorphism $f: N arrow M$ such that the transformed posterior $f^*mu$ is well-behaved with respect to some property. 
Note that we define the transformation as a function
#emph[from] the transformed space #emph[to] the original posterior, in keeping
consistent with the Normalizing Flow literature. Since the transformation is a
bijection, we can choose any direction we want, as long as we stay consistent
with our choice. $f^* mu$ refers to the pullback of the posterior (which we interpret as a volume
form), i.e. we #emph[pull it back] to the space $N$ along the transformation $f$.
We show later how this is done in practice. If $f$ is an affine transformation, this simplifies to mass-matrix
based HMC, which we discuss now. For example, choosing $f(x) = Sigma^(1/2)x + mu$ corresponds to the
mass matrix $Sigma^(-1)$. This is described in more detail in @neal_mcmc_2012.

HMC efficiency is notoriously dependent on the parametrization, so it's to be expected that
transformed HMC be much more efficient for some choices of $f$ than
for others. It is not, however, obvious what criterion should be used to evaluate 
a particular choice of $f$, in order to guide an automatic learning of the transformation. We
need a loss function that maps the diffeomorphism to a measure of difficulty for
HMC. This is hard to quantify in general, but we can observe that HMC efficiency
largely depends on the trajectory, which in fact does not depend on the
density directly, but rather only on the scores. Therefore, a reasonable
loss function might assess how well the transformed space's #emph[scores] align with
those of our desired transformed posterior. We choose the standard normal distribution 
as the ideal transformed posterior, since we know that HMC is efficient in this case, 
given the nice Gaussian properties such as constant curvature. 
This still leaves open the choice of a specific norm for
comparing the scores of the standard normal with those of the
transformed posterior. But since the standard normal distribution is defined in
terms of an inner product, we already have a well-defined norm on the scores
that allows us to evaluate their difference. This directly motivates the
following definition of the Fisher divergence.

== Fisher divergence

Let $(N, g)$ be a Riemannian manifold with probability volume forms $omega_1$
and $omega_2$. We can define a scalar function $z$ on $N$ by $omega_2 = z
omega_1$, or equivalently we also write this as $z = omega_2 / omega_1$.

We define the Fisher divergence of $omega_1$ and $omega_2$ as

$
  cal(F)_g (omega_1, omega_2) = integral norm(nabla log(z))^2_g d omega_1.
$

Note that $cal(F)$ requires more structure on $N$ than KL-divergence
$integral log(z) d omega_1$, as the norm depends on the metric tensor. Given a second 
(non-Riemannian) manifold $M$ with a probability volume form
$mu$, and a diffeomorphism $f: N arrow M$, we can define the divergence between
$mu$ and $omega_1$ by pulling back $mu$ to $N$, i.e. $cal(F)_g (f^* mu, omega_1)$.

We can also compute this Fisher divergence directly on $M$, by pushing the
metric tensor to $M$:

$
  cal(F)_g (f^* mu, omega_1) = cal(F)_((f^(-1))^*g) (mu, (f^(-1))^* omega_1)
$

In this case, $mu$ is our posterior, $M$ is the space on which it is originally defined, 
and $omega_1$ is the standard normal distribution.

== Affine choices for the diffeomorphism $F$
<specific-choices-for-the-diffeomorphism-f>

We focus on three families of diffeomorphisms $F$, for which derive specific results.

=== Diagonal mass matrix
<diagonal-mass-matrix>

If we choose $F_(sigma , mu) : Y arrow X$ as $x arrow.bar y dot.circle sigma +
mu$, we are effectively doing diagonal mass matrix estimation. In this case, 
the sample Fisher divergence reduces to

$
  hat(F)_(sigma , mu)(f^*Y, Z) = 1 / N sum_i norm(sigma dot.circle alpha_i
  + sigma^(-1) dot.circle (x_i - mu))^2
$

This is a special case of the affine transformation in
#ref(<appendix-proof-affine>) and minimal if $sigma^2 = var(x_i)^(1/2)
var(alpha_i)^(-1/2)$ and $mu = dash(x)_i + sigma^2 dash(s)_i$; the
same result from the solvable case in #ref(<motivation-gaussian>). This solution is very 
computationally inexpensive, and is hence the default in nutpie. Using Welford's algorithm 
to keep online estimates of the draw and score variances during sampling (thereby avoiding the 
need to explicity store scores), the mass matrix is set to a diagonal matrix with the $i$'th entry on 
the diagonal equal to $var(x_i)^(1/2)var(alpha_i)^(-1/2)$.

==== Some theoretical results for normal posteriors
<some-theoretical-results-for-normal-posteriors>

If the posterior is $N (mu , Sigma)$, then the minimizers $hat(mu)$ and
$hat(sigma)$ of $hat(cal(F))$ converge to $mu$ and $exp (1/2 log diag(Sigma) -
1/2 log diag(Sigma^(- 1))$ respectively. This is a direct consequence of
$cov(x_i) arrow Sigma$ and $cov(alpha_i) arrow Sigma^(-1)$.

$hat(cal(F))$ converges to $sum_i lambda_i + lambda_i^(- 1)$, where $lambda_i$
are the generalized eigenvalues of $Sigma$ with respect to $diag(hat(sigma)^2)$,
so large and small eigenvalues are penalized. When we choose $diag(Sigma)$ as
mass matrix, we effectively minimize $sum_i lambda_i$, and only penalize large
eigenvalues. If we choose $diag(bb(E) (alpha alpha^T))$ (as proposed, for
instance, in #cite(<tran_tuning_2024>, form: "prose")) we effectively minimize
$sum lambda_i^(- 1)$ and only penalize small eigenvalues. But based on
theoretical results for multivarite normal posteriors in
#cite(<langmore_condition_2020>, form: "prose"), we know that both large and
small eigenvalues make HMC less efficient. To this effect, we 

We can use the result in (todo ref) to evaluate the different diagonal mass
matrix choices on various gaussian posteriors, with different numbers of
observations. Figure todo shows the resulting condition numbers of the posterior
as seen by the sampler in the transformed space.

=== Full mass matrix
<full-mass-matrix>
We choose $f_(A , mu) (y) = A y + mu$. This corresponds to a mass matrix $M = (A
A^T)^(- 1)$. Because as we will see $hat(cal(F))$ only depends on $A A^T$ and $mu$,
we can restrict $A$ to be symmetric positive definite. We get 
$ 
  hat(cal(F)) [f] = 1/N sum norm(A^T s_i + A^(-1) (x_i - mu))^2
$
which is minimized when $A A^T cov(x_i) A A^T = cov(alpha_i)$ (proof in 
#ref(<appendix-proof-affine>)), and as such corresponds again to our earlier
derivation in #ref(<motivation-gaussian>). If the two covariance matrices are
full rank, we get a unique minimum at the geometric mean of $cov(x_i)$ and
$cov(s_i)$.

=== Diagonal plus low-rank

If the number of dimensions is larger than the number of draws, we can add
regularization terms. And to avoid $O (n^3)$ computation costs, we can project
draws and scores into the span of $x_i$ and $alpha_i$ and compute the regularized
mean in this subspace. If we only store the
components, we can avoid $O (n^2)$ storage, and still do all operations we need
for HMC quickly. To further reduce computational cost, we can ignore eigenvalues
that are close to one with a cutoff parameter $c$. The algorithm is as follows:

#algorithm({
  import algorithmic: *
  Function("Low-Rank-Adapt", args: ("D", "G", "c"), {
    Cmt[Combine bases]
    Assign([$U^D$], FnI[SVD][$D$])
    Assign([$U^G$], FnI[SVD][$G$])
    Assign[$S$][$[U^D  U^G]$]
    State[]
    Cmt[Get jointly-spanned orthonormal basis]
    Assign([$Q$, \_],FnI[QR_thin][$S$])
    Assign[$P^D$][$Q^T D$]
    Assign[$P^G$][$Q^T G$]
    State[]
    Assign[$C^D$][$P^D (P^D)^T + gamma I$]
    Assign[$C^G$][$P^D (P^D)^T + gamma I$]
    State[]
    Assign([$Sigma$],FnI[spdm][$C^D, C^G$])
    State[]
    Assign([$U Lambda U^(-1)$],FnI[eigendecompose][$Sigma$])
    Assign[$U_c$][${U_i: i in {i: lambda_i >= c}}$]
    Assign[$M$][$Q U_c (Lambda_c - 1) U_c^T + I$]
    Return[$M$]
  })
})

By subtracting the identity matrix from the diagonal matrix of eigenvalues, we obtain a 
matrix whose eigenvalues include all those of $Sigma$ which are sufficiently far 
from 1, with the remaining eigenvalues equal to 1. This is then a
"diagonal plus low-rank" matrix.

== Normalizing flows
<normalizing-flows>

Normalizing flows provide a large family of diffeomorphisms that we can use to
transform our posterior. We have rather strong requirements for the flows, however:
we need forward and inverse transformations, and we need to be able to compute the
log determinant of the jacobian of the transformation efficiently. A well studied
family of normalizing flows that provides all of those is RealNVP 
(#cite(<dinh_density_2017>, form: "prose")).
For our experiments, we used the library flowjax that implements RealNVP and
other normalizing flows in jax. I slightly changed the adaptation schema from
the usual nutpie algorithm (described in more detail in the next section), so
that for the first couple of windows only diagonal mass matrix adaptation is
used, and only after 150 draws do we start to fit a normalizing flow.
We then repeatedly run an Adam optimizer on a window of draws, and use the updated
normalizing flow to sample.

Especially for larger models the size of our training data set seems to be very
important, so I included the full trajectory of the HMC sampler as training
data, not just the draws themselves. I think this might be useful to do even if
the amount of training data is not a limiting factor, as we would like that the
transformed posterior matches the normal distribution everywhere the HMC sampler
evaluates it, not just at those points it accepts as draws.
So far I have not tested this systematically, but my experiments suggest that
this can help us to sample a wide range of posterior distributions a lot more
efficiently, or also to sample many distributions that were previously not
possible to sample without extensive manual reparametrizations. It also comes
with a large computational cost however, as the optimization itself can take a
long time. Luckily, it seems to run relatively efficiently on GPUs, so that even
if the logp function itself does not lend itself to evaluation on GPUs, we can
still spend most of the computational time running scalable code. Often, the
number of gradient evaluations that are necessary to sample even complicated
posterior distributions decrease a lot.

code: #link("https://github.com/pymc-devs/nutpie")

= Adaptation Schema
<adaptation-tuning-warmup-of-mass-matrix>

Whether we adapt a mass matrix using the posterior variance as Stan does, or if
we use a bijection based on the Fisher divergence, we
always have the same problem: in order to generate suitable posterior draws we need a good mass 
matrix (or bijection), but to estimate a good mass-matrix, we
need posterior draws. There is a well known way out of this "chicken and egg" conundrum: 
we start sampling with an initial transformation, and collect a number of draws. 
Based on those draws, we estimate a better transformation, and repeat. This adaptation-window approach
has long been used in the major implementations of HMC, and has remained largely
unchanged for a number of years. PyMC, Numpyro, and Blackjax all use the
same details as Stan, with at most minor modifications. There are, however, a couple of 
small changes that improve the efficiency of this schema significantly.

== Choice of initial position
<choice-of-init-point>

Stan draws initial points independently for each chain from the interval $(-2,
2)$ on the unconstrained space. I don't have any clear data to suggest so, but
for some time PyMC has initialized using draws around the prior mode instead,
and it seems to me that this tends to be more robust. This is of course only
possible for model definition languages that allow prior sampling, and would for
instance be difficult to implement in Stan.

== Choice of initial diffeomorphism
<choice-of-initial-mass-matrix>

Stan starts the first adaptation with an identity mass matrix. The very first
HMC trajectories seem to be overall much more reasonable if we use
$M=diag(alpha_0^T alpha_0)$ instead. This also makes the initialization
independent of variable scaling.

== Accelerated Window-Based Adaptation
<accelerated-window-based-adaptation-warmuptuning-scheme>

Stan and other samplers do not run vanilla HMC, but often the No-U-Turn Sampler (NUTS), where 
the Hamiltonian trajectory length is chosen automatically. This can make it very costly to
generate draws if the mass matrix is not adapted well, because in those cases we
often use a large number of HMC steps for each draw (typically up to 1000). Thus 
very early on during sampling, we have a large incentive to use
available information about the posterior as quickly as possible, to avoid these
long trajectories. By default, Stan starts adaptation with a step-size adaptation
window of 75 draws, where the mass matrix is untouched. This is followed by a mass 
matrix adaptation window consisting of a series of "memoryless" intervals of 
increasing length, the first of which (25 draws) still uses the initial mass matrix for sampling. 
These 100 draws before the first mass matrix change can constitute a sizable percentage 
of the total sampling time.

Intuition might suggest that we could just use a tailing window and update at each step based on the
previous $k$ draws. However, this is computationally inefficient (unless the logp function
is very expensive), and is not easily implemented as a streaming estimator
\(see below for more details). Using several overlapping estimation
windows, though, we can compromise between optimal information usage and computational
cost, while still using streaming estimators.

= Implementation in nutpie

nutpie implements:
- New adaptation schema
- Diagonal mass matrix adaptation based on the Fisher divergence
- Low rank mass matrix adaptation as described here
- Explicit transformation with normalizing flows

The core algorithms are implemented in rust, which all array operations
abstracted away, to allow users of the rust API to provide GPU implementations.

It can take PyMC or Stan models.

Stan is used through bridgestan, which compiles C libraries that nutpie can load
dynamically to call the logp function gradient with little overhead.

pymc models can be sampled either through the numba backend of pymc, which
also allows evaluating the density and its gradient with little overhead.
Alternatively, it can use the pymc jax backend. This incures a higher per-call
overhead, but allows evaluating the density on the GPU, which can significantly
speed up sampling for larger models.

nutpie returns sampling traces as arviz datasets, to allow easy posterior analysis
and convergence checks.

= Experimental evaluation of nutpie
<numerical-results>

We run nutpie and cmdstan on posteriordb to compare performance in terms of
effective sample size per gradient evaluation and in terms of effective sample
size per time...

#pagebreak()
#bibliography("FisherHMCPaper.bib", style: "ieee")

#pagebreak()
#show: appendix

= Minimize Fisher divergence for Affine Transformations
<appendix-proof-affine>

Here we prove that $hat(F)$ for $F(y) = A y + mu$ is minimal when $Sigma cov(alpha) Sigma = cov(x)$ and
$mu = dash(x) + Sigma dash(alpha)$, where $Sigma = A A^T$:

Let $G$ be the matrix of scores, with $alpha_i$ as the $i$th column, and similarly let 
$X$ be the draws matrix, consisting of $x_i$ as the $i$'th column,
and $Sigma = A A^T$. The Fisher divergence between some $p$ and $N(0,I_d)$ is 
$
  E_p [norm(nabla log p(x) + X)^2]
$
and then the estimated divergence is 

$
  hat(F) = 1/N norm(G+X)^2
$
Now, for some transformed $y=A^(-1)(x-mu)$, we have

$
hat(F)_y = 1/N norm(A^T G + Y)^2 = 1/N norm(A^T G + A^(-1)(X - mu 1^T))_F^2
$

Differentiating with respect to $mu$, we have:

$
  (d hat(F)) / (d mu) &= -2 / N tr[bold(1)^T (A^T G + A^(-1) (X - mu bold(1)^T))^T A^(-1)]
$

Setting this to zero,

$
  bold(1)^T (A^T G + A^(-1) (X - mu bold(1)^T))^T A^(-1) = 0
$

It follows that

$
  mu^* = dash(x) + Sigma dash(alpha)
$

Differentiating with respect to $A$:

$
  d hat(F) = 2 / N tr[(A^T G + A^(-1) (X - mu bold(1)^T))^T (d A^T G - A^(-1) d A A^(-1) (X - mu bold(1)^T))]
$

Plugging in the result for $mu^*$ and using the cyclic- and transpose-invariance properties of the trace gives us

$
  d hat(F) &= 2 / N tr[(A^T G + A^(-1) (tilde(X) - Sigma dash(alpha) bold(1)^T)) G^T d A] \
  &quad +2 / N tr[A^(-1) (Sigma dash(alpha) bold(1)^T - tilde(X))(A^T G + A^(-1) (tilde(X) - Sigma dash(alpha) bold(1)^T))^T A^(-1) d A ]
$
where $tilde(X) = X - dash(x) bold(1)^T$, the matrix with centered $x_i$ as the
columns. This is zero for all $d A$ iff
$
  0 = (A^T G + A^(-1) (tilde(X) - Sigma dash(alpha) bold(1)^T)) G^T
  + A^(-1) (Sigma dash(alpha) bold(1)^T - tilde(X))(A^T G + A^(-1) (tilde(X) - Sigma dash(alpha) bold(1)^T))^T A^(-1) \
  = A^T tilde(G) G^T + A^(-1) tilde(X) G^T + (A^T dash(alpha) bold(1)^T - A^(-1) tilde(X))(tilde(G)^T + tilde(X)^T Sigma^(-1)),
$

Where similarly $tilde(G) = G - dash(alpha) bold(1)^T$. Because $bold(1)^T tilde(X)^T = bold(1)^T
tilde(G)^T = 0$, this expands to
$
  0 = A^T tilde(G) G^T + A^(-1) tilde(X) G^T - A^(-1) tilde(X) tilde(G)^T - A^(-1)
  tilde(X) tilde(X)^T Sigma^(-1)
$
$
  = (tilde(G) + Sigma^(-1)tilde(X)) G^T - Sigma^(-1) tilde(X) tilde(G)^T - Sigma^(-1) tilde(X) tilde(X)^T Sigma^(-1) \
  = tilde(G) G^T + Sigma^(-1) tilde(X) bold(1) dash(alpha)^T - Sigma^(-1) tilde(X) tilde(X)^T Sigma^(-1) \
  = tilde(G) tilde(G)^T - Sigma^(-1) tilde(X) tilde(X)^T Sigma^(-1)
$

We can explicitly incorporate this transformation into HMC (or variations of it
like NUTS):

```python
def hmc_proposal(rng, x, mu_density, F, step_size, n_leapfrog):
    # Compute the log density and score (differential of the log density)
    # at the initial point.
    logp_x, score_x = score_and_value(mu_density)(x)
    # Compute the log density and score in the transformed space.
    # We will see later how to do this.
    y, score_y, logp_y = f_pullback_score(x, score_x, logp_x)

    # Sample a new velocity for our trajectory.
    # In the transformed space we assume an identity mass matrix,
    # so we don't have to distinguish between momentum and velocity.
    velocity = rng.normal(size=len(x))

    # Several leapfrog steps
    for i in range(n_leapfrog):
        # velocity half-step
        velocity += step_size / 2 * score_y
        # position step
        y += step_size * velocity

        # Transform back and evaluate density
        x = f_transform(y)
        logp_x, score_x = score_and_value(mu_density)(x)
        y, score_y, logp_y = f_pullback_score(x, score_x, logp_x)

        # second velocity half-step
        velocity += step_size / 2 * score_y

    return x
```



== Sobolev divergence

The Fisher divergence uses the information in the scores of our posterior. But it
still ignores some additional information we have. Specifically, for each pair of
points where we evaluate the posterior, we know the ratio of the densities at
those points. If our transformed posterior has density $p(x)$, then we'd like that $p(x) / p(y)
approx q(x) / q(y)$, where $q$ is the standard normal density. (We look only at pairwise density ratios, because we don't
know the normalization factor of our posterior, and thus can't compute the ratios
$p(x) / q(x)$ directly.)

This leads us to the additional loss term

$
  integral integral ((log(p(x)) - log(p(y))) - (log(q(x)) - log(q(y)))^2) p(x) p(y) d x d y \
  = 2var(log(p(x)) - log(q(x)), idx: p(x))
$

We extend the Fisher divergence to a new divergence, the
Sobolev divergence, by adding this second term:

$
  cal(S)_g(omega_1, omega_2) = cal(F)_g(omega_1, omega_2) + integral log(z)^2
  omega_1 - (integral log(z) omega_1)^2 = cal(F)_g(omega_1, omega_2) +
  var(log(z), idx: omega_1)
$

The name is due to the similarity to the Sobolev norm $norm(f)_(H^1)^2$ =
$integral abs(f)^2 + abs(nabla f)^2$.

== Transformations of scores
<pullbacks-of-scores>

In practice, we don't work with the measure $mu$ directly, but with the
densities relative to Lebesgue measures $lambda_N$ and $lambda_M$, so $mu = p
lambda_M$ and $omega = q lambda_N$. The change-of-variable tells us that $f^*
lambda_M = abs(det(d f)) lambda_N$. This allows us to compute the transformed
scores $d log(f^*p / lambda_N)$:

$
  d log(f^*mu / lambda_N) = d log(f^*p / lambda_M (f^*lambda_M)/lambda_N) = f^*
  d log(p) + d log abs(det(d f)) = hat(f)^*(d log(p), 1)
$

where $hat(f): N arrow M times bb(R), x arrow.bar (f(x), log abs(det(f)))$

This allows us to implement the score pullback function in autodiff systems, for
instance in Jax:

```python
def f_and_logdet(y):
    """Compute the transformation F and its log determininant jacobian."""
    ...

def f_inv(x):
    """Compute the inverse of F."""
    ...

def f_pullback_score(x, s_x, logp):
    """Compute the transformed position, score and logp."""
    y = F_inv(x)
    (_, log_det), pullback_fn = jax.vjp(f_and_logdet, y)
    s_y = pullback_fn(s_x, jnp.ones(()))
    return y, s_y, logp + log_det
```

We can further simplify the Fisher divergence if we set $omega$ to a standard
normal distribution, and use the standard euclidean inner product as the metric
tensor:

$
  cal(F)(f^*mu, omega) & = integral norm( nabla log (f^* mu)/omega)^2_g f^* mu\
  & = integral norm(hat(f)^*(d log p, 1) - d log(q))^2_g^(-1) f^* mu\
  & = integral norm(hat(f)^*(d log p_x, 1) + x)^2 f^* mu(x)\
$

Given posterior draws $x_i$ and corresponding scores $alpha_i$ in the
original posterior space $X$ we can approximate this expectation as

$ hat(cal(F)) = 1 / N sum_i norm( hat(f)^*(s_i, 1) + f^(-1) (x_i) )^2 $

Or in code:

```python
def log_loss(f_pullback_scores, draw_data):
    draws, scores, logp_vals = draw_data
    pullback = vectorize(f_pullback_scores)
    draws_y, scores_y, _ = pullback(draws, scores, logp_vals)
    return log((draws_y + scores).sum(0).mean())

def log_sobolev_loss(f_pullback_scores, draw_data):
    draws, scores, logp_vals = draw_data
    pullback = vectorize(f_pullback_scores)
    draws_y, scores_y, logp_y = pullback(draws, scores, logp_vals)

    fisher_loss = (draws_y + scores).sum(0).mean()
    std_normal_logp = - draws_y @ draws_y / 2
    var_loss = (logp_y - std_normal_logp).var()
    return log(fisher_loss + var_loss)
```

Note: Some previous literature (todo ref) proposed to minimize $bb(E) [alpha_x^T
alpha_x]$, which is similar, but does not solve the issue of choosing a well-defined
inner product. But finding a good inner product is the whole point of mass
matrix adaptation. If we pull pack the inner product of the standard normal
distribution to $X$ and use the corresponding inner product on the dual space of
1-forms, we end up with an equivalent definition for the loss function defined
above.
