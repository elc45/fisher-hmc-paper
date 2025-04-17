#import "@preview/arkheion:0.1.0": arkheion, arkheion-appendices

#import "@preview/algo:0.3.6": algo, i, d, comment, code
#import "@preview/cetz:0.3.4"  // load CeTZ

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
probabilistic programming libraries like Stan #cite(<carpenter_stan_2017>) and PyMC. However, the performance
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
dimensions, or with funnel-like pathologies. For researchers working with multilevel 
hierarchical models with correlated group-level parameters, manually
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
facilitates efficient sampling. In the first section, we motivate why the scores 
are useful for for Hamiltonian dynamics. We then present the Fisher divergence as a 
metric by which we can assess the transformations of target distributions, deriving 
closed-form solutions for optimal diffeomorphisms in the affine case, i.e. mass matrices.
Finally, we suggest additional modifications to the adaptation schedule shared by the major 
software implementations, which complement gradient-based adaptation.


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
find a transformation (or diffeomorphism) of our posterior such that HMC performs better on it. Formally, 
if our posterior $mu$ is defined on a space $M$, we try to find a
diffeomorphism $f: N arrow M$ such that the transformed posterior $f^*mu$ is well-behaved with respect to some property. 
Note that we define the transformation as a function
#emph[from] the transformed space #emph[to] the original posterior, in keeping
consistent with the Normalizing Flow literature. Since the transformation is a
bijection, we can choose any direction we want, as long as we stay consistent
with our choice. $f^* mu$ refers to the pullback of the posterior (which we can interpret as a volume
form), i.e. we #emph[pull it back] to the space $N$ along the transformation $f$. 
If $f$ is an affine transformation, this simplifies to mass matrix-based 
HMC, wherein choosing $f(x) = Sigma^(1/2)x + mu$ corresponds to the
mass matrix $Sigma^(-1)$, as described in more detail in @neal_mcmc_2012. In the present work,
we restrict ourselves to the subset of affine diffeomorphisms, meaning that the transformations 
we derive are implemented in HMC in the same way as previous mass matrix adaptative HMC schemas, 
in the leapfrog integrator.

HMC efficiency is notoriously dependent on the parametrization, so it is to be expected that
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

We focus on three families of affine diffeomorphisms $F$, for which derive specific results.

=== Diagonal mass matrix
<diagonal-mass-matrix>

If we choose $f_(sigma , mu) : Y arrow X$ as $x arrow.bar y dot.circle sigma +
mu$, we are effectively doing diagonal mass matrix estimation. In this case, 
the sample Fisher divergence reduces to

$
  hat(cal(F))_(sigma , mu)(f^*X, N(0,I_d)) = 1 / N sum_i norm(sigma dot.circle alpha_i
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

If the posterior is $N (mu , Sigma)$, then the minimizers $mu^*$ and
$sigma^*$ of $hat(cal(F))$ derived above converge to $mu$ and $exp (1/2 log diag(Sigma) -
1/2 log diag(Sigma^(- 1)))$, respectively. This is a direct consequence of the fact that 
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

Either to save computation in high dimensional settings, or if the number of dimensions is 
larger than the number of available draws at the time of adaptation, we can also approximate 
the full mass matrix with a "diagonal plus low-rank" matrix. This low rank matrix is parametrized 
by a cutoff $c$, determining a critical distance from one at which point we ignore eigenvectors, 
returning a truncated set of eigenvectors $U_c$ and corresponding eigenvalues $Lambda_c$ of $Sigma$. 
In fact, we implement this as the composition of two affine transformations, the first one being the element-wise (diagonal) affine 
transformation defined earlier, and the second a low-rank approximation to the geometric 
mean of the draw and gradient empirical covariance matrices. The corresponding 
normalizing flow is $f = f_(A,mu) compose f_(sigma, mu)$. Note that when we apply this 
transformation via $M$, though, it occurs as

$
  M = D^(1/2) (Q U_c (Lambda_c - 1) Q U_c^T + I)D^(1/2)
$ 

where $Q$ is an orthonormal basis for the shared subspace, which we'd like to optimize over $(D, U, Lambda)$. To do this, we do a greedy optimization 
where we first apply the optimal element-wise rescaling factor $sqrt(x_i / alpha_i)$, "pulling back" $mu$ to 
an intermediate space, from which we then optimize a low-rank transformation to the final 
transformed posterior. For this second leg of optimization, we project $x_i$ and $alpha_i$ into 
their joint span, compute the geometric mean in this subspace as in #ref(<full-mass-matrix>), and 
then decompose the resulting $Sigma$. We avoid $O (n^2)$ storage by keeping only the eigenvectors and eigenvalues, 
which are all that is needed for the HMC steps. To see this, note that 
the mass matrix is only needed in the leapfrog integrator (see algorithm), where we take 
$M^(-1) rho$, for the momentum vector $rho$. Since

$
  M^(-1) rho = rho - U U^T + U Lambda^(-1) U^T rho
$

this can be done with only $U$ and $Lambda^(-1)$. And 
since we first do the element-wise transformation, we store the diagonal components 
from this as well. The algorithm is as follows:

#algo(
  line-numbers: false,
  block-align: none,
  title: "low-rank-adapt",
  stroke: none,
  fill: none,
  parameters: ($X$, $S$, $c$, $gamma$),
)[
    $X <- (X-dash(X)) dot.circle hat(sigma)_X^(-1) * hat(sigma)_S$ #comment[apply diagonal transform]\
    $S <- (S-dash(S)) dot.circle hat(sigma)_X^(-1) * hat(sigma)_S$\
    \
    $U^X <-$ #smallcaps("svd")$(X), U^S <-$ #smallcaps("svd")$(S)$\
    \
    $Q, \_\_ <-$ #smallcaps("qr-thin")$([U^X space U^S\])$ #comment[Get jointly-spanned orthonormal basis]\
    $P^X <- Q^T X, P^S <- Q^T S$ #comment[Project onto shared subspace]\
    \
    $C^X <- P^X (P^X)^T + gamma I$ #comment[Get empirical covariances, regularize]\
    $C^S <- P^S (P^S)^T + gamma I$\
    \
    $Sigma <-$ #smallcaps("spdm")$(C^X, C^S)$ #comment[Solve $Sigma C^(S) Sigma = C^(X) "for" Sigma$]\
    \
    $U Lambda U^(-1) <-$ #smallcaps("eigendecompose")$(Sigma)$ #comment[Extract eigenvalues to subset]\
    \
    $U_c <- {U_i: i in {i: lambda_i >= c "or" <= 1/c}}$\
    $Lambda_c <- {lambda_i: i in {i: lambda_i >= c "or" <= 1/c}}$ #comment[Full matrix $Sigma = Q U_c (Lambda_c - 1) Q U_c^T + I$]\
    \
    return $Q U_c, Lambda_c$
  ]

= Adaptation Schema
<adaptation-tuning-warmup-of-mass-matrix>

Whether we adapt a mass matrix using the posterior variance as Stan does, or if
we use a bijection based on the Fisher divergence, we
inevitably have the same problem: in order to generate suitable posterior draws, we need a good mass 
matrix (or bijection), but to estimate a good mass-matrix, we
need posterior draws. There is a well-known way out of this "chicken and egg" conundrum: 
we start sampling with an initial transformation, and collect a number of draws; based 
on those draws, we estimate a better transformation, and repeat. This adaptation-window approach
has long been used in the major implementations of HMC, and has remained largely
unchanged for a number of years. PyMC, Numpyro, and Blackjax all use the
same details as Stan, with at most minor modifications. There are, however, a couple of 
small changes that improve the efficiency of this schema significantly.

== Choice of initial diffeomorphism
<choice-of-initial-mass-matrix>

Stan's HMC sampler begins warmup using an identity mass matrix. We instead initialize with 
$M=diag(alpha_0^T alpha_0)$, which in expectation is equal to the Fisher information. This 
also makes the initialization independent of variable scaling.

== Accelerated Window-Based Adaptation
<accelerated-window-based-adaptation-warmuptuning-scheme>

Most widely used HMC implementations do not run vanilla HMC, but variants, most notably the 
No-U-Turn Sampler (NUTS) #cite(<hoffman_nuts_2011>), where 
the length of the Hamiltonian trajectory is chosen dynamically. Such schemas can make 
it extremely costly to generate draws with a poor mass matrix, because in these cases the 
algorithm can take a huge number of HMC steps for each draw (typically up to 1000). Thus 
very early on during sampling, we have a big incentive to use
available information about the posterior as quickly as possible, to avoid these
scenarios. By default, Stan starts adaptation with a step-size adaptation
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
cost, while still using streaming estimators. We split the warmup into two adaptation regimes, 
the first with a very quick update frequency (10 draws) and the second with a much longer 
one (80 draws).

#algo(
  line-numbers: false,
  block-align: none,
  title: "warmup",
  stroke: none,
  fill: none,
  parameters: ($N$, $N_e$, $N_l$, $nu_e: 10$, $nu_l: 80$)
)[
    $theta_0, alpha_0 med \~ med p(theta)$ #comment[initial draw from prior]\
    $F$ = #smallcaps("MassMatrixEstimator")$()$\
    $F$ = #smallcaps("update")$(F, theta_0, alpha_0)$\
    $B$ = #smallcaps("MassMatrixEstimator")$()$\
    step_size_estimator = #smallcaps("StepSizeAdapt")$(theta_0, alpha_0)$\
    first_mass_matrix = 1\
    \
    for $i$ in 1 to $N$:#i\
      
      $e = i < N_e$ #comment[indicator for early regime]\
      $l = N - i < N_l$ #comment[indicator for late regime]\
      $M <- F$\
      $theta, rho$ = #smallcaps("hmc_step")$(M, epsilon, theta, alpha)$ #comment[simulate Hamiltonian]\
      \
      if $(1-e) or ("steps_from_init" > 4)$:#i\
      $F$ = #smallcaps("update")$(F, theta, alpha)$\
      $B$ = #smallcaps("update")$(B, theta, alpha)$#d\
      if $l$:#i\
      step_size = step_size_estimator.current_warmup()\
      continue #d\
      $nu <- nu_e$ if $e$ else $nu_l$ \
      $r <- N - i - N_l$\
      if $r > nu_l$ and #smallcaps("NumPoints")$(B) > nu$:#i\
        $F <- G$\
        $B <-$ #smallcaps("MassMatrixEstimator")$()$\
        if first mass matrix:#i\
          step_size_estimator.reset()\
          first_mass_matrix = 0 #d #d\
    return
  ]

#algo(
  line-numbers: false,
  block-align: none,
  title: "Update",
  stroke: none,
  fill: none,
  parameters: ($hat(sigma)^2_(n-1)$, $x_(n-1)$, $x_n$),
 )[
    $dash(x)_n <- dash(x)_(n-1) + 1/n (x_n - dash(x)_(n-1))$\
    $hat(sigma)^2_n <- (1/(n-1)) hat(sigma)^2_(n-1) + (x_n - dash(x)_(n-1))^2$\
    return $sigma^2$
  ]


= Implementation in nutpie

The core algorithms are implemented in rust, which all array operations
abstracted away, to allow users of the rust API to provide GPU implementations.

It can take PyMC or Stan models.

Stan is used through bridgestan, which compiles C libraries that nutpie can load
dynamically to call the logp function gradient with little overhead.

PyMC models can be sampled either through the numba backend, which
also allows evaluating the density and its gradient with little overhead.
Alternatively, it can use the pymc jax backend. This incurs a higher per-call
overhead, but allows evaluating the density on the GPU, which can significantly
speed up sampling for larger models.

nutpie returns sampling traces as ArviZ datasets, to allow easy posterior analysis
and convergence checks. Code: #link("https://github.com/pymc-devs/nutpie")

= Experimental evaluation of nutpie
<numerical-results>

We run nutpie and cmdstan on posteriordb to compare performance in terms of
effective sample size per gradient evaluation and in terms of effective sample
size per time...

#algo(
  line-numbers: false,
  block-align: none,
  title: "leapfrog",
  stroke: none,
  fill: none,
  parameters: ($theta$, $rho$, $L$, $epsilon$, $M$),
 )[
    $theta^(0) <- theta, rho^(0) <- rho$\
    for $i "from" 0 "to" L$:#i\
      $rho^((i+1/2)) <- rho^((i))- epsilon/2 nabla U(theta^((i)))$ #comment[half-step momentum]\
      $theta^((i+1)) <- theta^((i)) + epsilon M^(-1) rho^((i + 1/2))$ #comment[full-step position]\
      $rho^((i+1)) <- rho^((i+ 1/2))- epsilon/2 nabla U(theta^((i+1)))$ #comment[half-step momentum]#d\
    return $(theta^((L)),rho^((L)))$
  ]

#algo(
  line-numbers: false,
  block-align: none,
  title: "nuts",
  stroke: none,
  fill: none,
  parameters: ($theta$, $h$, $epsilon$, $M$),
 )[
    $rho ~ N(0,I_(d times d))$ #comment[refresh momentum]\
    $B ~ "Unif"({0,1}^M)$ #comment[resample Bernoulli process]\
    $(a,b,\_) <-$ #smallcaps("orbit-select") $(theta, rho, B, epsilon)$\
    $(theta^*,\_,\_) <-$ #smallcaps("index-select") $(theta, rho, a, b, epsilon)$\
    return $theta^*$
  ]

#algo(
  line-numbers: false,
  block-align: none,
  title: "orbit-select",
  stroke: none,
  fill: none,
  parameters: ($theta$, $rho$, $B$, $epsilon$),
 )[
    $a,b <- 0$\
    for $i$ from 0 to len($B$):#i\
      $tilde(a) <- a + (-1)^(B_i)2^(i-1), tilde(b) <- b + (-1)^(B_i)2^(i-1)$ #comment[tree doubling]\
      $II_("U-Turn") <-$ #smallcaps("u-turn") $(a,b,theta,rho,epsilon)$\
      $II_("Sub-U-Turn") <-$ #smallcaps("sub-u-turn") $(tilde(a), tilde(b), theta, rho, epsilon)$\
      if $max(II_("U-Turn"),II_("Sub-U-Turn"))=0$:#i\
        $a <- min(a,tilde(a)), b <- max(b,tilde(b))$#d\
      else:#i\
        #text(weight: "bold")[break] #d #d\
    return $a,b,nabla H$
  ]

#algo(
  line-numbers: false,
  block-align: none,
  title: "index-select",
  stroke: none,
  fill: none,
  parameters: ($theta$, $rho$, $a$, $b$, $epsilon$),
 )[
    $a,b <- 0$\
    for $i$ from 0 to len($B$):#i\
      $tilde(a) <- a + (-1)^(B_i)2^(i-1), tilde(b) <- b + (-1)^(B_i)2^(i-1)$\
      $II_("U-Turn") <-$ #smallcaps("u-turn") $(a,b,theta,rho,epsilon)$\
      $II_("Sub-U-Turn") <-$ #smallcaps("sub-u-turn") $(tilde(a), tilde(b), theta, rho, epsilon)$\
      if $max(II_("U-Turn"),II_("Sub-U-Turn"))=0$:#i\
        $a <- min(a,tilde(a)), b <- max(b,tilde(b))$#d\
      else:#i\
        #text(weight: "bold")[break] #d #d\
    return $a,b,nabla H$
  ]

#algo(
  line-numbers: false,
  block-align: none,
  title: "u-turn",
  stroke: none,
  fill: none,
  parameters: ($theta$, $rho$, $a$, $b$, $epsilon$),
 )[
    $theta^(-), rho^(-), H^(+), H^(-) <-$ #smallcaps("leapfrog")$(theta, rho, epsilon alpha, epsilon)$\
    $theta^(+), rho^(+), tilde(H)^(+), tilde(H)^(-) <-$ #smallcaps("leapfrog")$(theta, rho, epsilon beta, epsilon)$\
    $H^(+) = max(tilde(H)^(+), H^(+)), H^(-) = min(tilde(H)^(-), H^(-))$\
    $II_("U-Turn") = rho^(+) dot (theta^(+) - theta^(-)) < 0$ or $rho^(-) dot (theta^(+) - theta^(-)) < 0$\
    return $II_("U-Turn"), H^(+), H^(-)$
  ]

#algo(
  line-numbers: false,
  block-align: none,
  title: "sub-u-turn",
  stroke: none,
  fill: none,
  parameters: ($theta$, $rho$, $a$, $b$, $epsilon$),
 )[
    if $a=b$:#i\
      return 0 #d\
    $m <- floor((a+b) / 2)$\
    $"full"$ = #smallcaps("u-turn")$(a,b,theta,rho,epsilon)$\
    $"left"$ = #smallcaps("sub-u-turn")$(a,m,theta,rho,epsilon)$\
    $"right"$ = #smallcaps("sub-u-turn")$(m+1,b,theta,rho,epsilon)$\
    return $max("left","right","full")$
  ]

#figure(
  // CeTZ drawing canvas
  cetz.canvas({
    // pull in the drawing primitives & the brace decoration
    import cetz.draw: *
    import cetz.decorations: brace

    // ── geometry ───────────────────────────────────────────────
    // end‑points
    circle((0,0), radius: 2pt, fill: black, stroke: none)
    circle((2,0), radius: 2pt, fill: black, stroke: none)

    // solid segment a–b
    line((0,0), (2,0))

    // ── labels ─────────────────────────────────────────────────
    content((0,-0.25), [$a$], anchor: "north")
    content((2,-0.25), [$b$], anchor: "north")


    circle((0,-2), radius: 2pt, fill: black, stroke: none)
    circle((2,-2), radius: 2pt, fill: black, stroke: none)
    circle((6,-2), radius: 2pt, fill: black, stroke: none)

    // solid segment a–b
    line((0,-2), (2,-2))

    // dotted segment b–c
    line(stroke: (dash: "dotted"), (2,-2), (6,-2))
    line((1,-2.25), (1,-1.75))
    line((4,-2.25), (4,-1.75))

    // brace under a–b (opens upward)
    brace((1,-2.75), (4,-2.75), amplitude: .3, flip: true)
    brace((2,-1.75), (6,-1.75), amplitude: .3)

    content((0,-2.25), [$a=tilde(a)$], anchor: "north")
    content((2,-2.25), [$b$], anchor: "north")
    content((6,-2.25), [$tilde(b)$], anchor: "north")

  }),
  caption: [Modified NUTS orbit checks]
)

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
and as such the estimated divergence is 

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