#import "@preview/arkheion:0.1.0": arkheion, arkheion-appendices

#import "@preview/algo:0.3.6": algo, i, d, comment, code
#import "@preview/cetz:0.3.4"  // load CeTZ

#show: set text(size: 12pt)

#set par(
  first-line-indent: 1em
)

#show: arkheion.with(
  title: [
    If Only My Posterior Were Normal:\
    Fisher-Informed Hamiltonian Monte Carlo
  ],
  authors: (
    (
      name: "Adrian Seyboldt",
      email: "adrian.seyboldt@gmail.com",
      orcid: "0000-0002-4239-4541",
      affiliation: "PyMC Labs",
    ),
    (
      name: "Eliot Carlson",
      email: "eliot.carlson@pymc-labs.com",
      affiliation: "PyMC Labs",
    ),
    (
      name: "Bob Carpenter",
      email: "bcarpenter@flatironinstitute.org",
      affiliation: "Flatiron Institute",
    ),  
  ),
  
  abstract: [
    Although Hamiltonian Monte Carlo (HMC) scales well in dimension $(O(N^("5/4")))$, 
    traditional adaptation methods converge slowly to weak diagonal preconditioners 
    and scale poorly when expanded to dense preconditioners. We
    propose Fisher HMC, an adaptive framework that uses the Fisher divergence to
    guide transformations of the parameter space. By aligning the score function of 
    the transformed posterior with those of a standard
    normal distribution, our method identifies transformations that adapt to the
    posterior's scale and shape. We develop theoretical foundations with efficient
    implementation strategies exhibiting substantial sampling
    improvements over vanilla HMC. Our implementation, nutpie, integrates with PyMC 
    and Stan and delivers better efficiency compared to existing samplers.
  ],
  keywords: (
    "Bayesian inference",
    "Hamiltonian monte carlo",
    "mass matrix adaptation",
    "Fisher divergence",
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
distributions #cite(<betancourt_conceptual_2018>). HMC can explore high-dimensional parameter spaces more
efficiently both in theory and practice than traditional MCMC techniques such as 
Metropolis-Hastings or the Gibbs sampler, making it popular in
probabilistic programming libraries like Stan #cite(<carpenter_stan_2017>) and 
PyMC #cite(<pymc_2015>). However, the performance
of HMC is highly sensitive to the geometry of the posterior, which can be changed through 
model parameterizations. Current state of the art samplers in wide use automate a part 
of these reparametrizations by adapting a "mass matrix" in the warmup phase of sampling. A common approach in HMC 
is to estimate a mass matrix based on the inverse of the posterior covariance, typically 
in a diagonalized form, to adjust for differences in scale across dimensions. We can think of this as a
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
parameter space. Specifically, we find transformations such that when applied 
to our observed scores, the resulting transformed scores most closely align with 
those from a standard normal distribution. We use the Fisher divergence as a criterion 
to optimize from a class of linear transformations, of which we discuss three. In this way, 
we approximate an idealized parameterization facilitates efficient sampling. In the first section, we motivate why the scores 
are useful for for Hamiltonian dynamics. We then present the Fisher divergence as a 
metric by which we can assess the transformations of target distributions, deriving 
closed-form solutions for optimal diffeomorphisms in the affine case, i.e. mass matrices.
Finally, we suggest additional modifications to the adaptation schedule shared by the major 
software implementations, which complement gradient-based adaptation.


= Fisher HMC: Motivation and Theory

== Motivation: Example with Normal Posterior
<motivation-gaussian>

HMC is a gradient-based method, meaning that the algorithm computes the 
derivatives of the log posterior density, or scores. While these gradients 
contain significant information about the target density, traditional methods of 
mass matrix adaptation ignore them. To illustrate how useful the scores can be, 
consider a standard normal posterior with density $p(theta) prop exp(-(theta - mu)^2/sigma^2)$.
Assume we have two posterior draws $theta_1$ and $theta_2$, together
with the covector of scores
$
  alpha_i = frac(diff, diff theta_i) log p(theta_i) = sigma^(-2) (mu - theta_i).
$

Based on this information alone, we can directly compute $mu$ and $sigma$ to identify
the exact posterior. Solving for $mu$ and $sigma$, we get

$
  mu = dash(theta) + sigma^2 dash(alpha) quad "and" quad sigma^2 = var(theta_i)^(1/2) var(alpha_i)^(-1/2),
$
where $dash(theta)$ and $dash(alpha)$ are the sample means of $theta_i$ and $alpha_i$,
respectively. If we take advantage of the scores, we can compute the exact
posterior and thus an optimal mass matrix with no sample variance, based on just two draws. 
This generalizes directly to multivariate normal posteriors $N(mu, Sigma)$, where we can leverage
the fact that the scores are normally distributed with covariance $Sigma^(-1)$.
Assume we have $N + 1$ linearly independent draws $theta_i in RR^N$ with scores 
$alpha_i = Sigma^(-1) (theta_i - mu)$. The mean of these equations gives us
$mu = dash(theta) - Sigma dash(alpha)$. It follows that $Sigma^(-1) theta = S$, where
column $i$ of $S$ is $alpha_i - dash(alpha)$, and column $i$ of $theta$ is
$theta_i - dash(theta)$. Finally, we have 
$
  S S^T = cov(alpha_i) = Sigma^(- 1) theta theta^T Sigma^(-1) = Sigma^(-1) cov(theta_i) Sigma^(- 1)
$
and we can recover $Sigma$ as the
geometric mean of the positive symmetric matrices $cov(theta_i)$ and
$cov(s_i)^(-1)$:
$
  Sigma = cov(theta_i)^(-1/2)(cov(theta_i)^(1/2)cov(alpha_i)cov(theta_i)^(1/2))^(1/2)cov(theta_i)^(-1/2)
$
In this way we can compute the parameters of the normal distribution
exactly. Of course, most posterior distributions of interest are not multivariate normal; moreover if they were, we 
would not have to run MCMC in the first place. But it is common in Bayesian inference for the posterior
to approximate a normal distribution reasonably well, which suggests that the
scores contain useful information that is ignored in standard methods.

== Transformed HMC
<transformed-hmc>
Reparameterizing a model is akin to transforming a posterior such that HMC performs 
better on it. Formally, if our posterior $mu$ is defined on a space $M$, we try to find a
diffeomorphism $f: N arrow M$ such that the transformed posterior $f^*mu$ is well-behaved with respect to some property. 
Note that we define the transformation as a function
#emph[from] the transformed space #emph[to] the original posterior, in keeping
consistent with the normalizing flow literature.#footnote[Since the transformation 
is a bijection, we can choose any direction we want, as long as we stay consistent
with our choice.] $f^* mu$ refers to the pullback of the posterior (which we can 
interpret as a volume form), i.e. we #emph[pull it back] to the space $N$ along the 
transformation $f$. If $f$ is an affine transformation, this simplifies to mass matrix-based 
HMC, wherein choosing $f(x) = Sigma^("1/2")x + mu$ corresponds to the
mass matrix $Sigma^(-1)$, as described in more detail in #cite(<neal2011mcmc>, form: "prose"). In 
standard implementations, $Sigma^(-1)$ appears in full in the leapfrog 
integrator:

#algo(
  line-numbers: false,
  block-align: none,
  title: "leapfrog",
  stroke: none,
  fill: none,
  parameters: ($theta$, $rho$, $L$, $epsilon$, $Sigma$),
 )[
    $theta^(0) <- theta, rho^(0) <- rho$\
    for $i "from" 0 "to" L$:#i\
      $rho^((i+1/2)) <- rho^((i))- epsilon/2 nabla U(theta^((i)))$ #comment[half-step momentum]\
      $theta^((i+1)) <- theta^((i)) + epsilon Sigma^(-1) rho^((i + 1/2))$ #comment[full-step position]\
      $rho^((i+1)) <- rho^((i+ 1/2))- epsilon/2 nabla U(theta^((i+1)))$ #comment[half-step momentum]#d\
    return $(theta^((L)),rho^((L)))$
  ]

In a transformed setting, however, draws in the posterior space $M$ are pulled 
back along $f$ to the more forgiving space $N$. Leapfrog requires computing the log densities 
and scores in the transformed space. In practice, we work with densities $p$ and 
$q$ relative to Lebesgue measures $lambda_N, lambda_M$: $mu = p lambda_M, 
f^*mu = q lambda_N$. The transformed score is $nabla log(f^*mu / lambda_N)$. Note, 
however, that the computation of this transformed score requires a push-forward. 
Using the change-of-variables $f^*lambda_M = abs(det(d f)) lambda_N$, we have
$
  nabla log(f^*mu / lambda_N) = nabla log(f^*p / lambda_M (f^*lambda_M)/lambda_N) = f^*
  nabla log(p) + nabla log abs(det(d f)) = hat(f)^*(nabla log(p), 1)
$
where $hat(f): N arrow M times bb(R), x arrow.bar (f(x), log abs(det(f)))$. On 
$N$, the Hamiltonian is simulated using an identity mass matrix, meaning we no 
longer distinguish between momentum and velocity.

#algo(
  line-numbers: false,
  block-align: none,
  title: "nf-leapfrog",
  stroke: none,
  fill: none,
  parameters: ($theta$, $v$, $L$, $epsilon$, $f$),
 )[
    $theta^(0) <- theta, v^(0) <- v$\
    $y <- f^(-1)(theta^0)$\
    $delta <- nabla log(f^* mu / lambda_N) (y)$ #comment[evaluate score on $N$]\
    for $i "from" 1 "to" L$:#i\
      $v^((i+1/2)) <- v^((i))- epsilon/2 delta$ #comment[half-step velocity]\
      $y <- y + epsilon v^((i + 1/2))$ #comment[full-step position ($Sigma=I$)]\
      \
      $theta^((i)) <- f(y)$ #comment[recover corresponding draw on $M$]\
      $delta <- nabla log(f^* mu / lambda_N) (y)$\
      \
      $v^((i+1)) <- v^((i+ 1/2))- epsilon/2 delta$ #comment[half-step velocity]#d\
    return $theta$
  ]

#algo(
  line-numbers: false,
  block-align: none,
  title: "nf-leapfrog",
  stroke: none,
  fill: none,
  parameters: ($theta$, $v$, $L$, $epsilon$, $f$),
 )[
    $theta^(0) <- theta, v^(0) <- v$\
    $y, delta <-$ #smallcaps("pullback")$(f, theta^0)$\
    for $i "from" 1 "to" L$:#i\
      $v^((i+1/2)) <- v^((i))- epsilon/2 delta$ #comment[half-step velocity]\
      $y <- y + epsilon v^((i + 1/2))$ #comment[full-step position ($Sigma=I$)]\
      \
      $theta^((i)) <- f(y)$ #comment[recover corresponding draw on $M$]\
      $y, delta <-$ #smallcaps("pullback")$(f, theta^((i)))$\
      \
      $v^((i+1)) <- v^((i+ 1/2))- epsilon/2 delta$ #comment[half-step velocity]#d\
    return $theta^((L))$
  ]

#algo(
  line-numbers: false,
  block-align: none,
  title: "pullback",
  stroke: none,
  fill: none,
  parameters: ($f$, $theta$),
 )[
    $y <- f^(-1)(theta)$ #comment[pull back $theta$ to $N$]\
    $delta <- nabla log(f^* mu / lambda_N) (y)$ #comment[evaluate score on $N$]\
    
    return $(y, delta)$
  ]



== Fisher Divergence

HMC efficiency is notoriously dependent on the parametrization, so it is to be expected that
transformed HMC be much more efficient for some choices of $f$ than
for others. It is not, however, obvious what criterion should be used to evaluate 
a particular choice of $f$, in order to guide the learning of a transformation. We
need a loss function that maps the diffeomorphism to a measure of difficulty for
HMC. This is hard to quantify in general, but we can observe that HMC efficiency
largely depends on the trajectory, which in fact does not depend on the
density directly, but rather only on the scores. Therefore, a reasonable
loss function might assess how well the transformed space's #emph[scores] align with
those of our desired transformed posterior. We choose the standard normal distribution 
as the ideal transformed posterior, since we know that HMC is efficient in this case, 
given nice properties such as constant curvature. 
This still leaves open the choice of a specific norm for
comparing the scores of the standard normal with those of the
transformed posterior. But since the standard normal distribution is defined in
terms of an inner product, we already have a well-defined norm on the scores
that allows us to evaluate their difference. This directly motivates the
following definition of the Fisher divergence.

Let $(N, g)$ be a Riemannian manifold with probability volume forms $omega_1$
and $omega_2$. We define the Fisher divergence of $omega_1$ and $omega_2$ as

$
  cal(F)_g (omega_1, omega_2) = integral norm(nabla log(omega_2 / omega_1))^2_g d omega_1.
$

Note that $cal(F)$ requires more structure on $N$ than KL-divergence
$integral log(omega_2 / omega_1) d omega_1$, as the norm depends on the metric 
tensor $g$. Given a second (non-Riemannian) manifold $M$ with a probability volume form
$mu$ and a diffeomorphism $f: N arrow M$, we can define the divergence between
$mu$ and $omega_2$ by pulling back $mu$ to $N$, i.e. $cal(F)_g (f^* mu, omega_2)$. We 
can also compute this Fisher divergence directly on $M$, by pushing forward the
metric tensor:

$
  cal(F)_g (f^* mu, omega_2) = cal(F)_((f^(-1))^*g) (mu, (f^(-1))^* omega_2)
$

In this case, $mu$ is our posterior, $M$ is the space on which it is originally defined, 
and $omega_2$ is the standard normal distribution.

== Affine choices for the diffeomorphism
<diffeomorphism-choices>

We focus on three families of affine diffeomorphisms $F$, for which we derive specific results.

=== Diagonal mass matrix
<diagonal-mass-matrix>

Assuming a diffeomorphism of the form $f_(sigma , mu) : Z arrow X$ as $x arrow.bar z dot.circle sigma +
mu$ is equivalent to using a diagonal preconditioner. Recall that here $Z$ follows a 
standard normal distribution, and $X$ is the target. Given a set of draws $theta_i$ and 
corresponding scores $alpha_i$ from $X$, the aim is to find $sigma, mu$ which minimize the sample Fisher 
divergence between $f_(sigma, mu)^*X$ and $Z$. In this case the sample divergence reduces to

$
  hat(cal(F))(f_(sigma, mu)^*X, N(0,I_d)) = 1 / N sum_(i=1)^N norm(sigma dot.circle alpha_i
  + sigma^(-1) dot.circle (theta_i - mu))^2
$

which is minimal at 
$sigma^* = var(theta_i)^(1/2) var(alpha_i)^(-1/2)$ and $mu^* = dash(theta)_i + sigma^2 dash(s)_i$, 
corresponding to the result in #ref(<motivation-gaussian>). The resulting 
transformation corresponds to a diagonal mass matrix with entry $i$ equal 
to $var(theta_i)^(1/2)var(alpha_i)^(-1/2)$. Since HMC is translation invariant, we 
discard $mu$. The computation of this solution is linear in the posterior dimension and is hence the default in nutpie. Online estimates of the 
draw and score variances are kept during sampling using Welford's algorithm.

If our target density is $N(mu , Sigma)$, then the minimizers $mu^*$ and
$sigma^*$ of $hat(cal(F))$ derived above converge to $mu$ and $exp (1/2 log diag(Sigma) -
1/2 log diag(Sigma^(- 1)))$, respectively. This is a direct consequence of the fact that 
$cov(x_i) arrow Sigma$ and $cov(alpha_i) arrow Sigma^(-1)$. The divergence $hat(cal(F))$ converges to $sum_i lambda_i + lambda_i^(- 1)$, where $lambda_i$
are the generalized eigenvalues of $Sigma$ with respect to $diag(hat(sigma)^2)$.
So large and small eigenvalues are penalized. Choosing $diag(Sigma)$ as our 
mass matrix effectively minimizes $sum_i lambda_i$, only penalizing large
eigenvalues. If we choose $diag(bb(E) (alpha alpha^T))$, as proposed in 
#cite(<tran_tuning_2024>, form: "prose"), we effectively minimize
$sum lambda_i^(- 1)$ and only penalize small eigenvalues. But based on theoretical 
results for multivariate normal posteriors in #cite(<langmore_condition_2020>, form: "prose"), 
we know that both large and small eigenvalues make HMC less efficient. We can use this 
result to evaluate the different diagonal mass
matrix choices on various normal posteriors, with different numbers of
observations. Figure todo shows the resulting condition numbers of the posterior
as seen by the sampler in the transformed space.

=== Full mass matrix
<full-mass-matrix>
The full affine diffeomorphism $f_(A , mu) (y) = A y + mu$ corresponds to a mass matrix 
$M = (A A^T )^(-1)$. The Fisher divergence in this case is

$ 
  hat(cal(F)) [f_(A , mu)] = 1/N sum norm(A^T s_i + A^(-1) (x_i - mu))^2
$

which is minimized when $A A^T cov(x_i) A A^T = cov(alpha_i)$ (proof in 
#ref(<appendix-proof-affine>)), corresponding again to the derivation in 
#ref(<motivation-gaussian>). Because $hat(cal(F))$ only depends on $A A^T$ and $mu$,
we restrict $A$ to be symmetric positive definite such that there is a unique solution 
for $A$. If the two covariance matrices are full rank, we get a unique minimum at 
the geometric mean of $cov(x_i)$ and $cov(s_i)$.

=== Diagonal plus low-rank

Either to save computation in high dimensional settings, or if the number of dimensions is 
larger than the number of available draws at the time of adaptation, we can also approximate 
the full mass matrix with a "diagonal plus low-rank" matrix. This low rank matrix is parametrized 
by a cutoff $c$, determining a critical distance from one at which point we ignore eigenvectors, 
returning a truncated set of eigenvectors $U_c$ and corresponding eigenvalues $Lambda_c$ of $Sigma$. 
In fact, we implement this as the composition of two affine transformations, the first one being the element-wise (diagonal) affine 
transformation defined earlier, and the second a low-rank approximation to the geometric 
mean of the draw and gradient empirical covariance matrices. The corresponding 
normalizing flow is $f = f_(A,mu) compose f_(sigma, mu)$, and the mass matrix is

$
  Sigma = D^(1/2) (Q U_c (Lambda_c - 1) Q U_c^T + I)D^(1/2)
$ 

where $Q$ is an orthonormal basis for the shared subspace, which we'd like to optimize over $(D, U, Lambda)$. To do this, we do a greedy optimization 
where we first apply the optimal element-wise rescaling factor $var(x_i)^(1/2)
var(alpha_i)^(-1/2)$, "pulling back" $mu$ to 
an intermediate space, from which we then optimize a low-rank transformation to the final 
transformed posterior. For this second leg of optimization, we project $x_i$ and $alpha_i$ into 
their joint span, compute the geometric mean in this subspace as in #ref(<full-mass-matrix>), and 
then decompose the resulting $Sigma$. We can avoid $O (n^2)$ storage by keeping only 
$Lambda_c$ and $U_c$, which form the reduced $A$, and the diagonal components from the 
element-wise transformation. The algorithm is as follows:

#algo(
  line-numbers: false,
  block-align: none,
  title: "low-rank-adapt",
  stroke: none,
  fill: none,
  parameters: ($X$, $S$, $c$, $gamma$),
)[
    $X <- (X-dash(X)) dot.circle hat(sigma)_X^(-1) hat(sigma)_S$ #comment[apply diagonal transform]\
    $S <- (S-dash(S)) dot.circle hat(sigma)_X^(-1) hat(sigma)_S$\
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
invariably have the same challenge: in order to generate suitable posterior draws, we need a good mass 
matrix (or bijection), but to estimate a good mass-matrix, we
need posterior draws. There is a well-known way out of this "chicken and egg" conundrum: 
we start sampling with an initial transformation, and collect a number of draws; based 
on those draws, we estimate a better transformation, and repeat. This adaptation-window approach
has long been used in the major implementations of HMC, and has remained largely
unchanged for a number of years. PyMC, Numpyro, and Blackjax all use the
same details as Stan, with at most minor modifications. There are, however, a couple of 
small changes that improve the efficiency of this schema significantly.

Stan's HMC sampler begins warmup using an identity mass matrix. We instead initialize with 
$M=diag(alpha_0^T alpha_0)$, which in expectation is equal to the Fisher information. This 
also makes the initialization independent of variable scaling.

== Accelerated Window-Based Adaptation
<accelerated-window-based-adaptation-warmuptuning-scheme>

Most widely used HMC implementations do not run vanilla HMC, but variants, most notably the 
No-U-Turn Sampler (NUTS) #cite(<hoffman2014no>), where 
the length of the Hamiltonian trajectory is chosen dynamically (see #ref(<appendix-nuts>)). Such schemas can make 
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

Intuition might suggest that we could just use a tailing window and update the matrix at 
each iteration to an estimate based on the previous $k$ draws via Welford's algorithm 
for online variance estimation. However, removing the influence of early draws when they are no longer in the 
window requires rewinding the algorithm, which is unnecessarily inefficient and not 
easily implemented. Using two overlapping estimation windows - a "foreground" and a 
"background" - we can accomplish the goal of using recent information immediately, while 
avoiding the computational cost of pure streaming estimation. At each iteration in the 
warmup phase, the transformation used is taken from the "foreground" estimator (which 
itself is updated at each iteration). The "background" estimator, while using the same 
update logic, maintains a fresher estimate based on only the previous $n$ draws, 
periodically handing off this fresher estimate to the foreground, then resetting itself. 
The foreground estimator then builds on this estimate until recieving a new one. In this 
way, the estimate used for the transformation at any given iteration is informed by at most 
$2n$ draws, where $n$ is the update frequency. This scheme is illustrated 
in #ref(<adapt_figure>).

We split the matrix adaptation in the warmup phase into two regimes, 
the first with a very quick update frequency (10 draws) and the second with a much longer 
one (80 draws).

#algo(
  line-numbers: false,
  block-align: none,
  title: "warmup",
  stroke: none,
  fill: none,
  parameters: ($N$, $N_"early"$, $N_"late"$, $nu_"early": 10$, $nu_"late": 80$)
)[
    $theta_0, alpha_0 med \~ med p(theta)$ #comment[initial draw from prior]\
    $F$ = #smallcaps("MassMatrixEstimator")$()$#comment[foreground estimator]\
    #smallcaps("update-estimates")$(F, theta_0, alpha_0)$\
    $B$ = #smallcaps("MassMatrixEstimator")$()$#comment[background estimator]\
    $epsilon$ = #smallcaps("StepSizeAdapt")$(theta_0, alpha_0)$\
    $II_"init" <- 1$ #comment[indicator for initial mass matrix]\
    \
    for $i$ in 1 to $N$:#i\
      
      $"early" = i < N_"early"$ #comment[indicator for early regime]\
      $"late" = N - i < N_"late"$ #comment[indicator for late regime]\
      $f <-$ #smallcaps("get-transform")$(F)$\
      $theta^((i)), alpha^((i))$ = #smallcaps("hmc-step")$(theta^((i-1)), epsilon, f)$ #comment[simulate Hamiltonian]\
      \
      if #text(weight: "bold")[not] $"early"$:#i\
      #smallcaps("update-estimates")$(F, theta^((i)), alpha^((i)))$#comment[update both windows]\
      #smallcaps("update-estimates")$(B, theta^((i)), alpha^((i)))$#d\
      if $"late"$:#i\
      #smallcaps("update")$(epsilon)$\
      #text(weight: "bold")[continue] #d\
      #smallcaps("update")$(epsilon)$\
      $nu <- nu_"early"$ if $"early"$ else $nu_"late"$ \
      $N_"remain" <- N - i - N_"late"$\
      if $N_"remain" > nu_"late"$ and #smallcaps("n-draws")$(B) > nu$:#i\
        $F <- G$#comment[dump background into foreground]\
        $B <-$ #smallcaps("MassMatrixEstimator")$()$ #comment[reset background estimator]\
        if $II_"init"$:#i\
          #smallcaps("reset")$(epsilon)$\
          $II_"init" <- 0$ #d #d\
    return
  ]

#algo(
  line-numbers: false,
  block-align: none,
  title: "MassMatrixEstimator",
  stroke: none,
  fill: none,
  parameters: (),
 )[
    #text(weight: "bold")[def] #smallcaps("update-estimates")$("self", theta, alpha)$:#i\
        $n <- "self".n$\
        $dash(theta), hat(sigma)_theta^2 <-$ #smallcaps("update")$(n, hat(sigma)_(theta)^2, dash(theta), theta)$#comment[Welford update for draws]\
        $dash(alpha), hat(sigma)_alpha^2 <-$ #smallcaps("update")$(n, hat(sigma)_(alpha)^2, dash(alpha), alpha)$#comment[update gradients]\
        $"self".n <- n + 1$\
        return #d\

    #text(weight: "bold")[def] #smallcaps("current")$()$:#i\
        return $(dash(theta), dash(alpha), hat(sigma)_theta^2, hat(sigma)_alpha^2)$#d\
    #text(weight: "bold")[def] #smallcaps("n-points")$()$:#i\
        return $"self".n$#d
 ]

#algo(
  line-numbers: false,
  block-align: none,
  title: "update",
  stroke: none,
  fill: none,
  parameters: ($n$, $hat(sigma)^2_(n-1)$, $dash(x)_(n-1)$, $x_n$),
 )[
    $dash(x)_n <- dash(x)_(n-1) + 1/n (x_n - dash(x)_(n-1))$\
    $hat(sigma)^2_n <- (1/(n-1)) hat(sigma)^2_(n-1) + (x_n - dash(x)_(n-1))^2$\
    return $(dash(x)_n, hat(sigma)_n^2)$
  ]

#figure(
  image("figures/adapt_figure.png", width: 80%),
  caption: [Example mass matrix adaptation scheme with background (below) and 
  foreground (above) variance estimators, with a switch/flush frequency of 10 
  draws. Labels beneath states $Sigma_{"iter"}$ indicate the draws (and 
  their gradients) on which that estimate is based. The transformation used for 
  the Hamiltonian at each iteration is informed by the estimate stored in the 
  foreground's state.
  ],
) <adapt_figure>

= Implementation in nutpie

The core algorithms presented here are implemented in the rust language, with all 
array operations abstracted away to allow users of the rust API to provide GPU 
implementations. Nutpie accepts both PyMC and Stan models, the latter accessed through 
the bridgestan library, which compiles C libraries that nutpie can load dynamically 
to call the logp function gradient with little overhead. PyMC models can be sampled either through the numba backend, which
also allows evaluating the density and its gradient with little overhead.
Alternatively, it can use the PyMC Jax backend. This incurs a higher per-call
overhead, but allows evaluating the density on the GPU, which can significantly
speed up sampling for larger models. Traces are returned as ArviZ inference data objects, allowing for easy posterior analysis
and convergence checks. Code: #link("https://github.com/pymc-devs/nutpie")

= Experimental evaluation of nutpie
<numerical-results>

We run nutpie and cmdstan on posteriordb to compare performance in terms of
effective sample size per gradient evaluation and in terms of effective sample
size per time...

#pagebreak()
#bibliography("FisherHMCPaper.bib", style: "ieee")

#pagebreak()
#show: appendix

= Minimization of Fisher divergence for affine transformations
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

#pagebreak()

= The No-U-Turn Sampler
<appendix-nuts>


#algo(
  line-numbers: false,
  block-align: none,
  title: "nuts",
  stroke: none,
  fill: none,
  parameters: ($theta: "position"$, $epsilon: "step-size"$, $T: "max tree depth"$),
 )[
    $rho ~ N(0,I_(d times d))$ #comment[refresh momentum]\
    $B ~ "Unif"({0,1}^T)$ #comment[resample Bernoulli process]\
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
    for $i$ from 0 to $T$:#i\
      $tilde(a) <- a + (-1)^(B_i)2^(i-1), tilde(b) <- b + (-1)^(B_i)2^(i-1)$ #comment[tree doubling]\
      $II_("U-Turn") <-$ #smallcaps("u-turn") $(a,b,theta,rho,epsilon)$\
      $II_("Sub-U-Turn") <-$ #smallcaps("sub-u-turn") $(tilde(a), tilde(b), theta, rho, epsilon)$#comment[recursive U-turn checks]\
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

```python
def warmup(num_warmup, num_early, num_late, early_switch_freq, late_switch_freq):
    position, score = draw_from_prior()
    foreground_window = MassMatrixEstimator()
    foreground_window.update(position, score)
    background_window = MassMatrixEstimator()
    step_size_estimator = StepSizeAdapt(position, score)
    first_mass_matrix = True

    for draw in range(num_warmup):
        is_early = draw < num_early
        is_late = num_warmup - draw < num_late

        mass_matrix = foreground_window.current()
        step_size = step_size_estimator.current_warmup()
        (
          accept_stat, accept_stat_sym, position, score,
          diverging, steps_from_init
        ) = hmc_step(mass_matrix, step_size, position, score)

        # Early on we ignore diverging draws that did not move
        # several steps. They probably just used a terrible step size
        ok = (not is_early) or (not diverging) or (steps_from_init > 4)
        if ok:
            foreground_window.update(position, score)
            background_window.update(position, score)

        if is_late:
            step_size_estimator.update(accept_stat_sym)
            continue

        step_size_estimator.update(accept_stat)

        switch_freq = early_switch_freq if is_early else late_switch_freq
        remaining = num_warmup - draw - num_late
        if (remaining > late_switch_freq
            and background_window.num_points() > switch_freq
        ):

            foreground_window = background_window
            background_window = MassMatrixEstimator()
            if first_mass_matrix:
                step_size_estimator.reset()
            first_mass_matrix = False```
