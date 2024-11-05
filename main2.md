# Introduction

## Hamiltonian Markov Chain Monte Carlo

The performance of Hamiltonian MCMC samplers like PyMC or Stan depends
critically on the choice of mass matrix. The mass matrix represents a choice of
inner product in the parameter space.

(I talk about HMC, but everything applies to variations of this, like NUTS or
... todo refs)

The most common choice is to use the inverse of the posterior covariance as
mass matrix. Most commonly, the mass matrix is constrained to be a diagonal
matrix.

There are previous proposals to use the covariance of the fisher scores
instead, or to minimize a KL-divergence between a multivariate normal
distribution and the posterior, and to use the resulting precision matrix of
that normal distribution.

## Current challenges in HMC

- Slow warmup phase (chicken-egg problem). (mention long trajectories in early
phase)
- Bad performance with correlated posteriors. (related to eigenvalues as shown
later. Especially if datasets get large)
- Bad geometry in posterior (funnels etc)

## Motivation: Example with independent gaussian posterior

\label{motivation:gaussian}

When we run HMC, we compute the derivatives of the posterior log density (the
fisher scores). They contain a lot of information about the posterior, but when
we compute the mass matrix, we ignore this information. 

To illustrate how useful the fisher scores can be, let's start very simple, and
assume our posterior is a one dimensional normal distribution $N(\mu, \sigma^2)$
with density $d\mu$, and let's assume we have two posterior draws $x_1$ and
$x_2$, together with the fisher scores $s_i = \frac{\partial}{\partial x_i} \log
\pi(x_i) = \sigma^{-2}(\mu - x_i)$. Based on this information we can directly
compute $\mu$ and $\sigma$, and so identify the exact posterior, by solving for
$\mu$ and $\sigma$. We get $\mu = \frac{x_1 + x_2}{2} + \sigma^2\frac{s_1 +
s_2}{2} = \bar{x} + \sigma^2 \bar{s}$ and $\sigma^2 =
\sqrt{\frac{\text{Var}(x_i)}{\text{Var}(s_i)}}$, where $\bar{x}$ and $\bar{s}$
are the sample means of $x_i$ and $s_i$ respectively. So we can compute an ideal
mass matrix with no sample variance based on just two points!

This generalizes directly to multivariate normal posteriors $N(\mu, \Sigma)$.
Let's assume we have $N+1$ linearly independent draws $x_i$ as columns in the
matrix $X$, and the fisher scores $s_i = \Sigma^{-1}(x_i - \mu)$ or $S =
\Sigma^{-1}(X - \mu (1, 1, \dots))$. The mean of these equations gives us $\mu =
\bar{x} - \Sigma\bar{s}$. It follows that $S=\Sigma^{-1}X$, where the i-th
column of $S$ is $s_i - \bar{s}$, and the i-th column of $X$ is $x_i - \bar{x}$.
And finally we get $SS^T = \text{Cov}[s_i] = \Sigma^{-1}XX^T\Sigma^{-1} =
\Sigma^{-1}\text{Cov}[x_i]\Sigma^{-1}$. We can recognize $\Sigma$ as the
geometric mean of the positive symmetric matrices $\text{Cov}[x_i]$ and
$\text{Cov}[s_i]^{-1}$, so $\Sigma = \exp(\log(\text{Cov}(x_i))/2 -
\log(\text{Cov}(s_i)) / 2)$ or some expression with lots of matrix square roots
(todo).

Given the fisher scores, we can compute the parameters of a normal distribution
exactly. Most posterior distributions are not multivariate normal distributions,
or we would not have to run MCMC in the first place. It is quite common that
they approximate normal distributions reasonably well, so this should indicate
that the fisher scores contain useful information we are currently neglecting.

## Pullbacks of fisher scores

Let $\mu$ be a probability measure on some space $X$ with density $d\mu(x)$, and
$F: Y \to X$ a diffeomorphism. (Note that we define the diffeomorphism as going
from the transformed space to the original space $X$, not the other way round).

The pullback of $\mu$ defines a measure $F^*\mu$ on $Y$ with density
$d(F^*\mu)(y) = d\mu(F(y)) \lvert \text{det}(\frac{\partial}{\partial y}F(y))$,
where we use the usual change-of-variable formula.

We define $F^{\hat{*}}s_x$ for a fisher score $s_x$ as the fisher score of the transformed distribution on $Y$, so
$$
\begin{aligned}
F^{\hat{*}}s_x &= \frac{\partial}{\partial y} \log d(F^*\mu)(y)\mid_{y = F^{-1}(x)}
\end{aligned}
$$

We can simplify this a bit by defining $\hat{F}\colon Y\to X\times \mathbb{R}$,
where $\hat{F}(y) = (F(y), \text{log det} \frac{\partial F}{\partial y})$. Given
a fisher score $s_x$ on the space $X$, we can show that

$$
\begin{aligned}
F^{\hat{*}}s_x = \hat{F}^*(s_x, 1)
\end{aligned}
$$

This allows us to implement the fisher score pullback using autodiff systems,
for instance in Jax (although there are more efficient implementations in
special cases):

```python
def F_and_logdet(y):
    """Compute the transformation F and its log determininant jacobian."""
    ...

def F_inv(x):
    """Compute the inverse of F."""
    ...

def F_pullback_fisher_score(x, s_x, logp):
    y = F_inv(x)
    (_, logdet), pullback_fn = jax.vjp(F_and_logdet, y)
    s_y = pullback_fn(s_x)
    return y, s_y, logp + logdet  # TODO or - logdet?
```

## Transformed HMC

Given a posterior $\mu$ on $X$ and a diffeomorphism $F\colon Y\to X$, we can run
HMC on the transformed space $Y$:

```python
def hmc_step(rng, x, d_mu, F, step_size, n_leapfrog):
    logp_x, s_x = gradient_and_value(d_mu)(x)
    logp_y, s_y = F.inverse.hat(logp, s_x)

    velocity = rng.normal(size=len(x))
    for i in range(n_leapfrog):
        ...
```

If $F$ is an affine transformation, this simplifies to the usual mass-matrix
based HMC. For instance, if we choose $F(x) = \sigma \odot x$, this corresponds
to the mass matrix $\text{diag}(\sigma^{-2})$.

HMC efficiency is famously dependent on the parametrization, so we know that
this is much more or less efficient for some choices of $F$ than for others. It
is however not obvious what criterion we could use to decide how good a
particular choice of $F$ is. We need a loss function $D$ that maps the
diffeomorphism to a measure of difficulty for HMC.

This hard to quantify in general, but we can notice that the efficiency of HMC
largely depends on the trajectory, and this trajectory does not depend on the
density directly, but only the fisher scores.  We also know that HMC is
efficient if the posterior is a standard normal distribution. So a reasonable
loss function can ask how different the fisher scores on the transformed space
are from the fisher scores of a standard normal distribution.  If they match
well, we will use the same trajectory we would use for a standard normal
distribution, which we know to be a good trajectory. And because the standard
normal distribution is defined in terms of an inner product, we already have a
well-defined norm on the fisher scores that we can use to evaluate their
difference. So we define

$$
\begin{aligned}
D[F] &= \int \lVert \frac{\partial}{\partial y} \log d(F^*\mu)(y) -
\frac{\partial}{\partial y} \log N(y\mid 0, 1) \rVert^2 d(F^*\mu)(y) \\
&= \int \lVert \frac{\partial}{\partial y} \log d(F^*\mu)(y) + y \rVert^2 d(F^*\mu)(y) \\
&= \int \lVert F^{\hat{*}} \frac{\partial}{\partial x} \log (d\mu(x)) +
y\rVert^2d(F^*\mu)(y)
\end{aligned}
$$

Given posterior draws $x_i$ and corresponding fisher scores $s_i$ we can
approximate this expectation as

$$
\begin{aligned}% \label{eq:finite-fisher-div}
\hat{D}_F = \frac{1}{N} \sum_i \lVert F^{\hat{*}}s_i + F^{-1}(x_i) \rVert^2
\end{aligned}
$$

Or in code:

```python
def log_loss(F_pullback_fisher_scores, draws, fisher_scores, logp_vals):
    pullback = vectorize(F_pullback_fisher_scores)
    draws_y, fisher_scores_y, _ = pullback(draws, fisher_scores, logp_vals)
    return log((draws_y + fisher_scores).sum(0).mean())
```

Note: Some previous literature (todo ref) proposed to minimize
$\mathbb{E}[s_x^Ts_x]$, which is similar, but does not solve the issue of
choosing a well-defined inner product. But finding a good inner product is the
whole point of mass matrix adaptation. If we pull pack the inner product of the
standard normal distribution to $X$ and use the corresponding inner product on
the dual space of 1-forms, we end up with an equivalent definition for the loss
function defined above.

## Specific choices for the diffeomorphism $F$

Depending on the restrictions we choose for our diffeomorphism $F$, we can get
more specific results.

### Diagonal mass matrix

If we choose $F_{\sigma, \mu}: Y\to X$ as $x\mapsto y\odot \sigma + \mu$, we get
the same effect as diagonal mass matrix estimation.

In this case, the fisher divergence \autoref{eq:finite-fisher-div} reduces to

todo, check!

$$
D_{\sigma, \mu} = \frac{1}{N}\sum_i
\lVert \sigma \odot s_i + \sigma - (x_i - \mu) \odot \sigma^{-1} \rVert^2
$$

It is not hard to see that this is minimal if $\sigma^2 =
\sqrt{\frac{\text{Var}(x1)}{\text{Var}(s_i)}}$ and $\mu = \bar{x_i} + \sigma^2
\bar{s_i}$. So we recover the same result we got in
\label{section:motivation-gaussian}. From those results we also know that this
recovers the exact posterior covariance with any two points in the posterior
space.

It has the advantage that is it very easy to compute (also using an online
algorithm for the variance to avoid having to store the $x_i$ and $s_i$ values).
It is therefore the default in nutpie, and in a later section we will show some
benchmarks to compare its performance.

#### Some theoretical results for normal posteriors

If the posterior is $N(\mu, \Sigma)$, then the minimizers $\hat{\mu}$ and
$\hat{\sigma}$ of $\hat{D_F}$ converge to $\mu$, and $\hat{\sigma}^2 \to
\exp(\frac{1}{2} \log\text{diag}(\Sigma) - \frac{1}{2}
\log\text{diag}(\Sigma^{-1}))$. This is a direct consequence of $\text{Cov}[x_i]
\to \Sigma$ and $\text{Cov}[s_i] \to \Sigma^{-1}$.

$D_F$ converges to $\sum_i \lambda_i + \lambda_i^{-1}$, where $\lambda_i$ are
the generalized eigenvalues of $\Sigma$ with respect to
$\text{diag}(\hat{\sigma})$, so large and small eigenvalues are penalized. When
we choose $\text{diag}(\Sigma)$ as mass matrix, we minimize $\sum_i \lambda_i$,
and only penalize large eigenvalues. If we minimize $\mathbb{E}(ss^T)$ we
minimize $\sum \lambda_i^{-1}$ and only penalize small eigenvalues. But based on
some theoretical work for multivariate normal distributions, we know that both
large and small eigenvalues make HMC less efficient. (todo ref)

### Full mass matrix

We choose $F_{A, \mu}(y) = Ay + \mu$. This corresponds to a mass matrix $M =
(AA^T)^{-1}$. Because as we will see in a second $\hat{D}_F$ only depends on
$AA^T$ and $\mu$, we can restrict $A$ to be symmetric positive definite.

We get
$$
D[F] = \dots
$$

which is minimal if $AA^T\text{Cov}[x_i]AA^T = \text{Cov}[s_i]$, and as such
corresponds again to our earlier derivation in
\autoref{section:motivation-gaussian}. If the two covariance matrices are full
rank, we get a unique minimum at the geometric mean of $\text{Cov}[x_i]$ and
$\text{Cov}[s_i]$. If the number of dimensions is larger than the number of
draws, we can add regularization terms. And to avoid $O(n^3)$ computational
cost, we can project draws and fisher scores into the span of $x_i$ and $s_i$,
compute the regularized mean in this subspace and use the mass matrix $\dots$.
If we only store the components, we can avoid $O(n^2)$ storage, and still do all
operations we need for HMC quickly. To further reduce computational cost, we can
ignore eigenvalues that are close to one. todo expand paragraph

### Model specific diffeomorphisms

Sometimes we can suggest useful families of diffeomorphism based on the model itself.
A common reparametrization for instance is the non-centered parametrization,
where we change a model from the centered parametrization

$$
x \sim N(0, \sigma^2)
$$

into the non-centered parametrization

$$
x' \sim N(0, 1)\\
x = x' \odot \sigma
$$

We can generalize this for any choice of $0 \leq t \leq 1$ to (todo add ref, ask Aki)
$$
x' \sim N(0, \sigma^{2t})\\
x = x' \odot \sigma^{1 - t}
$$

This suggests $F_t = \dots$...

This might also have a closed form solution, I haven't checked yet.

### Normalizing flows

Define $F_\eta$ as a normalizing flow with parameters $\eta$. Use adam or
similar to minimize $D[F]$.

# Adaptation (tuning / warmup) of mass matrix

## Current state of the art

(explain Stan/PyMC algorithm)

## Choice of initial mass matrix

We use the abs of gradient at initial position. Experimentally it turned out to
be a favorable starting estimate. (Maybe show one example?)

One paragraph of intuition. What happens if we run a few leapfrog steps with
constant huge or small gradient?

## Accelerated window based adaptation (warmup/tuning) scheme

Constant step size adaptation with dual averaging. Overlapping windows, so that
we can switch to a better estimate quickly.

Intuition: We want to update quickly, and not use an old mass matrix estimate at
a point when we have more information and could already compute a better
estimate.  We could just use a tailing window and update in each step with the
previous k draws. This is computationally inefficient (unless the logp function
is very expensive), and can not easily be implemented as a streaming estimator
(see below for more details). But if we use several overlapping estimation
windows, we can compromise between optimal information usage and computational
cost, and still use streaming mass matrix estimators.

Three phases:
- Initial phase with small window size to find the typical set. Discard duplicate draws.
- Main phase with longer windows
- Final phase with constant mass matrix: Only step size is adapted. Use a
symmetric estimate of the acceptance statistic (todo ref stan discourse).

This scheme can be used with arbitrary mass matrix estimators. If the estimator
allows a streaming (ref) implementation, we do not need to store the draws
within each window.

# Numerical results
