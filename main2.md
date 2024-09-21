# Introduction

## Hamiltonian markov chain monte carlo

Incl. what is the mass matrix
Importance of mass matrix choice
How is it chosen today

## Current challences in HMC

- Slow warmup phase (chicken-egg problem). (mention long trajectories in early phase)
- Bad performance with correlated posteriors. (related to eigenvalues as shown later. Especially if datasets get large)
- Bad geometry in posterior (funnels etc)


# Adaptation (tuning / warmup) of mass matrix

## Current state of the art
(explain stan/pymc algo)

## Choice of initial mass matrix

We use the abs of gradient at initial position. Experimentially it turned out to be a favorable starting estimate. (Maybe show one example?)

One paragraph of
intuition. What happens if we run a few leaprog steps with constant huge or small gradient? (for more intuition see following chapter)

## Accelerated window based adaptation (warmup/tuning) scheme

Constant step size adaptation with dual averaging. Overlapping windows, so that we can switch to a better estimate quickly.

Intuition: We want to update quickly, and not use an old mass matrix estimate at a point when we have more information and could already compute a better estimate.
We could just use a tailing window and update in each step with the previous k draws. This is computationally inefficient (unless the logp function is very expensive), and can not easily be implemented as a streaming estimator (see below for more details). But if we use several overlpping estimation windows, we can compromize between optimal information usage and computational cost, and still use streaming mass matrix estimators.

Three phases:
- Initial phase with small window size to find the typical set. Discard duplicate draws.
- Main phase with longer windows
- Final phase with constant mass matrix: Only step size is adapted, and we use a symmetric estimate.

This scheme can be used with arbitrary mass matrix estimators (for instance the estimators from the following chapter). If the estimator allows a streaming (ref)
implementation, we do not need to store the draws within each window.

# Construct a mass matrix from a small number of samples

## Current state of the art

- Use the inverse of the variances as mass matrix.
 * Diagonal 
 * Full

 KL divergence based estimator.

## Motivation: Example with independent gaussian posterior

Can we do better than using the variance of the posterior?
Let's assume the posterior has N(mu_i, \sigma^2_i). The best mass matrix
estimate woauld be 1/\sigma_i^2.

If we have two points where we know the gradient of the logp density, we can compute this mass matrix exactly as $var(x_i) / var(grad_i)$.

Proof....

## A new mass matrix estimator based on gradients

Just definitions, explanations later.

diagonal and low rank are available in nutpie only.

For simplicity we assume n < k so that all covariances are positive definite. For different case see below.

### Full mass matrix


def

intuition: eigenvalues of gradient and variance matrix

Estimate1: mean of draws. Estimate 2: mean of grads. Combine using intrinsic metric on space of symmetric positive matrices.

### Diagonal

Here n < k doesn't matter.

definition

intuition

### Low rank mass matrix

### Arbitrary normalizing flows

Can be extended to normalizing flows.


## Properties and connections

### gaussian posterior, with n dims, n draws

exact result

### gaussian posterior, with n dims, k draws

exact in a k-dim subspace

### Connection to condition number

minimize |log(eigvals)|

### As minimum of fisher divergence


## Regularization

# Implementation

# Numerical results
