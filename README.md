# Abstract

Hamiltonian  Monte  Carlo  (HMC)  is  a  powerful  tool  for  Bayesian
inference,  as  it  can  explore  complex  and  high-dimensional  parameter
spaces.  But  HMC’s  performance  is  highly  sensitive  to  the  geometry
of  the  posterior  distribution,  which  is  often  poorly  approximated  by
traditional mass matrix adaptations, especially in cases of non-normal or
correlated posteriors. We propose Fisher HMC, an adaptive framework
that uses the Fisher divergence to guide transformations of the parameter
space.  It  generalizes  mass  matrix  adaptation  from  affine  functions
to  arbitrary  diffeomorphisms.  By  aligning  the  score  function  of  the
transformed posterior with those of a standard normal distribution, our
method  identifies  transformations  that  adapt  to  the  posterior’s  scale
and shape. We develop theoretical foundations efficient implementation
strategies,  and  demonstrate  significant  sampling  improvements.  Our
implementation, nutpie, integrates with PyMC and Stan and delivers
better efficiency compared to existing samplers.
