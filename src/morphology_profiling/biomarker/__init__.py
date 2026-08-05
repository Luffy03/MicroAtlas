"""
biomarker/
==========
Unsupervised biomarker discovery modules for U2OS-Cell-Painting.

Modules:
  - perturbation:        Compound vs DMSO statistical testing + dose-response
  - unsupervised:        UMAP + Agglomerative clustering
  - feature_attribution: Statistical biomarker identification (t-test, Cohen's d)

No supervised classifiers are used. All analysis is based on
statistical testing, clustering, and unsupervised learning.
"""
