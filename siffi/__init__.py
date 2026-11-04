#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Antti Kukkurainen
# Finnish Meteorological Institute
# Correspondence: antti.kukkurainen@fmi.fi

"""
SIFFI
=====

SIFFI retrieval algorithm [1]_. The approximation error model is 
described in [2]_.

How to use the code
----------------------------
1. Initialize retrieval input using::

    retrieval_input = siffi.RetrievalConfig()

All attributes initialized as ``None`` need to be set, see
``help(siffi.RetrievalConfig)`` for the full list of inputs and options.
The forward model and its Jacobian are called as::

    forward_model(surface_reflectance, sif, retrieval_input.fmodel_kwargs)
    jacobian(surface_reflectance, sif, retrieval_input.fmodel_kwargs)

2. Run the retrieval using::

    retrieval_results = siffi.retrieval(retrieval_input)

References:
----------------------------
.. [1] A. Kukkurainen, A. Lipponen, V. Kolehmainen, A. Arola, S. Cogliati
   and N. Sabater, "SIFFI: Bayesian solar-induced fluorescence retrieval
   algorithm for remote sensing of vegetation", Remote Sensing of
   Environment, vol. 318, 114558, 2025.
   https://doi.org/10.1016/j.rse.2024.114558
.. [2] A. Kukkurainen, A. Lipponen, N. Sabater, A. Arola and
   V. Kolehmainen, "Simulation study on improving spectrally resolved
   solar-induced fluorescence satellite retrievals under uncertain
   atmospheric states", IEEE Transactions on Geoscience and Remote
   Sensing, vol. 63, pp. 1-9, 2025.
   https://doi.org/10.1109/TGRS.2025.3590496

Contact info
----------------------------
Author: Antti Kukkurainen

Finnish Meteorological Institute

Correspondence: antti.kukkurainen@fmi.fi

"""


from .core import BoundWarning, RetrievalConfig, retrieval

__all__ = ["RetrievalConfig", "retrieval", "BoundWarning"]

__version__ = "0.7.0"


def __dir__():
    return __all__ + ["__version__"]
