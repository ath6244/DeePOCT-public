"""DeepOCT — multi-task OCT speckle denoising + choroid-layer segmentation.

Public surface:

``model``
    ``DeepOCTUltra``, the dual-decoder U-Net with the optional DT-CWT bottleneck,
    horizontal axial attention and the boundary-distribution head.
``losses``
    ``DeepOCTLoss`` (ZNCC + GAT variance + FFL + BCE/Dice segmentation) and
    ``BoundaryDistributionLoss`` with its decode helpers.
``datasets``
    Manifest-driven loaders for the OIMHS and SDOCT B-scan corpora.
``calibration``
    Conformal calibration of per-column boundary and thickness uncertainty.
``provenance``
    Run-provenance capture written into every result JSON.
"""

__all__ = ["model", "losses", "datasets", "calibration", "provenance"]
