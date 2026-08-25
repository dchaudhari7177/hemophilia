"""
FVIII / F8 structural and biochemical reference tables.

Everything in this module is *prior biological knowledge* -- it is derived from
UniProt P00451, RefSeq NM_000132.4 and the published FVIII inhibitor-epitope
literature. None of it is derived from the CHAMP labels, so it can be applied to
train, validation and test rows alike without leaking the outcome.

This is the layer the reference works do not have: instead of label-encoding the
raw HGVS string (which is unique per patient and therefore an identifier), we
turn each variant into a vector of *mechanistic* descriptors that transfer to a
patient the model has never seen.
"""

from __future__ import annotations
