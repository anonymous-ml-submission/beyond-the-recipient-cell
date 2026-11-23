# Adapted GHIST: provenance and installation boundary

This directory contains a **partial copy of the original GHIST code**, not a
standalone distribution of the full GHIST training and tutorial package.
The original `external_methods/GHIST/README.md` is retained as upstream
attribution, and **its references to `requirements.txt`, tutorial notebooks,
and `Figure1.png` do not describe files bundled here**. Do not run the
upstream README's installation command from this subdirectory.

- Original GHIST: https://github.com/SydneyBioX/GHIST
- Original GHIST license retained in `external_methods/GHIST/LICENSE` (GPL-3.0).
- This study uses an **adapted GHIST comparison**, which omitted optional
  components of the original published model. It is not the complete native
  GHIST workflow and should not be labeled as such.
- The project's preserved experiment implementation is invoked through
  `experiments/ghist_training.py` (which forwards to
  `scripts/ghist_donor_trainer_v1.py`). The study also includes its own
  P1 exact-benchmark code where present. These commands require the
  relevant dataset, configuration and weights; a source checkout alone
  is not an executable GHIST benchmark.

## Environment

Install the project's core requirements using its root `pyproject.toml`
according to the environment supported by your Python and PyTorch versions.
For **some optional histology workflows**, dependencies are listed separately
in `requirements-histology-optional.txt`. This list is not a pinned or validated
full training environment. Original GHIST and other optional third-party
workflows may require additional upstream dependencies and separate compatible
environments. Do **not** treat that list as an environment-reproduction proof.

## Missing assets and reproduction status

The subset deliberately does **not** ship GHIST tutorial notebooks, its
`requirements.txt`, `Figure1.png`, the original dataset, study-specific frozen
checkpoints, or the study's full saved numerical outputs. Obtain the native
GHIST tutorial and its requirements from the upstream repository when the
**native** GHIST workflow is needed, while keeping that workflow distinct from
the **adapted** model reported in this paper.

The research comparison and its numerical result are documented in the
submitted paper and `docs/REPRODUCIBILITY.md`. A full fresh training/inference
replay has not been independently established from this GitHub checkout.
