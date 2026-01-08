# Prime Power Detection via Quasimodular Polynomials

**Author**: Arvind N. Venkat

This repository contains the reference implementation and verification scripts associated with the manuscript **"Prime Power Detection via Quasimodular Polynomials: The Binomial Master Theorem and Universal Detectors"**.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18067765.svg)](https://doi.org/10.5281/zenodo.18067765)

### Pre-print:
* **DOI** - 10.5281/zenodo.18067765
* **URL** - https://zenodo.org/records/18067765

## Abstract

We extend the Craig–van Ittersum–Ono framework for prime detection to prime powers via two foundational results: the **Prime Recovery Lemma**, which reduces detection to polynomial elimination, and the **Binomial Master Theorem**, which generates explicit detectors. We further derive a **Universal Detector** $\mathcal{U}(n)$ that vanishes on all prime powers simultaneously.

This repository provides the computational engine used to:
1.  **Discover** these polynomial identities via nullspace analysis.
2.  **Prove** them algebraically (Symbolic Verification).
3.  **Verify** the converse direction empirically up to $N = 10^8$ (Numerical Verification).

## Repository Contents

The source code is located in the `code/` directory:

* **`verify_detectors.py`** The main verification engine. It performs two critical tasks:
  1.  **Symbolic Proof:** Uses SymPy to algebraically prove that the detectors vanish on their target loci (e.g., verifying $\mathcal{U}(p^k) \equiv 0$).
  2.  **Numerical Search:** Runs a high-performance sieve to check for false positives up to $N=100,000,000$.

* **`1_prime_power_detector_search.py`** The discovery algorithm. It generates the training set of prime powers, constructs the Vandermonde-style matrix of MacMahon functions, and computes the nullspace to discover new detector formulas.

* **`2_merge_detectors.py`** Utility script to consolidate, rank, and format the discovered formulas into LaTeX tables and Python tuples.

## Prerequisites

* Python 3.8 or higher
* Python libraries: `sympy`, `numpy`, `numba`

## Getting Started

To run the code and reproduce the verification results, you will need to have Python 3.8+ and the necessary libraries installed.

### 1. Clone this repository:

```bash
git clone https://github.com/arvindvenkat01/polynomial-prime-power-detectors
cd polynomial-prime-power-detectors
```

### 2. Install required dependencies

We use numba for high-performance integer sieving and sympy for exact symbolic algebra.

```bash
pip install -r requirements.txt
```

### 3. Run the verification

Navigate to the code directory and run the verification engine:

```bash
cd code
python 3_verify_detectors.py
```
Output will display the status (VALID/FAIL) and the algebraic proof for each detector.


To run the Discovery Algorithm (search for new formulas):

```bash
cd code
python 1_prime_power_detector_search.py
```

## Citation
If you use this work, please cite the paper using the following BibTeX entry:

@misc{naladiga_venkat_2025_18067765,
  author       = {Naladiga Venkat, Arvind},
  title        = {Prime Power Detection via Quasimodular Forms: The
                   Binomial Master Theorem and Universal Invariants
                  },
  month        = dec,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v2},
  doi          = {10.5281/zenodo.18067765},
  url          = {https://doi.org/10.5281/zenodo.18067765},
}



---

## License

The content of this repository is dual-licensed:

- **MIT License** for `prime-cube-taxicab-verifier.py`  
- **CC BY 4.0** (Creative Commons Attribution 4.0 International) for all other content (paper, results, README, etc.)
