# Polynomial Prime-Power Detectors

**Author**: Arvind N. Venkat

This work is associated with the paper regarding the detection of prime powers using partition-theoretic modular forms and has been archived and assigned a permanent identifier on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18005216.svg)](https://doi.org/10.5281/zenodo.18005216)

### Pre-print:
* **DOI** - 10.5281/zenodo.18005216
* **URL** - https://zenodo.org/records/18005216

## Abstract

We present a computational verification of a hierarchy of polynomial identities that distinguish prime powers $p^k$ from composite integers. These detectors are derived from the theory of partition moments and MacMahon's modular forms ($M_1, M_2$). This repository contains the source code for both the symbolic proofs (confirming the algebraic validity of the identities) and the numerical verification (confirming the converse direction). Using a sieve algorithm for divisor sums, we provide strong empirical evidence that these polynomial identities vanish *only* on the intended prime power sets for all integers $n \le 200,000,000$.

## Repository Contents


* `1_verify_numeric.py` : A high-performance sieve algorithm that tests the detectors against integers up to $N=200,000,000$ to empirically validate the converse direction (zero false positives).
* `1_results_verify_numeric.txt` : The execution log containing the summary of the numerical scan, false positive checks, and performance metrics.
* `2_verify_symbolic.py` : A computer algebra script (using SymPy) that symbolically expands the modular form identities to prove they vanish exactly on the target prime power sets.
* `2_results_verify_symbolic.txt` : The generated output log confirming the algebraic proofs for detectors $p^2$ through $p^6$.

## Prerequisites

* Python 3.8 or higher
* Python libraries: `sympy`, `numpy`

## Getting Started

To run the code and reproduce the verification results, you will need to have Python 3.8+ and the necessary libraries installed.

### 1. Clone this repository:

```bash
git clone https://github.com/arvindvenkat01/polynomial-prime-power-detectors
cd prime-power-detectors
```

### 2. Install required dependencies

The symbolic verification requires sympy for algebraic manipulation, and the numeric verification may utilize numpy for array operations.

```bash
pip install sympy numpy
```

### 3. Run the verification scripts

To run the Symbolic Verification (Algebraic Proofs):

```bash
python verify_symbolic.py
```

To run the Numerical Verification (Sieve up to 200M):

```bash
python verify_numeric.py
```

## Citation
If you use this work, please cite the paper using the following BibTeX entry:

@misc{naladiga_venkat_2025_18005216,
  author       = {Naladiga Venkat, Arvind},
  title        = {Polynomial Detectors of Prime Powers:  Universal
                   and Recursive Identities for MacMahon Partition
                   Functions
                  },
  month        = dec,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18005216},
  url          = {https://doi.org/10.5281/zenodo.18005216},
}



---

## License

The content of this repository is dual-licensed:

- **MIT License** for `prime-cube-taxicab-verifier.py`  
- **CC BY 4.0** (Creative Commons Attribution 4.0 International) for all other content (paper, results, README, etc.)
