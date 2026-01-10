# Prime Power Detection via Quasimodular Polynomials

**Author**: Arvind N. Venkat

This repository contains the reference implementation and verification scripts associated with the manuscript **"Prime Power Detection via Quasimodular Polynomials: The Binomial Master Theorem and Universal Detectors"**.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18207268.svg)](https://doi.org/10.5281/zenodo.18207268)

### Pre-print:
* **DOI** - 10.5281/zenodo.18207268
* **URL** - https://zenodo.org/records/18207268

## Abstract

We extend the Craig–van Ittersum–Ono framework for prime detection to prime powers via two foundational results: the **Prime Recovery Lemma**, which reduces detection to polynomial elimination, and the **Binomial Master Theorem**, which generates explicit detectors. We further derive a **Universal Detector** $\mathcal{U}(n)$ that vanishes on all prime powers simultaneously.

This repository provides the computational engine used to:
1.  **Discover** these polynomial identities via nullspace analysis.
2.  **Prove** them algebraically (Symbolic Verification).
3.  **Verify** the converse direction empirically up to $N = 10^8$ (Numerical Verification).

## Repository Contents

### 1. Source Code (`code/`)

* **`verify_detectors.py`** The main verification engine. It performs two critical tasks:
  1.  **Symbolic Proof:** Uses SymPy to algebraically prove that the detectors vanish on their target loci (e.g., verifying $\mathcal{U}(p^k) \equiv 0$).
  2.  **Numerical Search:** Runs a high-performance sieve to check for false positives up to $N=100,000,000$.

* **`1_prime_power_detector_search.py`** The discovery algorithm. It generates the training set of prime powers, constructs the Vandermonde-style matrix of MacMahon functions, and computes the nullspace to discover new detector formulas.

* **`2_merge_detectors.py`** Utility script to consolidate, rank, and format the discovered formulas into LaTeX tables and Python tuples.

### 2. Experimental Data (`/results`)
The results folder documents the full scientific pipeline, from raw discovery to final verification.

#### A. Discovery Logs (`results/1_detector search/`)
This folder documents the mining process, starting from individual search runs and culminating in the merged master datasets.

* **`multiple runs output/`**: The raw, granular logs from individual search iterations (e.g., `p3_detectors...`, `p4_detectors...`), documenting exactly how specific prime power orders were isolated.
* **`merge_detectors_console_output.txt`**: A verbatim capture of the console stream during the merging process. This log provides a sequential record of how individual search files were integrated and validated to produce the master lists.
  
**Master Outputs (Post-Merge):**
* **`master_detectors_tuples.py`**: A structured, Python-importable list containing every discovered polynomial equation as a tuple. This file serves as the direct data source for the verification scripts.
* **`master_detectors_readable.txt`**: A comprehensive, human-readable catalogue of all polynomial detectors discovered by the engine.
* **`master_appendix.tex`**: Auto-generated LaTeX code containing the full set of discovered equations.
    * *Note: For the sake of brevity, the filtered appendix in the manuscript contains only a selected subset of these equations.*

#### B. Verification Logs (`results/2_detector verification/`)
This folder contains the computational proofs of correctness.

* **`verification_sample_log.txt`**: **[Primary Reference]** A clean execution log verifying the specific Theorems and Lemmas presented in the manuscript (Universal Detector, Hybrid Detectors, etc.).
    * *Note: This log covers a sample range up to N=200,000 to allow for immediate reproducibility of every equation cited in the paper.*
* **`verification_all_detectors_from_search.txt`**: A comprehensive validation log checking the hundreds of raw equations found during the search phase to ensure no false positives exist in the broader dataset.
    * *Note: Due to algebraic equivalence, some binomial equations appear here in simplified or alternate forms compared to the specific instances derived in the paper.*

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

@misc{naladiga_venkat_2026_18207268,
  author       = {Naladiga Venkat, Arvind},
  title        = {Prime Power Detection via Quasimodular
                   Polynomials: The Binomial Master Theorem and
                   Universal Detectors
                  },
  month        = jan,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18207268},
  url          = {https://doi.org/10.5281/zenodo.18207268},
}



---

## License

The content of this repository is dual-licensed:

- **MIT License** for `prime-cube-taxicab-verifier.py`  
- **CC BY 4.0** (Creative Commons Attribution 4.0 International) for all other content (paper, results, README, etc.)
