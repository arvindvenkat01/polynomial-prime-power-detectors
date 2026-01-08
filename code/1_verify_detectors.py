"""
============================================================================
MacMahon Formula Verification System v2(Optimized Production)
============================================================================
Author: Arvind Venkat
Date: 2026-01-02
Version: 2.0 (Safe BigInt + Auto M3 Optimization)

DESCRIPTION:
    A high-performance verification engine for MacMahon-style polynomial 
    detectors (p^k). It uses a hybrid approach:
    1. Fast Sieve (Numba) for factorization.
    2. Python Big-Integers (NumPy Object Arrays) for formula evaluation 
       to prevent overflow on high powers (e.g., n^7, M3 terms).
    3. Automatic Optimization: Skips expensive M3/Sigma_5 calculations 
       if the input formulas do not require them.

USAGE:
    1. Define your formulas in the 'all_detectors' list at the bottom.
    2. Set N_END to your desired range (e.g., 200_000_000).
    3. Run. The system will auto-detect if M3 is needed.

LICENSE: MIT
============================================================================
"""

import time
import re
import sys
import numpy as np
from numba import njit
from sympy import symbols, simplify, Rational

# ============================================================================
# ⚙️ CONFIGURATION
# ============================================================================
N_START = 2
N_END   = 10_000_000      # Set to desired range. Will be slow beyond 10M. Will need parallel multi-core processing.
CHUNK_SIZE = 50_000       # Rows per batch (Keeps RAM usage low)

# ============================================================================
# 1. CORE NUMBA KERNELS (Fast Factorization)
# ============================================================================

@njit(fastmath=True)
def get_spf(max_n):
    """
    Computes Smallest Prime Factor (SPF) sieve up to max_n.
    Complexity: O(N log log N)
    """
    spf = np.arange(max_n + 1, dtype=np.int32)
    spf[0:2] = 0
    i = 2
    while i * i <= max_n:
        if spf[i] == i:
            for j in range(i * i, max_n + 1, i):
                if spf[j] == j: spf[j] = i
        i += 1
    return spf

# ============================================================================
# 2. DATA GENERATOR (Safe Big-Int Mode)
# ============================================================================

def generate_data_safe(start, end, spf, calc_m3=True):
    """
    Generates M1, M2, (and optionally M3) using Python Big-Integers.
    
    Args:
        start (int): Start of range.
        end (int): End of range.
        spf (np.array): Precomputed SPF sieve.
        calc_m3 (bool): If False, skips Sigma_5 and M3 calculation for speed.
        
    Returns:
        dict: Dictionary of numpy object arrays containing n, M1, M2, M3.
    """
    size = end - start + 1
    n_vals = list(range(start, end + 1))
    
    # Pre-allocate lists (Python lists hold arbitrary precision ints automatically)
    M1_list = [0] * size
    M2_list = [0] * size
    M3_list = [0] * size
    codes = np.zeros(size, dtype=np.int8)
    powers = np.zeros(size, dtype=np.int8)
    
    for i, n in enumerate(n_vals):
        # --- A. Factorize (Fast Lookup) ---
        temp = n
        factors = {}
        while temp > 1:
            p = int(spf[temp])
            factors[p] = factors.get(p, 0) + 1
            temp //= p
            
        # --- B. Classify Number ---
        uniq = len(factors)
        tot = sum(factors.values())
        if uniq == 1:
            codes[i] = 1 if tot == 1 else 2 # Prime or Prime Power
            powers[i] = tot
        elif tot == 2:
            codes[i] = 3 # Semiprime
        else:
            codes[i] = 4 # Composite
            
        # --- C. Compute Sigmas (Python Big Ints) ---
        s1, s3 = 1, 1
        s5 = 1
        
        for p, a in factors.items():
            s1 *= (p**(a+1) - 1) // (p - 1)
            p3 = p**3
            s3 *= (p3**(a+1) - 1) // (p3 - 1)
            
            # Optimization: Skip s5 if M3 is not needed
            if calc_m3:
                p5 = p**5
                s5 *= (p5**(a+1) - 1) // (p5 - 1)
            
        # --- D. Compute MacMahon Arrays ---
        # Formulas use pure Python math to avoid numpy int64 overflow
        
        # M1 = Sigma_1
        M1_list[i] = s1
        
        # M2 = [(1-2n)s1 + s3] / 8
        term_n = (1 - 2*n)
        M2_list[i] = (term_n * s1 + s3) // 8
        
        # M3 = [(40n^2 - 100n + 37)s1 - 10(3n-5)s3 + 3s5] / 1920
        if calc_m3:
            t1 = (40*n**2 - 100*n + 37) * s1
            t2 = 10 * (3*n - 5) * s3
            M3_list[i] = (t1 - t2 + 3*s5) // 1920
        else:
            M3_list[i] = 0
        
    return {
        'n': np.array(n_vals, dtype=object),
        'M1': np.array(M1_list, dtype=object),
        'M2': np.array(M2_list, dtype=object),
        'M3': np.array(M3_list, dtype=object),
        'codes': codes,
        'powers': powers
    }

# ============================================================================
# 3. UTILITIES & SYMBOLIC PROOF
# ============================================================================

def clean_formula(f_str):
    """Normalizes formula string for Python eval()."""
    s = f_str.replace('^', '**')
    # Add explicit multiplication: 2n -> 2*n, )n -> )*n
    s = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', s)
    s = re.sub(r'\)([\w])', r')*\1', s)
    return s

def get_symbolic_proof(formula_str, valid_k_list):
    """
    Uses SymPy to mathematically prove the formula for the detected powers.
    This provides verification beyond the numeric range tested.
    """
    integers = sorted(list(set([k for k in valid_k_list if isinstance(k, int)])))[:3]
    if not integers: return " [Sym: N/A]"
    
    p = symbols('p', integer=True, positive=True)
    f_clean = clean_formula(formula_str)
    proven = []
    
    for k in integers:
        # Define n as p^k
        n_sym = p**k
        
        # Define Sigma functions for p^k
        def sig(x): return (p**(x*(k+1)) - 1) / (p**x - 1)
        s1 = sig(1); s3 = sig(3); s5 = sig(5)
        
        # Define M functions symbolically
        M1 = s1
        M2 = ((1 - 2*n_sym)*s1 + s3) * Rational(1, 8)
        
        t1 = (40*n_sym**2 - 100*n_sym + 37) * s1
        t2 = 10 * (3*n_sym - 5) * s3
        M3 = (t1 - t2 + 3*s5) * Rational(1, 1920)
        
        M_sym = {'n': n_sym, 'M1': M1, 'M2': M2, 'M3': M3}
        
        try:
            # Check if Expression simplifies to 0
            expr = eval(f_clean, {"__builtins__": {}}, M_sym)
            if simplify(expr) == 0: proven.append(str(k))
        except Exception: pass
        
    if proven: return f" [SymPy: Verified p^{{{', '.join(proven)}}}]"
    return " [SymPy: Fail]"

def print_progress(curr, total, t0, status=""):
    """Display progress bar."""
    pct = curr / total * 100
    elap = time.time() - t0
    rate = curr / elap if elap > 0 else 0
    eta = (total - curr) / rate if rate > 0 else 0
    
    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "-" * (bar_len - filled)
    
    sys.stdout.write(f"\r|{bar}| {pct:5.1f}% [ETA: {eta:3.0f}s] {status}")
    sys.stdout.flush()

# ============================================================================
# 4. EXECUTION ENGINE
# ============================================================================

def run_verification(formulas):
    print(f"\n{'='*100}")
    print(f"MacMahon Formula Verification System v12.1")
    print(f"Range: {N_START:,} to {N_END:,} | Count: {len(formulas)} Formulas")
    print(f"{'='*100}\n")
    
    # --- 1. Prepare Formulas & Optimize ---
    print("Step 1: Preparing Formulas...", end=" ")
    clean_formulas = []
    requires_m3 = False
    
    for name, f_str in formulas:
        c_str = clean_formula(f_str)
        clean_formulas.append( (name, c_str, f_str) )
        if "M3" in f_str:
            requires_m3 = True
            
    print("Done.")
    if requires_m3:
        print("        M3 Detected. Full calculation enabled.")
    else:
        print("        No M3 found. Skipping expensive Sigma_5 calculations. (FAST MODE)")
    
    # --- 2. Initialize SPF Sieve ---
    print("Step 2: Initializing SPF Sieve...", end=" ")
    t0_init = time.time()
    spf = get_spf(N_END)
    print(f"Done ({time.time()-t0_init:.2f}s)\n")
    
    # Initialize Stats 
    stats = {n: {'pkt':{}, 'pk':{}, 'semi':0, 'st':0, 'fp':0} for n, _, _ in clean_formulas}
    
    # --- 3. Batch Processing Loop ---
    curr = N_START
    t_start = time.time()
    
    while curr <= N_END:
        c_end = min(curr + CHUNK_SIZE - 1, N_END)
        print_progress(curr, N_END, t_start, f"Processing {curr:,}...")
        
        # Generate Chunk (Auto-optimizes M3 based on flag)
        chunk = generate_data_safe(curr, c_end, spf, calc_m3=requires_m3)
        
        # Pre-calculate masks
        valid_pk = (chunk['codes'] == 1) | (chunk['codes'] == 2)
        u_p, c_p = np.unique(chunk['powers'][valid_pk], return_counts=True)
        semi_cnt = np.count_nonzero(chunk['codes'] == 3)
        
        # Evaluate Each Formula
        for name, f_clean, _ in clean_formulas:
            # Track Totals
            for k, count in zip(u_p, c_p):
                stats[name]['pkt'][int(k)] = stats[name]['pkt'].get(int(k), 0) + count
            stats[name]['st'] += semi_cnt
            
            try:
                # Big-Int Evaluation
                res = eval(f_clean, {"__builtins__": {}}, {
                    'n': chunk['n'], 'M1': chunk['M1'], 
                    'M2': chunk['M2'], 'M3': chunk['M3']
                })
                
                detected_mask = (res == 0)
                if not np.any(detected_mask): continue
                
                # Record Hits (Prime Powers)
                pk_hits = detected_mask & valid_pk
                if np.any(pk_hits):
                    hk, hc = np.unique(chunk['powers'][pk_hits], return_counts=True)
                    for k, count in zip(hk, hc):
                        stats[name]['pk'][int(k)] = stats[name]['pk'].get(int(k), 0) + count
                        
                # Record Semiprimes
                stats[name]['semi'] += np.count_nonzero(detected_mask & (chunk['codes'] == 3))
                
                # Record False Positives
                stats[name]['fp'] += np.count_nonzero(detected_mask & (chunk['codes'] == 4))
                
            except Exception as e:
                # Log error but continue
                pass
            
        curr = c_end + 1
        
    print_progress(N_END, N_END, t_start, "Complete!          ")
    
    # --- 4. Final Report ---
    print(f"\n\n{'='*115}")
    print(f"{'ID':<15} | {'Status':<15} | {'FP':<5} | {'Detected Categories & Symbolic Proof'}")
    print(f"{'-'*115}")
    
    for name, _, f_orig in clean_formulas:
        st = stats[name]
        targets = []
        
        # Determine validated powers (>99% detection rate)
        for k in sorted(st['pkt'].keys()):
            total = st['pkt'][k]
            hits = st['pk'].get(k, 0)
            if total > 0 and (hits / total > 0.99):
                targets.append(k)
                
        # Determine semiprimes
        if st['st'] > 0 and (st['semi'] / st['st'] > 0.99): 
            targets.append("Semi")
        
        fp = st['fp']
        
        # Determine Status
        if not targets and fp == 0:
            status = "❌ FAIL"
            det = "None"
        elif fp > 0:
            status = "⚠️ MIXED"
            det = f"Targets: {targets} (FP: {fp})"
        else:
            status = "✅ VALID"
            # Format nicely
            ints = [x for x in targets if isinstance(x, int)]
            if len(ints) > 4 and ints[-1]-ints[0] == len(ints)-1:
                det = f"Universal (p^{ints[0]}..p^{ints[-1]})"
            else:
                det = f"{targets}"
            
            # Append Symbolic Proof
            det += get_symbolic_proof(f_orig, targets)
            
        print(f"{name:<15} | {status:<15} | {fp:<5} | {det}")
    print(f"{'='*115}\n")


# ============================================================================
# MAIN: DEFINE YOUR DETECTORS HERE
# ============================================================================


if __name__ == "__main__":
    
# Example detectors - replace with your actual list

    all_detectors = [
    		('E1.1', '(n + 1) - 1*M1'),
           	('E2.2', '(n^3 - n^2 - n + 1) - (n - 1)*M1 - 8*M2'),
        	('U3.1', '(2*n^3 - 2*n^2 + n - 1)*M1 + (8*n^2 + 8*n + 8)*M2 - (3*n^2 - 3)*M1^2 - (24*n + 24)*M1*M2 - (n^2 - 5*n + 4)*M1^3 + 24*M1^2*M2'),
        	# Add more detectors here...
        
        ]

    
# Run verification
    run_verification(all_detectors)
