# MIT License
#
# Copyright (c) 2025 Arvind N. Venkat
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
import re
import numpy as np
from numba import njit

# ============================================================================
# 1. CORE NUMBA KERNELS (Fast Data Generation)
# ============================================================================

@njit(parallel=True, fastmath=True)
def precompute_divisor_sums_numba(max_n, powers):
    """
    Computes sigma_k(n) for k in powers efficiently using parallel processing.
    """
    n_powers = len(powers)
    sigma = np.zeros((n_powers, max_n + 1), dtype=np.int64)
    for i in range(n_powers):
        p = powers[i]
        for d in range(1, max_n + 1):
            d_pow = d ** p
            # Add d^p to all multiples of d
            sigma[i, d::d] += d_pow
    return sigma

@njit(fastmath=True)
def precompute_spf(max_n):
    """
    Sieve of Eratosthenes to find Smallest Prime Factor (SPF) for every number.
    """
    spf = np.arange(max_n + 1)
    spf[0:2] = 0
    i = 2
    while i * i <= max_n:
        if spf[i] == i:
            for j in range(i * i, max_n + 1, i):
                if spf[j] == j:
                    spf[j] = i
        i += 1
    return spf

@njit(fastmath=True)
def get_categories_batch(n_start, n_end, spf):
    """
    Classifies numbers into:
    1: Prime
    2: Prime Power (p^k)
    3: Semiprime (p*q)
    4: Composite (other)
    """
    size = n_end - n_start + 1
    codes = np.zeros(size, dtype=np.int8)
    powers = np.zeros(size, dtype=np.int8)
    
    for idx in range(size):
        n = n_start + idx
        if n < 2: 
            codes[idx] = 0
            continue
        
        # 1. Prime Check
        if spf[n] == n:
            codes[idx] = 1 # Prime
            powers[idx] = 1
            continue
        
        # 2. Factorization
        temp = n
        unique_cnt = 0
        total_cnt = 0
        last_p = -1
        
        while temp > 1:
            p = spf[temp]
            total_cnt += 1
            if p != last_p:
                unique_cnt += 1
                last_p = p
            temp //= p
            
        # 3. Classification
        if unique_cnt == 1:
            codes[idx] = 2 # Prime Power (p^k)
            powers[idx] = total_cnt
        elif total_cnt == 2:
            codes[idx] = 3 # Semiprime (p * q)
        else:
            codes[idx] = 4 # Other Composite
            
    return codes, powers

# ============================================================================
# 2. ROBUST PARSER & DATA PREP
# ============================================================================

def compute_macmahon_arrays(sigma, n_array):
    """
    Computes M1, M2, M3 vectors based on divisor sums.
    """
    s1 = sigma[0, n_array] # sigma_1
    s3 = sigma[1, n_array] # sigma_3
    s5 = sigma[2, n_array] # sigma_5
    
    M = {}
    # M1(n) = sigma_1(n)
    M['M1'] = s1.astype(np.int64)
    
    # M2(n) = [ (1-2n)*sigma_1(n) + sigma_3(n) ] / 8
    M['M2'] = ((1 - 2*n_array)*s1 + s3) // 8
    
    # M3 is rarely used but included for completeness
    # M3(n) = [ (40n^2 - 100n + 37)*sigma_1 - 10(3n-5)*sigma_3 + 3*sigma_5 ] / 1920
    term1 = (40*n_array**2 - 100*n_array + 37) * s1
    term2 = 10*(3*n_array - 5) * s3
    M['M3'] = (term1 - term2 + 3*s5) // 1920
    
    return M

def robust_evaluate(n_array, M_dict, formula_str):
    """
    Parses the formula as a full mathematical expression using Python's eval().
    This handles subtraction, parentheses, and order of operations correctly.
    """
    # 1. Replace caret powers: n^5 -> n**5
    cleaned = formula_str.replace('^', '**')
    
    # 2. Fix implicit multiplication: "12n" -> "12*n"
    #    Regex looks for a digit followed immediately by 'n'
    cleaned = re.sub(r'(\d)n', r'\1*n', cleaned)
    
    # 3. Prepare the context (variables available to the formula)
    context = M_dict.copy()
    context['n'] = n_array
    
    # 4. Evaluate using Python's native engine
    try:
        # Evaluate formula in a restricted scope (just numpy math + variables)
        result = eval(cleaned, {"__builtins__": {}}, context)
        
        # Handle case where result is a single scalar (e.g., formula is "0")
        if np.ndim(result) == 0:
            result = np.full(n_array.shape, result, dtype=np.int64)
            
        return result.astype(np.int64)
    except Exception as e:
        raise ValueError(f"Parser failed on formula.\nCleaned version: {cleaned}\nError: {e}")

def prepare_data(n_start, n_end):
    print(f"Initializing Data for Range {n_start} to {n_end}...")
    t0 = time.time()
    
    # We need sigma_1, sigma_3, sigma_5 for M1, M2, M3
    powers = np.array([1, 3, 5], dtype=np.int64)
    sigma = precompute_divisor_sums_numba(n_end, powers)
    
    spf = precompute_spf(n_end)
    codes, powers_arr = get_categories_batch(n_start, n_end, spf)
    
    n_array = np.arange(n_start, n_end + 1, dtype=np.int64)
    M_dict = compute_macmahon_arrays(sigma, n_array)
    
    t1 = time.time()
    print(f"Initialization Complete in {t1-t0:.2f}s")
    
    return {
        'n_array': n_array,
        'M_dict': M_dict,
        'codes': codes,
        'powers': powers_arr,
        'range': (n_start, n_end)
    }

# ============================================================================
# 3. VERIFICATION LOGIC
# ============================================================================

def verify_formula(data, name, formula_str):
    print(f"\n>> Testing: {name}")
    t0 = time.time()
    
    # --- EVALUATE FORMULA ---
    try:
        L_n = robust_evaluate(data['n_array'], data['M_dict'], formula_str)
    except Exception as e:
        print(f"   [ERROR] {e}")
        return
    # ------------------------

    # Mask of where the formula yields 0 (Detection)
    detected_mask = (L_n == 0)
    
    # --- SEMIPRIME CHECK ---
    mask_semi = (data['codes'] == 3)
    total_semi = np.count_nonzero(mask_semi)
    det_semi = np.count_nonzero(mask_semi & detected_mask)
    
    semi_status = "✗"
    is_valid_semi = False
    
    if total_semi > 0:
        pct_semi = 100.0 * det_semi / total_semi
        if det_semi == total_semi:
            semi_status = "✓"
            is_valid_semi = True
        elif det_semi > 0:
            semi_status = "⚠" # Partial detection
    
    # --- COMPOSITE CHECK (False Positives) ---
    mask_comp = (data['codes'] == 4)
    det_comp = np.count_nonzero(mask_comp & detected_mask)
    
    # --- PRIME POWER CHECK ---
    # Get all power levels present in the data (e.g., 1, 2, 3...)
    valid_powers_mask = (data['codes'] == 1) | (data['codes'] == 2)
    existing_powers = np.unique(data['powers'][valid_powers_mask]).astype(int) 
    existing_powers.sort()
    
    perfect_powers = []
    
    print(f"   {'Category':<12} | {'Detected':<10} | {'Total':<10} | {'%':<6}")
    print("   " + "-"*45)
    
    for p_k in existing_powers:
        mask_k = (data['powers'] == p_k) & valid_powers_mask
        total_k = np.count_nonzero(mask_k)
        det_k = np.count_nonzero(mask_k & detected_mask)
        
        if total_k > 0:
            pct = 100.0 * det_k / total_k
            status_sym = ""
            if det_k == total_k: 
                status_sym = "✓"
                perfect_powers.append(p_k)
            elif det_k == 0: status_sym = "✗"
            else: status_sym = "~"
                
            print(f"   p^{p_k:<10} | {det_k:<10} | {total_k:<10} | {pct:>5.1f}% {status_sym}")

    if total_semi > 0:
        print(f"   {'Semiprimes':<12} | {det_semi:<10} | {total_semi:<10} | {pct_semi:>5.1f}% {semi_status}")

    t1 = time.time()
    
    # --- RESULT SUMMARY ---
    print("   " + "-"*45)
    powers_str = "{" + ", ".join(map(str, perfect_powers)) + "}"
    
    has_composites = det_comp > 0
    has_partial_semi = (det_semi > 0) and (not is_valid_semi)
    
    if has_composites:
        print(f"   ⚠ FAILED: Found {det_comp} Composites (False Positives)")
        
    elif has_partial_semi:
        print(f"   ⚠ PARTIAL FAILURE: {det_semi} Semiprimes detected")
        if perfect_powers:
             print(f"     (Note: It effectively detects p^{powers_str})")
             
    elif not perfect_powers and not is_valid_semi:
        print(f"   NO VALID DETECTION")
        
    else:
        parts = []
        if perfect_powers:
            parts.append(f"p^{powers_str}")
        if is_valid_semi:
            parts.append("Semiprimes")
        print(f"    VALID DETECTOR: {' + '.join(parts)}")
        
    print(f"   Time: {t1-t0:.4f}s")

# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Prepare data up to 200,000 (Takes < 0.5 seconds usually)
    # Increase this number if you need to test rarer cases
    BATCH_DATA = prepare_data(2, 200_000_000)
    
    # Detectors for p^6 and p^7
    my_formulas = [
   
  ################ UNIVERSAL PRIME-POWER DETECTOR #######################
    # Detects all p^k values
   
    ("U3.1",
     "(2n^3 - 2n^2 + n - 1)*M1 + (8n^2 + 8n + 8)*M2 - (3n^2 - 3)*M1^2 - (24n + 24)*M1*M2 - (n^2 - 5n + 4)*M1^3 + 24*M1^2*M2"
    ),

   ################ HYBRID PRIME-POWER DETECTOR #######################
    # Detects p^1, p^4, and Semiprimes
    ("H2.2",
     "(3n^2 + n + 4)*M1 - 8*M2 - (3n + 3)*M1^2 + M1^3"
    ),

    # Hybrid Detector: p^1 + p^2
    ("H2.1",
     "-(n^3 - n^2 - n + 1) + (n - 1)*M1 + 8*M2"
    ),

    # Hybrid Detector: p^1 + p^5
    ("H6.13",
     "(n^6 - n^5 - n^4 + 2n^3 - n^2 - n + 1) - (3n^4 - 6n^3 - n^2 + 6n - 2)*M1 - (16n^3 - 24n^2 - 24n + 16)*M2 + (n^3 - 3n^2 + n + 1)*M1^2 - 16*M1*M2 + 64*M2^2"
    ),

    

    
   ######################################### P ^ 2 DETECTORS ################################


    # Detectors for p^2
    ("E2.1.1",
     "-(6n + 6) + (10n + 9)*M1 + (16n + 16)*M2 - (7n + 1)*M1^2 - 24*M1*M2 + n*M1^3"
    ),

    ("E2.1.2",
     "-3 + (5n + 5)*M1 + 8*M2 - (3n + 6)*M1^2 + 2*M1^3"
    ),

    ("E2.2.1",
     "(2n^2 + 2n - 1) - (2n^2 + n + 1)*M1 + 8*M2 + n*M1^2"
    ),

    ("E2.2.2",
     "(n^2 + n + 1) - (2n + 2)*M1 + M1^2"
    ),

    ("E2.2.3",
     "-(6n^2 + 3n - 6) + (7n^2 - n - 2)*M1 - (8n + 32)*M2 - n^2*M1^2 + 16*M1*M2"
    ),

    ("E2.2.4",
     "-(4n^2 + 4n + 1) + (8n^2 + 8n - 1)*M1 + (16n^2 + 16n + 8)*M2 - 7n^2*M1^2 - 24n*M1*M2 + n^2*M1^3"
    ),

    ("E2.2.5",
     "(12n^2 + 9n - 6) - (7n^2 + n + 6)*M1 + (8n + 48)*M2 - 3n^2*M1^2 + 2n*M1^3"
    ),

    ("E2.2.6",
     "(6n^2 + 6n) - (3n^2 + 5n + 5)*M1 + 16*M2 + M1^3"
    ),

    ("E2.2.7",
     "-(6n^2 + 3n - 6) + (7n^2 - n - 2)*M1 + (32n^2 + 24n - 48)*M2 - n^2*M1^2 - (32n^2 + 16n)*M1*M2 + 128*M2^2 + 16n*M1^2*M2"
    ),

    ("E2.2.8",
     "-(6n^2 + 3n - 6) + (7n^2 - n - 2)*M1 + (8n^2 - 24)*M2 - n^2*M1^2 - 16n*M1*M2 + 8*M1^2*M2"
    ),

    ("E2.3.6",
     "-(n^3 - 1) - (n^3 - 3n^2 - n + 3)*M1 + 8*M1*M2"
    ),

    ("E2.3.7",
     "(2n^3 + 5n^2 - 4n) - (9n^3 - 7n^2 - 6n + 8)*M1 - (16n^3 - 24n^2 - 48n)*M2 + n^3*M1^2 + 128*M2^2"
    ),

    ("E2.3.15",
     "-(7n^3 + 4n^2 - 5n - 3) + (5n^3 + 5n^2 - n - 5)*M1 - 24n*M2 - n^3*M1^2 + 8*M1^2*M2"
    ),

    ("E2.3.20",
     "(2n^3 + n^2 - 8n + 8) - (13n^3 - 19n^2 - 6n + 16)*M1 - (16n^3 - 8n^2 - 32n)*M2 + n^3*M1^2 - (16n^3 - 48n^2)*M1*M2 + 128*M1*M2^2"
    ),

    ("E2.3.23",
     "(10n^3 + 9n^2 - 18n + 8) - (29n^3 - 27n^2 - 14n + 24)*M1 + (16n^3 + 64n^2 + 64n)*M2 + 3n^3*M1^2 - (72n^3 - 56n^2)*M1*M2 - (128n^3 - 192n^2 - 384n)*M2^2 + 8n^3*M1^2*M2 + 1024*M2^3"
    ),

    ("E2.4.11",
     "(2n^4 - 3n^3 + n) + (n^4 - 4n^3 + n^2 + 6n - 4)*M1 - (8n^3 - 8n^2)*M2 + 64*M2^2"
    ),

    ("E2.4.22",
     "-(n^4 + 2n^3 + 2n^2 - 2n - 3) - (2n^4 - 4n^3 - 7n^2 + 4n + 5)*M1 + 8n^2*M2 + 8*M1^2*M2"
    ),

    ("E2.4.29",
     "(20n^4 + 5n^3 - 20n^2 - 4n + 8) - (19n^4 + 9n^3 - 22n^2 - 10n + 16)*M1 + (8n^3 + 96n^2)*M2 + 3n^4*M1^2 - 16n^3*M1*M2 + 128*M1*M2^2"
    ),

    ("E2.4.34",
     "(20n^4 - 7n^3 - 8n^2 - 10n + 8) + (5n^4 - 43n^3 + 12n^2 + 46n - 24)*M1 + (32n^4 - 40n^3 + 32n^2)*M2 + n^4*M1^2 + (16n^4 - 64n^3)*M1*M2 - (128n^3 - 128n^2)*M2^2 + 1024*M2^3"
    ),

    ("E2.5.15",
     "(n^5 + n^4 - 4n^3 + n^2 + n) + (n^4 - 5n^3 + 2n^2 + 6n - 4)*M1 - 8n^3*M2 + 64*M2^2"
    ),

    ("E2.5.29",
     "(n^5 - 2n^4 - 3n^3 - n^2 + 2n + 3) - (2n^4 - 3n^3 - 8n^2 + 4n + 5)*M1 + 8*M1^2*M2"
    ),

    ("E2.5.38",
     "-(n^4 - 2n^3 + 8n^2 + 4n - 8) + (13n^5 - 17n^4 - 20n^3 + 34n^2 + 10n - 16)*M1 - (8n^4 + 48n^3)*M2 - n^5*M1^2 + 128*M1*M2^2"
    ),

    ("E2.5.45",
     "-(24n^5 + 3n^4 - 18n^3 + 4n^2 + 10n - 8) + (39n^5 - 5n^4 - 60n^3 + 24n^2 + 46n - 24)*M1 + (16n^5 - 24n^4 - 224n^3)*M2 - 5n^5*M1^2 + 16n^4*M1*M2 - 128n^3*M2^2 + 1024*M2^3"
    ),
######################################### P^3 DETECTORS ####################################

    # Detectors for p^3
    ("E3.2.1",
     "-(2n^2 - n + 3)*M1 - 8n*M2 - (3n^2 - 3n + 3)*M1^2 + 24*M1*M2 + n*M1^3"
    ),

    ("E3.2.3",
     "(18n^2 - 2n + 28)*M1 + (8n^2 + 48n - 24)*M2 + (12n^2 - 23n + 4)*M1^2 - (24n + 128)*M1*M2 - n^2*M1^3 + 24*M1^2*M2"
    ),

    ("E3.3.1",
     "(n^3 + n^2 + n + 1) + (1 - 2n)*M1 - 8*M2"
    ),

    ("E3.3.3",
     "(n^3 + n^2 + n + 1) + (n^3 + n^2 - n + 2)*M1 + (8n^3 + 8n^2 + 8n)*M2 + (1 - 2n)*M1^2 - 16n*M1*M2 - 64*M2^2"
    ),


    ("E3.3.10",
     "-(3n^3 + 3n^2 + 3n + 3) - (57n^3 - 4n^2 + 74n - 6)*M1 - (16n^3 + 96n^2 - 96n)*M2 - (27n^3 - 61n^2 - 4n + 15)*M1^2 - (24n^3 - 24n^2 - 232n)*M1*M2 + 2n^3*M1^3 + 192*M1*M2^2"
    ),

    ("E3.3.12",
     "-(6n^3 + 6n^2 + 6n + 6) - (60n^3 - n^2 + 71n)*M1 - (40n^3 + 120n^2 - 72n)*M2 - (27n^3 - 61n^2 - 10n + 18)*M1^2 - (24n^3 - 24n^2 - 280n)*M1*M2 - (192n^3 + 192n^2 + 192n)*M2^2 + 2n^3*M1^3 + 384n*M1*M2^2 + 1536*M2^3"
    ),

    ("E3.4.6",
     "-(n^4 + 2n^3 + 2n^2 + 2n + 1) + (2n^4 + n^3 + 3n^2 + 2n - 2)*M1 - (8n^3 + 8n^2)*M2 - (4n^2 - 4n + 1)*M1^2 + 64*M2^2"
    ),


################################################### P ^ 4 Detectors ########################################

  	("E4.2.1",
     "(3n^2 - 2n + 4)*M1 - 8*M2 - (3n + 3)*M1^2 + M1^3"
    ),

    ("E4.2.4",
     "(9n^2 - 6n + 12)*M1 - 24*M2 + (3n^2 - 11n - 5)*M1^2 - 8*M1*M2 - 3n*M1^3 + M1^4"
    ),

    ("E4.2.2",
     "(34n^2 - 29n + 45)*M1 + (24n^2 + 40n - 72)*M2 - (3n^2 + 30n + 27)*M1^2 - (72n + 72)*M1*M2 - (3n^2 - 13n)*M1^3 + 72*M1^2*M2"
    ),

    ("E4.2.5",
     "(34n^2 - 29n + 45)*M1 + (24n^2 + 40n - 72)*M2 - (3n^2 + 30n + 27)*M1^2 + (72n^2 - 120n + 24)*M1*M2 - 192*M2^2 - (3n^2 - 13n)*M1^3 - 72n*M1^2*M2 + 24*M1^3*M2"
    ),

    ("E4.2.7",
     "(102n^2 - 87n + 135)*M1 + (72n^2 + 120n - 216)*M2 - (9n^2 + 90n + 81)*M1^2 + (272n^2 - 448n + 144)*M1*M2 + (192n^2 + 320n - 576)*M2^2 - (9n^2 - 39n)*M1^3 - (24n^2 + 240n)*M1^2*M2 - (576n + 576)*M1*M2^2 - (24n^2 - 104n)*M1^3*M2 + 576*M1^2*M2^2"
    ),

    ("E4.4.1",
     "(n^4 + n^3 + n^2 + n + 1) - (n^3 + 3n^2 + 2n)*M1 - (8n + 8)*M2 + (2n - 1)*M1^2 + 8*M1*M2"
    ),

    ("E4.4.9",
     "(7n^4 + 7n^3 + 7n^2 + 7n + 7) + (5n^4 - 26n^3 + 7n^2 - 37n + 9)*M1 + (8n^4 - 8n^3 - 40n^2 - 32n - 72)*M2 + (2n^4 - 13n^3 + 25n^2 - n - 6)*M1^2 - (8n^3 + 24n^2 - 120n)*M1*M2 - (64n - 64)*M2^2 + 64*M1*M2^2"
    ),

    ("E4.4.30",
     "-(993n^4 + 993n^3 + 993n^2 + 993n + 993) + (487n^4 + 1274n^3 + 2393n^2 + 2495n - 566)*M1 - (32n^4 - 1592n^3 - 3440n^2 - 11000n - 10592)*M2 - (303n^4 - 900n^3 + 3654n^2 - 1631n - 283)*M1^2 + (120n^4 - 24n^3 + 3368n^2 - 22048n)*M1*M2 + (192n^4 - 192n^3 - 960n^2 - 768n - 22720)*M2^2 + 25n^4*M1^3 + (48n^4 - 312n^3)*M1^2*M2 - (192n^3 + 576n^2 - 2880n)*M1*M2^2 - (1536n - 1536)*M2^3 + 1536*M1*M2^3"
    ),

    ("E4.5.3",
     "-(8n^5 + 5n^4 + 5n^3 + 5n^2 + 5n - 3) + (n^5 + 5n^4 + 22n^3 + 8n^2 - 8n + 2)*M1 - (8n^3 - 56n^2 - 32n + 32)*M2 - (3n^3 + 4n^2 - 3n + 1)*M1^2 - 8n^2*M1*M2 + 64*M2^2"
    ),

    ("E4.5.17",
     "(n^5 + 12n^4 + 12n^3 + 12n^2 + 12n + 11) - (26n^5 - 17n^4 + 51n^3 - 6n^2 + 36n - 13)*M1 - (56n^5 + 64n^4 + 104n^3 + 232n^2 + 200n + 104)*M2 + (8n^5 - 9n^4 + 81n^3 - 61n^2 + 5n - 6)*M1^2 + (8n^5 + 16n^4 + 48n^3 + 600n^2)*M1*M2 - (64n^3 - 448n^2 - 832n)*M2^2 - n^5*M1^3 - 64n^2*M1*M2^2 + 512*M2^3"
    ),

    ("E4.5.49",
     "-(1303n^5 + 1271n^4 + 1271n^3 + 1271n^2 + 1271n - 32) + (399n^5 + 2100n^4 + 2454n^3 + 3745n^2 - 993n + 56)*M1 - (208n^5 - 2096n^4 - 4912n^3 - 13144n^2 - 13592n + 320)*M2 - (385n^5 - 1343n^4 + 4700n^3 - 1912n^2 - 421n + 24)*M1^2 - (208n^5 - 784n^4 - 3048n^3 + 26872n^2)*M1*M2 - (448n^5 + 512n^4 + 832n^3 + 1856n^2 + 27968n)*M2^2 + 27n^5*M1^3 + (64n^5 - 72n^4)*M1^2*M2 + (64n^5 + 128n^4 + 384n^3 + 4800n^2)*M1*M2^2 - (512n^3 - 3584n^2 - 6656n)*M2^3 - 8n^5*M1^3*M2 - 512n^2*M1*M2^3 + 4096*M2^4"
    ),


#################################################### P ^ 5 Detectors ###########################################


    ("E5.3.12",
     "-(282n^3 + 282) + (460n^3 + 359n^2 + 946n + 783)*M1 - (96n^3 + 2456n^2 + 3752n - 840)*M2 - (133n^3 + 565n^2 + 1612n + 1315)*M1^2 - (360n^3 - 2720n^2 - 5184n - 968)*M1*M2 + (1728n - 3264)*M2^2 + (57n^3 - 239n^2 + 1120n + 924)*M1^3 + (432n^2 - 3696n)*M1^2*M2 - 1728*M1*M2^2 - (36n^3 - 119n^2 + 304)*M1^4 + 216*M1^4*M2"
    ),
    ("E5.3.15",
     "(2424n^3 + 2424) + (1094n^3 - 4532n^2 + 950n - 2628)*M1 - (7608n^3 + 14512n^2 + 16240n + 13224)*M2 - (1358n^3 + 1007n^2 + 1733n + 1112)*M1^2 - (1296n^3 - 22780n^2 - 35100n - 18808)*M1*M2 - (4320n^3 + 6624n^2 + 4320n + 2496)*M2^2 + (1311n^3 - 2698n^2 + 7715n + 48)*M1^3 - (396n^3 - 3816n^2 + 26148n)*M1^2*M2 + (13536n^2 + 14688n + 5184)*M1*M2^2 + (117n^3 - 611n^2 - 404)*M1^4 + (540n^3 - 1548n^2)*M1^3*M2 - 14688n*M1^2*M2^2 - 72n^2*M1^4*M2 + 1728*M1^3*M2^2"
    ),
    ("E5.3.18",
     "(157584n^3 + 157584) + (48482n^3 - 288170n^2 + 21689n - 189657)*M1 - (434928n^3 + 786520n^2 + 840088n + 808392)*M2 - (76736n^3 + 39611n^2 + 43532n + 16181)*M1^2 - (72144n^3 - 1248856n^2 - 1903608n - 1076224)*M1*M2 - (302400n^3 + 441216n^2 + 369792n + 193728)*M2^2 + (77136n^3 - 154885n^2 + 424985n - 32820)*M1^3 - (35496n^3 - 232128n^2 + 1417152n)*M1^2*M2 + (5184n^3 + 807552n^2 + 1057536n + 563328)*M1*M2^2 + (124416n^2 + 124416n + 262656)*M2^3 + (8352n^3 - 41132n^2 - 13088)*M1^4 + (41040n^3 - 93024n^2)*M1^3*M2 + (31104n^2 - 857088n)*M1^2*M2^2 - (373248n + 373248)*M1*M2^3 + (3240n^3 - 17352n^2)*M1^4*M2 - 15552n^2*M1^3*M2^2 + 373248*M1^2*M2^3"
    ),
    ("E5.3.2",
     "-(6n^3 + 6) - (2n^3 - 11n^2 + 3n - 8)*M1 + (24n^3 + 40n^2 + 40n + 32)*M2 + (3n^3 + 2n^2 + 5n)*M1^2 - (72n^2 + 88n + 48)*M1*M2 - (3n^3 - 7n^2 + 20n - 2)*M1^3 + 72n*M1^2*M2 + 2n*M1^4"
    ),
    ("E5.3.4",
     "(30n^3 + 30) + (14n^3 - 59n^2 + 17n - 42)*M1 - (120n^3 + 184n^2 + 184n + 144)*M2 - (11n^3 + 20n^2 + 23n - 4)*M1^2 + (376n^2 + 408n + 208)*M1*M2 + (15n^3 - 43n^2 + 110n - 12)*M1^3 - 408n*M1^2*M2 - (2n^2 + 8)*M1^4 + 48*M1^3*M2"
    ),
    ("E5.3.5",
     "(1230n^3 + 1230) + (298n^3 - 2233n^2 + 154n - 1629)*M1 - (4200n^3 + 6128n^2 + 6128n + 6648)*M2 - (493n^3 + 304n^2 + 244n - 161)*M1^2 + (72n^3 + 11216n^2 + 14688n + 8816)*M1*M2 + (1728n^2 + 1728n + 3648)*M2^2 + (570n^3 - 1292n^2 + 3280n - 408)*M1^3 + (432n^2 - 11904n)*M1^2*M2 - (5184n + 5184)*M1*M2^2 + (45n^3 - 241n^2 - 124)*M1^4 - 216n^2*M1^3*M2 + 5184*M1^2*M2^2"
    ),
    ("E5.3.8",
     "-(6n^3 + 6) + (10n^3 - 7n^2 + 18n + 23)*M1 + (72n^3 + 40n^2 - 104n - 16)*M2 - (7n^3 - 8n^2 + 22n + 40)*M1^2 - (48n^3 + 88n^2 - 72n - 80)*M1*M2 + 192n*M2^2 + (10n^3 - 10n^2 + 16n + 23)*M1^3 + (64n^2 - 72n)*M1^2*M2 - 192*M1*M2^2 - (3n^3 + 4n^2 + 4)*M1^4 + n^2*M1^5"
    ),
    ("E5.4.1",
     "-(9n^3 + 9) + (5n^4 - 6n^3 + 8n^2 + 30n + 2)*M1 + (24n^4 + 56n^3 - 32n^2 - 56n + 80)*M2 - (6n^4 - 3n^3 - 3n^2 + 21n + 3)*M1^2 - (72n^3 + 48n^2 + 24n)*M1*M2 - 192*M2^2 - (3n^4 - 14n^3 + 11n^2 - n - 2)*M1^3 + 72n^2*M1^2*M2"
    ),
    ("E5.5.0",
     "-3*(n^5 + n^4 + n^3 + n^2 + n + 1) + (6n^4 + 10n^3 + 11n^2 + 8n + 4)*M1 + (16n^2 + 16n + 16)*M2 - (3n^3 + 12n^2 + 9n)*M1^2 - (24n + 24)*M1*M2 + (n^2 + n + 1)*M1^3"
    ),
    ("E5.5.2",
     "(3n^5 + 3n^4 - 6n^3 + 3n^2 + 3n - 6) - (6n^5 - 5n^4 + 19n^3 - 22n + 2)*M1 + (32n^3 - 72n^2 - 72n + 64)*M2 + (3n^4 + 6n^3 + 6n^2 - 12n - 3)*M1^2 + (24n^2 + 24)*M1*M2 - 192*M2^2 + (1 - n^3)*M1^3"
    ),
    ("E5.5.7",
     "(198n^5 + 189n^4 - 144n^3 + 198n^2 + 189n - 144) - (223n^5 - 7n^4 + 971n^3 + 292n^2 - 612n + 186)*M1 + (32n^4 + 1160n^3 - 3208n^2 - 3072n + 2016)*M2 + (3n^5 + 99n^4 + 342n^3 + 537n^2 - 243n - 90)*M1^2 - (72n^3 - 1032n^2 - 720n - 1392)*M1*M2 - (192n + 7296)*M2^2 - (n^4 + 31n^3 + 49n^2)*M1^3 + 384*M1*M2^2"
    ),
    ("E5.6.12",
     "-(279n^6 + 309n^5 - 258n^4 + 291n^3 + 309n^2 - 258n + 12) + (382n^6 - 121n^5 + 1520n^4 + 472n^3 - 1074n^2 + 264n + 39)*M1 + (24n^6 - 32n^5 - 2024n^4 + 5008n^3 + 5304n^2 - 3288n + 48)*M2 - (186n^5 + 501n^4 + 738n^3 - 384n^2 - 102n - 27)*M1^2 - (48n^4 + 1512n^3 + 648n^2 + 2472n + 360)*M1*M2 - (384n^3 - 576n^2 - 12672n)*M2^2 + (n^5 + 55n^4 + 58n^3)*M1^3 + 1536*M2^3"
    ),


##################################################### P ^ 6 ################################################


    # Detectors for p^6
    ("E6.5.4",
     "(12n^5 - 13n^4 + 33n^3 - 10n^2 + 18n + 2)*M1 - (24n^5 + 24n^4 + 88n^3 + 32n^2 + 32n - 32)*M2 + (9n^5 - 24n^4 + 3n^3 - 24n^2 - 36n + 24)*M1^2 + (72n^4 - 24n^3 + 96n^2 - 192)*M1*M2 + (3n^5 - 15n^4 + 20n^3 - 20n^2 + 28n - 10)*M1^3 - 72n^3*M1^2*M2 + 384*M1*M2^2"
    ),

    ("E6.6.1",
     "(n^6 + n^5 + n^4 + n^3 + n^2 + n + 1) - (4n^4 + 2n^3 + 2n^2 + 2n - 2)*M1 - (16n^3 + 16n^2 + 16n + 16)*M2 + (4n^2 - 4n + 1)*M1^2 + (32n - 16)*M1*M2 + 64*M2^2"
    ),

    ("E6.6.7",
     "(3n^6 + 3n^5 - 5n^4 + 15n^3 - 5n^2 + 9n + 1)*M1 - (32n^3 + 16n^2 + 16n - 16)*M2 - (12n^4 - 6n^3 + 12n^2 + 18n - 12)*M1^2 - (48n^3 - 48n^2 + 96)*M1*M2 + (4n^3 - 10n^2 + 14n - 5)*M1^3 + 192*M1*M2^2"
    ),

    ("E6.6.8",
     "(2n^6 + 2n^5 + 2n^4 + 2n^3 + 2n^2 + 2n + 2) - (6n^6 - 8n^5 + 28n^4 - 8n^3 + 16n^2 + n - 5)*M1 + (24n^6 + 24n^5 + 56n^4 - 24n^3 - 16n^2 - 48n - 16)*M2 - (6n^6 - 16n^5 + 6n^4 - 16n^3 - 21n^2 + 32n - 9)*M1^2 - (48n^5 + 16n^4 + 80n^3 - 16n^2 - 152n + 72)*M1*M2 - (128n^3 + 128n^2 + 128n)*M2^2 - (2n^6 - 10n^5 + 12n^4 - 8n^3 + 13n^2 - 9n + 2)*M1^3 + 48n^4*M1^2*M2 + 512*M2^3"
    ),

    ("E6.7.12",
     "(2n^7 + 4n^6 + 4n^5 + 4n^4 + 4n^3 + 4n^2 + 4n + 2) - (4n^7 + 2n^6 + 2n^5 + 30n^4 - 4n^3 + 20n^2 - 3n - 5)*M1 + (8n^6 + 8n^5 + 8n^4 - 56n^3 - 48n^2 - 80n - 16)*M2 + (16n^5 - 12n^4 + 24n^3 + 13n^2 - 30n + 9)*M1^2 + (32n^4 - 80n^3 + 80n^2 + 120n - 72)*M1*M2 - (128n^3 + 128n^2)*M2^2 - (4n^4 - 8n^3 + 13n^2 - 9n + 2)*M1^3 + 512*M2^3"
    ),




    ]
    

    for name, f_str in my_formulas:
        verify_formula(BATCH_DATA, name, f_str)