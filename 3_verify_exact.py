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

'''
NOTE:
Algorithmically inferior code to 1_verify_numeric.py and extremely slow.
However, it uses Pure Python Int which avoids any overflow issues compared to NumPy.
'''




import multiprocessing
import time
import math

# =============================================================================
# ⚙️ CONFIGURATION
# =============================================================================
RANGE_START = 2
RANGE_END   = 100_000_000  # Adjust based on patience (2M takes ~10s, 100M takes ~10-15 mins)
NUM_CORES   = multiprocessing.cpu_count()  # Use all cores

# =============================================================================
# 1. EXACT MATHEMATICAL FORMULAS (Pure Python)
# =============================================================================
def get_M_values(n):
    """Computes M1, M2 using pure python integers (No Overflow)."""
    s1 = 0
    # Optimized Sigma1/Sigma3 calculation
    limit = int(n**0.5)
    for i in range(1, limit + 1):
        if n % i == 0:
            s1 += i
            if i*i != n:
                s1 += n // i

    # We only need M1 for the new Master Formulas
    # We calculate M2 only if needed for the Universal/Hybrid ones
    return s1

def check_universal(n, M1):
    # This is the Universal Detector U3.1 (Requires M2, so we skip or recalc if needed)
    # For speed, let's strictly check the Master Formulas for p^k first
    return False

def check_pk_master(n, M1, k):
    """
    Checks the Master Formula: (M1 - 1)^k == n * (M1 - n)^k
    This is mathematically identical to the long polynomial but faster.
    """
    lhs = (M1 - 1) ** k
    rhs = n * ((M1 - n) ** k)
    return lhs == rhs

# =============================================================================
# 2. WORKER FUNCTION (Runs on each Core)
# =============================================================================
def scan_chunk(args):
    """
    Scans a range of numbers for p^2, p^3 ... p^7 matches.
    """
    start, end = args
    results = {
        'p2_hits': [], 'p3_hits': [], 'p4_hits': [],
        'p5_hits': [], 'p6_hits': [], 'p7_hits': []
    }

    # Iterate through the chunk
    for n in range(start, end):
        # 1. Compute M1 (Sum of Divisors)
        M1 = 0
        limit = int(n**0.5)
        for i in range(1, limit + 1):
            if n % i == 0:
                M1 += i
                if i*i != n: M1 += n // i

        # 2. Check Formulas (Lowest complexity first)
        # Note: We check ALL of them to ensure no cross-talk (False Positives)

        # p^2
        if (M1 - 1)**2 == n * (M1 - n)**2:
            results['p2_hits'].append(n)

        # p^3
        if (M1 - 1)**3 == n * (M1 - n)**3:
            results['p3_hits'].append(n)

        # p^4
        if (M1 - 1)**4 == n * (M1 - n)**4:
            results['p4_hits'].append(n)

        # p^5
        if (M1 - 1)**5 == n * (M1 - n)**5:
            results['p5_hits'].append(n)

        # p^6
        if (M1 - 1)**6 == n * (M1 - n)**6:
            results['p6_hits'].append(n)

        # p^7
        if (M1 - 1)**7 == n * (M1 - n)**7:
            results['p7_hits'].append(n)

    return results

# =============================================================================
# 3. VERIFICATION LOGIC
# =============================================================================
def is_true_power(n, k):
    """Verifies if n is ACTUALLY p^k."""
    # Fast root check
    root = int(round(n**(1/k)))
    if root**k != n: return False
    # Primality check on root
    if root < 2: return False
    if root == 2: return True
    if root % 2 == 0: return False
    for i in range(3, int(root**0.5) + 1, 2):
        if n % i == 0: return False
    return True

def main():
    print(f"{'='*60}")
    print(f"PARALLEL EXACT VERIFICATION (Python Integer Math)")
    print(f"Range: {RANGE_START:,} to {RANGE_END:,}")
    print(f"Cores: {NUM_CORES}")
    print(f"{'='*60}\n")

    # 1. Prepare Chunks
    chunk_size = 100_000
    chunks = []
    curr = RANGE_START
    while curr < RANGE_END:
        end = min(curr + chunk_size, RANGE_END + 1)
        chunks.append((curr, end))
        curr = end

    print(f"Scanning via {len(chunks)} chunks...")
    t0 = time.time()

    # 2. Run Parallel Scan
    with multiprocessing.Pool(NUM_CORES) as pool:
        all_results = pool.map(scan_chunk, chunks)

    # 3. Aggregate Results
    combined = {
        'p2': [], 'p3': [], 'p4': [],
        'p5': [], 'p6': [], 'p7': []
    }

    for res in all_results:
        combined['p2'].extend(res['p2_hits'])
        combined['p3'].extend(res['p3_hits'])
        combined['p4'].extend(res['p4_hits'])
        combined['p5'].extend(res['p5_hits'])
        combined['p6'].extend(res['p6_hits'])
        combined['p7'].extend(res['p7_hits'])

    total_time = time.time() - t0
    print(f"\nScan Complete in {total_time:.2f}s\n")

    # 4. Analyze False Positives
    print(f"{'Detector':<10} | {'Found':<8} | {'Status'}")
    print("-" * 40)

    categories = [
        ('p^2', 2, combined['p2']),
        ('p^3', 3, combined['p3']),
        ('p^4', 4, combined['p4']),
        ('p^5', 5, combined['p5']),
        ('p^6', 6, combined['p6']),
        ('p^7', 7, combined['p7']),
    ]

    global_fp = 0

    for name, k, hits in categories:
        fp_list = []
        for n in hits:
            # We strictly verify if the hit is physically p^k
            # If the formula says YES but math says NO, it's a False Positive
            if not is_true_power(n, k):
                fp_list.append(n)

        status = "✅ CLEAN"
        if len(fp_list) > 0:
            status = f"❌ {len(fp_list)} FP!"
            global_fp += len(fp_list)
        elif len(hits) == 0:
            status = "⚠️ No hits (Range too small?)"

        print(f"{name:<10} | {len(hits):<8} | {status}")

        if fp_list:
            print(f"   >>> FP Examples: {fp_list[:5]}")

    print("-" * 40)
    if global_fp == 0:
        print("\n🎉 SUCCESS: All formulas verified with ZERO False Positives.")
        print("   The weird results before were purely Integer Overflow.")
    else:
        print("\n❌ FAILURE: Mathematical False Positives found.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()