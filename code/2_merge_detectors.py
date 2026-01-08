"""
merge_detectors.py

Consolidates multiple detector output files into unified master files.
Handles deduplication, re-sorting by complexity, and global re-indexing.

Usage:
    1. Place this script in the same folder as your detector output files.
    2. Run: python merge_detectors.py

Author: Arvind Venkat
Date: Jan 2025
"""

import os
import re
import glob

# =============================================================================
# PARSING LOGIC (Fixed for Multi-line Format)
# =============================================================================
def parse_readable_file(filepath):
    """Extracts formulas from a readable .txt file using block parsing."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split file into blocks based on double newlines
    # (The search script writes \n\n after each entry)
    blocks = content.split('\n\n')
    
    results = []
    for block in blocks:
        block = block.strip()
        if not block: continue
        if not block.startswith('E'): continue # Skip headers/footers
        
        lines = block.split('\n')
        if len(lines) < 2: continue
        
        # Line 1: E4.1 (Deg 3) [Len 45]
        # Line 2: L(n) = ...
        header_line = lines[0]
        formula_line = " ".join(lines[1:]) # Join in case formula wraps
        
        # Extract Power from ID (E4.1 -> 4)
        id_match = re.match(r"E(\d+)\.(\d+)", header_line)
        if not id_match: continue
        
        power = int(id_match.group(1))
        
        # Extract Formula
        if "L(n) =" in formula_line:
            clean_formula = formula_line.split("L(n) =", 1)[1].strip()
            
            # Normalize formula string (remove spaces) for deduplication
            norm = clean_formula.replace(" ", "").strip()
            
            results.append({
                'power': power,
                'formula': clean_formula,
                'norm': norm,
                'len': len(clean_formula)
            })
            
    return results

def formula_to_latex(formula_str):
    """Simple LaTeX converter for the table."""
    latex = formula_str.replace('*', '')
    latex = re.sub(r'M(\d+)', r'M_{\1}(n)', latex)
    
    # Break lines logic
    lines = []
    current = ""
    terms = re.split(r'(\s*[+\-]\s*)', latex)
    for term in terms:
        if len(current) + len(term) > 85 and current:
            lines.append(current.strip())
            current = "\\quad " + term
        else:
            current += term
    if current: lines.append(current.strip())
    
    if len(lines) == 1: return f"\\({lines[0]}\\)"
    return " \\\\\n& ".join([f"\\({l}\\)" for l in lines])

# =============================================================================
# MAIN MERGE LOGIC
# =============================================================================
def merge_files():
    print(f"{'='*60}")
    print("DETECTOR MERGE UTILITY (Fixed Parser)")
    print(f"{'='*60}")
    
    # Find all readable files (exclude previous master files)
    files = [f for f in glob.glob("*_readable.txt") if "master" not in f]
    
    if not files:
        print("No *_readable.txt files found in directory.")
        return

    print(f"Found {len(files)} files to merge:")
    for f in files: print(f"  - {f}")
    
    # 1. Collect all raw entries
    all_data = []
    for fname in files:
        entries = parse_readable_file(fname)
        all_data.extend(entries)
    print(f"\nTotal raw entries: {len(all_data)}")
    
    if len(all_data) == 0:
        print("⚠️  Warning: Still found 0 entries. Check if input files are empty.")
        return
    
    # 2. Group by Power (p2, p3...)
    grouped = {}
    for item in all_data:
        p = item['power']
        grouped.setdefault(p, []).append(item)
        
    # 3. Process each power
    master_list = []
    
    print("\nProcessing Matches:")
    for p in sorted(grouped.keys()):
        raw_list = grouped[p]
        
        # Deduplicate
        unique_map = {}
        for item in raw_list:
            norm = item['norm']
            if norm not in unique_map:
                unique_map[norm] = item
            else:
                # Keep shortest representation
                if item['len'] < unique_map[norm]['len']:
                    unique_map[norm] = item
        
        unique_list = list(unique_map.values())
        
        # Sort: Length (Complexity) -> Alphabetical
        unique_list.sort(key=lambda x: (x['len'], x['formula']))
        
        print(f"  p^{p}: {len(raw_list)} raw -> {len(unique_list)} unique")
        
        # Assign new IDs
        for i, item in enumerate(unique_list):
            item['id'] = f"E{p}.{i+1}"
            master_list.append(item)

    # 4. Write Outputs
    
    # OUTPUT A: Master Readable File
    fn_read = "master_detectors_readable.txt"
    with open(fn_read, 'w') as f:
        f.write("MASTER DETECTOR LIST (Merged & Deduplicated)\n")
        f.write("============================================\n\n")
        
        current_p = None
        for item in master_list:
            if item['power'] != current_p:
                current_p = item['power']
                f.write(f"\n--- Detectors for p^{current_p} ---\n\n")
            
            f.write(f"{item['id']} [Len {item['len']}]\nL(n) = {item['formula']}\n\n")
            
    # OUTPUT B: Python List (for verification script)
    fn_py = "master_detectors_tuples.py"
    with open(fn_py, 'w') as f:
        f.write("# Master list for verification\n\n")
        f.write("all_detectors = [\n")
        for item in master_list:
            f.write(f'    ("E{item["power"]}.{item["id"].split(".")[1]}", "{item["formula"]}"),\n')
        f.write("]\n")
        
    # OUTPUT C: LaTeX Appendix
    fn_tex = "master_appendix.tex"
    with open(fn_tex, 'w') as f:
        f.write("% Auto-generated Appendix Table\n\n")
        
        current_p = None
        for item in master_list:
            if item['power'] != current_p:
                if current_p is not None:
                    f.write("\\bottomrule\n\\end{longtable}\n\n")
                
                current_p = item['power']
                f.write(f"\\subsection{{Detectors for $p^{current_p}$}}\n")
                f.write("\\begin{longtable}{l p{12cm}}\n")
                f.write("\\toprule\n\\textbf{ID} & \\textbf{Formula} \\\\\n\\midrule\n\\endhead\n")
            
            tex_eq = formula_to_latex(item['formula'])
            f.write(f"\\textbf{{{item['id']}}} & {tex_eq} \\\\\n")
            f.write(f"\\addlinespace[6pt] \\midrule[0.1pt]\n")
            
        if current_p is not None:
             f.write("\\bottomrule\n\\end{longtable}\n")

    print(f"\n✅ Successfully created:")
    print(f"  - {fn_read}")
    print(f"  - {fn_py}")
    print(f"  - {fn_tex}")

if __name__ == "__main__":
    merge_files()
