#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaussian Steps GUI — one-step or full workflow generator
(With per-step submit + formchk helpers)

What it does
------------
• Reads input .com files (or a folder/glob)
• Generates Gaussian .com + .sh per step (1..7)
• In FULL mode, links geometry via %oldchk + geom=check unless a step is
  selected for INLINE; inlining copies coords from your input .com
• Writes helper scripts in the output folder:
    - submit_all.sh                 (qsub/sbatch/bash <job>.sh for everything)
    - 01sub.sh, 02sub.sh, ...       (per-step submit)
    - 01formchk.sh, 02formchk.sh... (per-step formchk)

Run
---
python gaussian_steps_gui.py

Build for Windows
-----------------
IMPORTANT: RDKit must be installed before building the EXE!

1. Install RDKit:
   conda install -c conda-forge rdkit
   OR
   pip install rdkit

2. Build using the spec file (recommended):
   pyinstaller gaussian_steps_gui.spec

3. OR build directly (may miss RDKit dependencies):
   pyinstaller --noconfirm --onefile --windowed gaussian_steps_gui.py
   
Note: The spec file ensures RDKit is properly bundled. If RDKit is not
found in the EXE, use the spec file method.
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import List, Sequence, Tuple, Optional
import json, re, glob, os, stat, subprocess, sys

# Import TICT rotation module
try:
    from tict_rotation import generate_tict_rotations
    TICT_AVAILABLE = True
except ImportError:
    TICT_AVAILABLE = False

# Try to import RDKit for SMILES support
RDKIT_AVAILABLE = False
RDKIT_ERROR = None
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    # Test that RDKit actually works (not just imported)
    try:
        test_mol = Chem.MolFromSmiles("C")
        if test_mol is None:
            raise ImportError("RDKit imported but not functional")
        RDKIT_AVAILABLE = True
    except Exception as e:
        RDKIT_ERROR = f"RDKit imported but not functional: {str(e)}"
        RDKIT_AVAILABLE = False
except ImportError as e:
    RDKIT_ERROR = str(e)
    RDKIT_AVAILABLE = False
except Exception as e:
    RDKIT_ERROR = f"Unexpected error loading RDKit: {str(e)}"
    RDKIT_AVAILABLE = False

# Try to import XML parser for SVG files
try:
    import xml.etree.ElementTree as ET
    XML_AVAILABLE = True
except ImportError:
    XML_AVAILABLE = False

PREFS_FILE = Path.home() / ".gaussian_steps_gui.json"

# ===================== Backend logic (adapted from your script) =====================
# Defaults shown in GUI first; actual generation pulls from GUI state.
DEFAULTS = dict(
    MODE="single", STEP=4,
    INPUTS="", OUT_DIR="",
    INPUT_TYPE="com",  # "com", "log", or "smiles"
    SMILES_INPUT="",  # SMILES string input
    FUNCTIONAL="m062x", BASIS="def2SVP",
    SOLVENT_MODEL="SMD", SOLVENT_NAME="DMSO",
    TD_NSTATES=3, TD_ROOT=1,
    STATE_TYPE="singlet",  # "singlet", "triplet", or "mixed" (50-50)
    SOC_ENABLE=False,  # Enable PySOC preparation (saves RWF, adds 6D 10F GFInput)
    POP_FULL=False, DISPERSION=False,
    NPROC=64, MEM="128GB",
    SCHEDULER="pbs", QUEUE="normal", WALLTIME="24:00:00",
    PROJECT="15002108", ACCOUNT="",
    INLINE_STEPS=[], INLINE_SOURCE_5TO7=4,
    CHARGE=None, MULT=None,
    REMOVE_PREFIX="", REMOVE_SUFFIX="",
    MULTI_STEPS=[],  # For single mode with multiple steps
    MANUAL_ROUTES={},  # Per-step manual route overrides
    REDUNDANT_COORDS="",  # Redundant coordinates string (Step 4 only)
    GEOM_SOURCE={},  # Per-step geometry source: "coords_1", "coords_4", "oldchk_1", "oldchk_4", or None for default
)

# ---------------- utilities ----------------
def natural_key(s: str):
    parts = re.split(r"(\d+)", s.lower())
    return tuple(int(p) if p.isdigit() else p for p in parts)

def read_lines(p: Path) -> List[str]:
    return p.read_text(errors="ignore").splitlines()

def write_lines(p: Path, lines: Sequence[str]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="\n", encoding="utf-8") as f:
        for L in lines:
            f.write(str(L).rstrip("\r\n") + "\n")

def write_exec(p: Path, lines: Sequence[str]):
    if not lines:
        return
    write_lines(p, ["#!/bin/bash", *[str(L).rstrip("\r\n") for L in lines]])
    try:
        os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass

def find_geoms(pattern: str, input_type: str = "com") -> List[Path]:
    """
    Find geometry files based on input type.
    
    Args:
        pattern: File pattern or directory path
        input_type: "com", "log", or "smiles"
    
    Returns:
        List of file paths
    """
    if input_type == "smiles":
        return []  # SMILES are handled separately
    
    extension = ".log" if input_type == "log" else ".com"
    path = Path(pattern)
    if path.exists() and path.is_dir():
        files = sorted(path.glob(f"*{extension}"), key=lambda x: natural_key(x.name))
    else:
        files = [Path(m) for m in glob.glob(pattern)]
        files = sorted([p for p in files if p.is_file() and p.suffix.lower() == extension], key=lambda x: natural_key(x.name))
    return files

def remove_prefix_suffix(filename: str, prefix: str, suffix: str) -> str:
    """Remove prefix and/or suffix from filename (works with any extension)"""
    # Get just the filename without path
    filename = Path(filename).name
    
    # Get the base name and extension separately
    if '.' in filename:
        base_name, ext = filename.rsplit('.', 1)
        ext = '.' + ext
    else:
        base_name = filename
        ext = ''
    
    # Remove prefix from base name
    if prefix and prefix.strip():
        prefix_clean = prefix.strip()
        if base_name.startswith(prefix_clean):
            base_name = base_name[len(prefix_clean):]
    
    # Remove suffix from base name (before extension)
    if suffix and suffix.strip():
        suffix_clean = suffix.strip()
        # Try multiple strategies to remove suffix
        
        # Strategy 1: Exact match at end
        if base_name.endswith(suffix_clean):
            base_name = base_name[:-len(suffix_clean)]
        # Strategy 2: Suffix with underscore at end
        elif base_name.endswith('_' + suffix_clean):
            base_name = base_name[:-len('_' + suffix_clean)]
        # Strategy 3: Suffix appears anywhere in the middle - remove the suffix and its surrounding separators
        elif suffix_clean in base_name:
            idx = base_name.find(suffix_clean)
            if idx != -1:
                # Remove the suffix itself
                # Check if there's a separator before the suffix
                start_idx = idx
                if idx > 0 and base_name[idx-1] in ['_', '-']:
                    start_idx = idx - 1  # Include the separator before
                
                # Check if there's a separator after the suffix
                end_idx = idx + len(suffix_clean)
                if end_idx < len(base_name) and base_name[end_idx] in ['_', '-']:
                    end_idx = end_idx + 1  # Include the separator after
                
                # Remove the suffix and separators
                base_name = base_name[:start_idx] + base_name[end_idx:]
    
    # Return the cleaned name with extension
    return base_name + ext

def add_redundant_coords(coords: List[str], redundant: str, step: int) -> List[str]:
    """Add redundant coordinates only to Step 4, with blank line after coords"""
    if step != 4:
        return coords
    if not redundant or not redundant.strip():
        return coords
    # Remove trailing empty lines from coords
    while coords and not coords[-1].strip():
        coords = coords[:-1]
    # Append redundant coordinates (one per line, remove empty lines)
    redundant_lines = [line.strip() for line in redundant.strip().split('\n') if line.strip()]
    if redundant_lines:
        # Add blank line after coordinates, then redundant coords
        return coords + [""] + redundant_lines + [""]
    return coords + [""]

def extract_cm_coords(lines: List[str]) -> Tuple[str, List[str]]:
    empties = [i for i, L in enumerate(lines) if not L.strip()]
    coords = lines[empties[1]+1:] if len(empties) >= 2 else lines[:]
    atom_pat = re.compile(r"^[A-Za-z]{1,2}\s+[-\d]")
    cm = "0 1"
    for i, L in enumerate(coords):
        t = L.strip()
        if not t: continue
        if not atom_pat.match(t):
            if re.match(r"^-?\d+\s+-?\d+$", t):
                cm = t
                coords = coords[i+1:]
            break
    while coords and not coords[-1].strip():
        coords.pop()
    return cm, coords

def parse_gaussian_log(log_file: Path) -> Tuple[str, List[str], str]:
    """
    Parse a Gaussian log file to extract charge, multiplicity, and coordinates.
    Based on the MATLAB script logic.
    
    Args:
        log_file: Path to the Gaussian .log file
    
    Returns:
        Tuple of (charge_mult_string, coordinate_lines, base_name)
    """
    # Periodic table symbols (atomic number -> element symbol)
    # Index 0 is empty, so atomic number Z maps to index Z+1
    periodic_table = [
        '',  # 0
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
        'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
        'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
        'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
        'Pb', 'Bi', 'Po', 'At', 'Rn'
    ]
    
    log_file = Path(log_file)
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    # Read the entire file
    text = log_file.read_text(errors='ignore')
    lines = text.splitlines()
    
    # Extract charge and multiplicity - find last occurrence
    cm_pattern = re.compile(r'Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)', re.IGNORECASE)
    cm_matches = cm_pattern.findall(text)
    if not cm_matches:
        raise ValueError("Charge/Multiplicity not found in log file")
    
    # Use the last occurrence (final geometry)
    charge, multiplicity = cm_matches[-1]
    cm = f"{charge} {multiplicity}"
    
    # Find all "Standard orientation:" occurrences
    std_orientation_indices = []
    for i, line in enumerate(lines):
        if 'Standard orientation:' in line or 'standard orientation:' in line.lower():
            std_orientation_indices.append(i)
    
    if not std_orientation_indices:
        raise ValueError('"Standard orientation" not found in log file')
    
    # Use the last occurrence (final geometry)
    std_row = std_orientation_indices[-1]
    
    # Find the header line (usually 5 lines after "Standard orientation:")
    # Then find the separator line (starts with " ---")
    header_line = std_row + 5
    coords = []
    
    # Look for coordinate data
    for i in range(header_line, len(lines)):
        line = lines[i].strip()
        
        # Stop at separator line
        if line.startswith('---') or line.startswith('==='):
            # Check if we've already found coordinates
            if coords:
                break
            continue
        
        # Try to parse coordinate line: "  1   6   0   0.000000   0.000000   0.000000"
        # Format: Center Number, Atomic Number, Atomic Type, X, Y, Z
        # Use regex to match the pattern more robustly (handles various whitespace)
        coord_match = re.match(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', line)
        if coord_match:
            try:
                center_num = int(coord_match.group(1))
                atomic_num = int(coord_match.group(2))
                atomic_type = int(coord_match.group(3))
                x = float(coord_match.group(4))
                y = float(coord_match.group(5))
                z = float(coord_match.group(6))
                
                # Validate atomic number
                if atomic_num < 1 or atomic_num >= len(periodic_table):
                    continue
                
                element = periodic_table[atomic_num]
                if not element:
                    raise ValueError(f"Atomic number {atomic_num} not in symbol table")
                
                # Format as Gaussian coordinate: "Element   X   Y   Z"
                coords.append(f"{element:>2s}  {x:15.10f}  {y:15.10f}  {z:15.10f}")
            except (ValueError, IndexError):
                # Not a coordinate line, continue
                continue
    
    if not coords:
        raise ValueError("Coordinate table empty or unreadable")
    
    # Generate base name from log filename
    base_name = log_file.stem  # filename without extension
    
    return cm, coords, base_name

def parse_smiles_line(line: str) -> Tuple[Optional[str], str]:
    """
    Parse a line that may contain a name and SMILES.
    Supports formats:
    - "name:SMILES"
    - "name\tSMILES" (tab-separated)
    - "SMILES" (just SMILES, no name)
    - "# comment" (comment line, returns None, None)
    
    Returns:
        Tuple of (name_or_none, smiles) or (None, None) for comments
    """
    line = line.strip()
    
    # Skip empty lines and comments
    if not line or line.startswith('#'):
        return None, None
    
    # Try colon separator first (most common)
    if ':' in line and not line.startswith('http'):  # Avoid URLs
        parts = line.split(':', 1)
        if len(parts) == 2:
            name = parts[0].strip()
            smiles = parts[1].strip()
            if name and smiles:
                return name, smiles
    
    # Try tab separator (ChemDraw often uses this format)
    if '\t' in line:
        parts = [p.strip() for p in line.split('\t') if p.strip()]
        if len(parts) >= 2:
            # ChemDraw sometimes exports: Name \t Other \t SMILES
            # Try to find the SMILES (usually contains special chars like ()[]=+-)
            potential_smiles = parts[-1]  # Last non-empty part is usually SMILES
            potential_name = parts[0] if len(parts) > 1 else None
            
            # Validate: SMILES usually contains parentheses, brackets, or numbers
            if any(c in potential_smiles for c in '()[]=+-0123456789'):
                if potential_name and potential_name != potential_smiles and len(potential_name) < 50:
                    return potential_name, potential_smiles
                else:
                    return None, potential_smiles
        elif len(parts) == 1:
            # Single tab-separated value, treat as SMILES
            return None, parts[0]
    
    # Just SMILES, no name
    return None, line

def smiles_to_coords(smiles: str, charge: Optional[int] = None, mult: Optional[int] = None, custom_name: Optional[str] = None) -> Tuple[str, List[str], str]:
    """
    Convert SMILES string to 3D coordinates in Gaussian format.
    
    Args:
        smiles: SMILES string
        charge: Optional charge (auto-detected if None)
        mult: Optional multiplicity (defaults to 1 if None)
        custom_name: Optional custom name to use instead of auto-generated
    
    Returns:
        Tuple of (charge_mult_string, coordinate_lines, base_name)
    """
    if not RDKIT_AVAILABLE:
        error_msg = "RDKit is required for SMILES input."
        if RDKIT_ERROR:
            error_msg += f"\n\nError: {RDKIT_ERROR}"
        error_msg += "\n\nInstall with: conda install -c conda-forge rdkit"
        error_msg += "\nOR: pip install rdkit"
        raise ImportError(error_msg)
    
    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Get charge and multiplicity
    if charge is None:
        # Try to calculate formal charge
        charge = Chem.rdmolops.GetFormalCharge(mol)
    if mult is None:
        # Default to singlet (multiplicity 1)
        mult = 1
    
    cm = f"{charge} {mult}"
    
    # Convert to Gaussian coordinate format
    coords = []
    conf = mol.GetConformer()
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        symbol = atom.GetSymbol()
        coords.append(f"{symbol:>2s}  {pos.x:15.10f}  {pos.y:15.10f}  {pos.z:15.10f}")
    
    # Generate base name
    if custom_name:
        # Use provided name, sanitize for filename
        base_name = re.sub(r'[^\w\-_]', '_', custom_name)
    else:
        # Generate a generic name instead of using SMILES string
        # This will be numbered by the caller if needed
        base_name = "molecule"
    
    return cm, coords, base_name

def parse_svg_transform(transform_str: str) -> Tuple[float, float]:
    """Parse SVG transform attribute to extract x, y coordinates"""
    if not transform_str:
        return 0.0, 0.0
    
    x, y = 0.0, 0.0
    
    # Look for matrix(a b c d e f) - e and f are the translation (x, y)
    matrix_match = re.search(r'matrix\s*\(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)', transform_str)
    if matrix_match:
        # In SVG matrix, the 5th and 6th values (indices 4 and 5) are the translation
        x = float(matrix_match.group(5))
        y = float(matrix_match.group(6))
        return x, y
    
    # Look for translate(x, y) or translateX(x) translateY(y)
    translate_match = re.search(r'translate\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)', transform_str)
    if translate_match:
        x = float(translate_match.group(1))
        y = float(translate_match.group(2))
        return x, y
    
    # Try separate translateX and translateY
    tx_match = re.search(r'translateX\s*\(\s*([-\d.]+)\s*\)', transform_str)
    ty_match = re.search(r'translateY\s*\(\s*([-\d.]+)\s*\)', transform_str)
    if tx_match:
        x = float(tx_match.group(1))
    if ty_match:
        y = float(ty_match.group(1))
    
    return x, y

def extract_names_from_svg(file_path: Path) -> List[str]:
    """
    Extract text labels/names from an SVG file exported from ChemDraw.
    Uses spatial coordinates to match names with molecules accurately.
    Names are sorted by position (top to bottom, left to right) to match molecule order.
    
    Returns:
        List of names in spatial order (matching molecule order)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not XML_AVAILABLE:
        raise ImportError("XML parser not available")
    
    text_items = []  # List of (text, x, y) tuples
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # SVG namespace
        svg_ns = 'http://www.w3.org/2000/svg'
        
        # Find all text elements in SVG
        text_elements = root.findall('.//{%s}text' % svg_ns) or root.findall('.//text')
        tspan_elements = root.findall('.//{%s}tspan' % svg_ns) or root.findall('.//tspan')
        
        all_text_elements = text_elements + tspan_elements
        
        # Common chemical abbreviations to filter out
        common_abbrevs = {
            'COOH', 'Boc', 'CN', 'NO2', 'NH2', 'OH', 'Me', 'Et', 'Pr', 'Bu', 
            'Ph', 'Bn', 'Ac', 'Ts', 'Ms', 'Tf', 'Bz', 'Piv', 'Cbz', 'Fmoc',
            'OMe', 'OEt', 'NMe2', 'SO2', 'SO3', 'PO4', 'CO2', 'NHBoc', 'BocHN'
        }
        
        for text_elem in all_text_elements:
            # Get text content
            text_content = text_elem.text
            if not text_content:
                # Sometimes text is in tail or in child tspan
                text_content = text_elem.tail or ''
                # Check for nested tspan
                for child in text_elem:
                    if child.tag.endswith('tspan') or child.tag == 'tspan':
                        if child.text:
                            text_content = child.text
                            break
            
            text_str = text_content.strip() if text_content else ''
            
            if text_str and len(text_str) > 1:
                # Filter out common abbreviations
                if text_str not in common_abbrevs and len(text_str) > 2:
                    # Keep text that looks like compound names
                    if any(c.isalpha() for c in text_str):
                        if not (len(text_str) == 1 and text_str in '+-=()[]{}'):
                            # Extract position
                            x, y = 0.0, 0.0
                            
                            # Try to get x, y from attributes
                            if 'x' in text_elem.attrib:
                                try:
                                    x = float(text_elem.attrib['x'])
                                except (ValueError, TypeError):
                                    pass
                            if 'y' in text_elem.attrib:
                                try:
                                    y = float(text_elem.attrib['y'])
                                except (ValueError, TypeError):
                                    pass
                            
                            # Try to get from transform attribute (handles matrix() and translate())
                            if 'transform' in text_elem.attrib:
                                tx, ty = parse_svg_transform(text_elem.attrib['transform'])
                                # Transform matrix/translate gives absolute position, so use it directly
                                x = tx
                                y = ty
                            
                            text_items.append((text_str, x, y))
        
        # Also check for text in title/desc attributes
        for elem in root.iter():
            for attr in ['title', 'id', 'data-name']:
                if attr in elem.attrib:
                    text_str = elem.attrib[attr].strip()
                    if text_str and len(text_str) > 2 and any(c.isalpha() for c in text_str):
                        if text_str not in common_abbrevs:
                            # Try to get position from element
                            x, y = 0.0, 0.0
                            if 'x' in elem.attrib:
                                try:
                                    x = float(elem.attrib['x'])
                                except (ValueError, TypeError):
                                    pass
                            if 'y' in elem.attrib:
                                try:
                                    y = float(elem.attrib['y'])
                                except (ValueError, TypeError):
                                    pass
                            text_items.append((text_str, x, y))
        
        if not text_items:
            return []
        
        # Sort by position: top to bottom (y coordinate), then left to right (x coordinate)
        # In SVG, y increases downward, so smaller y = higher on page
        text_items.sort(key=lambda item: (item[2], item[1]))  # Sort by y, then x
        
        # Extract just the names in sorted order
        names = [text for text, x, y in text_items]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)
        
        return unique_names
    
    except ET.ParseError as e:
        raise ValueError(f"Error parsing SVG file: {str(e)}\n\nMake sure the file is a valid SVG.")
    except Exception as e:
        raise ValueError(f"Could not extract names from SVG: {str(e)}")

def build_scrf(model: str, solvent: str, tail: str = "") -> str:
    if (model or "").lower() in ("", "none", "vac", "vacuum"): return ""
    parts = [model]
    if solvent: parts.append(f"solvent={solvent}")
    if tail:    parts.append(tail)
    return ", ".join(parts)

def route_line(method: str, basis: str, td: str = "", scrf: str = "", extras: str = "", soc_enable: bool = False) -> str:
    # TD can be in format: "TD(...)" or just the content for TD=(...)
    if td:
        # Check if td already contains "TD(" format
        if td.startswith("TD("):
            td_part = f" {td}"
        else:
            td_part = f" TD=({td})"
    else:
        td_part = ""
    scrf_part = f" SCRF=({scrf})" if scrf else ""
    # Add 6D 10F GFInput for PySOC compatibility (required for basis set printing)
    soc_keywords = " 6D 10F GFInput" if soc_enable else ""
    base = f"# {method}/{basis}{td_part}{scrf_part}{soc_keywords}"
    if extras: base += f" {extras}"
    return re.sub(r"\s+", " ", base).strip()

def cm_override(parsed_cm: str, charge, mult) -> str:
    if charge is None or mult is None:
        return parsed_cm
    return f"{int(charge)} {int(mult)}"

def make_com_inline(job: str, nproc: int, mem: str, route: str, title: str, cm: str, coords: List[str], save_rwf: bool = False) -> List[str]:
    """Create inline .com file. If save_rwf=True, adds %rwf line for PySOC compatibility."""
    lines = [f"%nprocshared={nproc}", f"%mem={mem}", f"%chk={job}.chk"]
    if save_rwf:
        lines.append(f"%rwf={job}.rwf")
    lines.extend([route, "", title, "", cm, *coords, "", ""])
    return lines

def make_com_linked(job: str, nproc: int, mem: str, oldchk: str, route: str, title: str, cm: str, save_rwf: bool = False) -> List[str]:
    """Create linked .com file. If save_rwf=True, adds %rwf line for PySOC compatibility."""
    lines = [f"%nprocshared={nproc}", f"%mem={mem}", f"%oldchk={oldchk}", f"%chk={job}.chk"]
    if save_rwf:
        lines.append(f"%rwf={job}.rwf")
    lines.extend([route, "", title, "", cm, "", ""])
    return lines

def pbs_script(job: str, cfg) -> List[str]:
    return [
        "#!/bin/bash",
        f"#PBS -q {cfg['QUEUE']}",
        f"#PBS -N {job}",
        f"#PBS -l select=1:ncpus={cfg['NPROC']}:mpiprocs={cfg['NPROC']}:mem={cfg['MEM']}",
        (f"#PBS -l walltime={cfg['WALLTIME']}" if cfg['WALLTIME'] else "").strip(),
        (f"#PBS -P {cfg['PROJECT']}" if cfg['PROJECT'] else "").strip(),
        f"#PBS -o {job}.o", f"#PBS -e {job}.e",
        "cd $PBS_O_WORKDIR",
        f"g16 < {job}.com > {job}.log",
    ]

def slurm_script(job: str, cfg) -> List[str]:
    return [
        "#!/bin/bash",
        f"#SBATCH -J {job}",
        f"#SBATCH -p {cfg['QUEUE']}",
        "#SBATCH -N 1",
        f"#SBATCH --ntasks={cfg['NPROC']}",
        f"#SBATCH --mem={cfg['MEM']}",
        (f"#SBATCH -t {cfg['WALLTIME']}" if cfg['WALLTIME'] else "").strip(),
        (f"#SBATCH -A {cfg['ACCOUNT']}" if cfg['ACCOUNT'] else "").strip(),
        f"#SBATCH -o {job}.out", f"#SBATCH -e {job}.err",
        f"g16 < {job}.com > {job}.log",
    ]

def local_script(job: str, cfg) -> List[str]:
    return ["#!/bin/bash", f"g16 < {job}.com > {job}.log &"]

def write_sh(job: str, cfg) -> List[str]:
    if cfg['SCHEDULER'] == "pbs":   return pbs_script(job, cfg)
    if cfg['SCHEDULER'] == "slurm": return slurm_script(job, cfg)
    return local_script(job, cfg)

# ---------- routes per step ----------

def td_block(cfg):
    """
    Generate TD block based on state type.
    For singlet: TD(NStates=n, Root=r) - default, closed shell
    For triplet: TD(Triplets, NStates=n) - explicit triplet states
    For mixed: TD(50-50, NStates=n) - closed shell singlet-triplet mixed
    
    Note: If SOC_ENABLE is True, automatically uses 50-50 for singlet states
    (required for PySOC calculations which need both singlet and triplet states)
    """
    state_type = cfg.get('STATE_TYPE', 'singlet').lower()
    nstates = int(cfg['TD_NSTATES'])
    root = int(cfg['TD_ROOT'])
    soc_enable = cfg.get('SOC_ENABLE', False)
    
    # SOC calculations require 50-50 (mixed singlet-triplet) for closed shell systems
    if soc_enable and state_type == 'singlet':
        # Override to 50-50 when SOC is enabled (required for SOC)
        # Match PySOC example format: TD(50-50,nstates=5)
        return f"TD(50-50,nstates={nstates})"
    
    if state_type == 'triplet':
        # For triplet states: TD(Triplets, NStates=n)
        return f"TD(Triplets, NStates={nstates})"
    elif state_type == 'mixed':
        # Mixed singlet-triplet: TD(50-50, NStates=n)
        return f"TD(50-50, NStates={nstates})"
    else:  # singlet (default, closed shell)
        # For singlet: TD(NStates=n, Root=r)
        return f"TD(NStates={nstates}, Root={root})"

def pop_kw(cfg):
    return " pop=(full,orbitals=2,threshorbitals=1)" if cfg['POP_FULL'] else ""

def disp_kw(cfg):
    return " EmpiricalDispersion=GD3BJ" if cfg['DISPERSION'] else ""

def scrf(cfg):
    return build_scrf(cfg['SOLVENT_MODEL'], cfg['SOLVENT_NAME'])

def scrf_clr(cfg):
    return build_scrf(cfg['SOLVENT_MODEL'], cfg['SOLVENT_NAME'], "CorrectedLR")

def route_step1(cfg): 
    manual = cfg.get('MANUAL_ROUTES', {}).get(1, "").strip()
    if manual: return manual
    soc_enable = cfg.get('SOC_ENABLE', False)
    return route_line(cfg['FUNCTIONAL'], cfg['BASIS'], scrf=scrf(cfg), extras=f"Opt Freq{pop_kw(cfg)}{disp_kw(cfg)}", soc_enable=soc_enable)

def route_step2(cfg): 
    manual = cfg.get('MANUAL_ROUTES', {}).get(2, "").strip()
    if manual: return manual
    soc_enable = cfg.get('SOC_ENABLE', False)
    return route_line(cfg['FUNCTIONAL'], cfg['BASIS'], td=td_block(cfg), scrf=scrf(cfg), extras=f"{pop_kw(cfg)}{disp_kw(cfg)}", soc_enable=soc_enable)

def route_step3(cfg): 
    manual = cfg.get('MANUAL_ROUTES', {}).get(3, "").strip()
    if manual: return manual
    soc_enable = cfg.get('SOC_ENABLE', False)
    return route_line(cfg['FUNCTIONAL'], cfg['BASIS'], td=td_block(cfg), scrf=scrf_clr(cfg), extras=f"{pop_kw(cfg)}{disp_kw(cfg)}", soc_enable=soc_enable)

def route_step4(cfg): 
    manual = cfg.get('MANUAL_ROUTES', {}).get(4, "").strip()
    if manual: return manual
    soc_enable = cfg.get('SOC_ENABLE', False)
    # For PySOC, don't include Opt and Freq keywords
    if soc_enable:
        extras = f"{disp_kw(cfg)}"
    else:
        extras = f"Opt=CalcFC Freq{disp_kw(cfg)}"
    return route_line(cfg['FUNCTIONAL'], cfg['BASIS'], td=td_block(cfg), scrf=scrf(cfg), extras=extras, soc_enable=soc_enable)

def route_step5(cfg): 
    manual = cfg.get('MANUAL_ROUTES', {}).get(5, "").strip()
    if manual: return manual
    soc_enable = cfg.get('SOC_ENABLE', False)
    return route_line(cfg['FUNCTIONAL'], cfg['BASIS'], scrf=scrf(cfg), extras=f"density{pop_kw(cfg)}{disp_kw(cfg)}", soc_enable=soc_enable)

def route_step6(cfg): 
    manual = cfg.get('MANUAL_ROUTES', {}).get(6, "").strip()
    if manual: return manual
    soc_enable = cfg.get('SOC_ENABLE', False)
    return route_line(cfg['FUNCTIONAL'], cfg['BASIS'], td=td_block(cfg), scrf=scrf_clr(cfg), extras=f"{disp_kw(cfg)}", soc_enable=soc_enable)

def route_step7(cfg): 
    manual = cfg.get('MANUAL_ROUTES', {}).get(7, "").strip()
    if manual: return manual
    soc_enable = cfg.get('SOC_ENABLE', False)
    return route_line(cfg['FUNCTIONAL'], cfg['BASIS'], td=td_block(cfg), scrf=scrf(cfg), extras=f"{pop_kw(cfg)}{disp_kw(cfg)}", soc_enable=soc_enable)

# ---------- generation ----------

def solv_tag(model: str, name: str) -> str:
    return "vac" if (model or "").lower() in ("none","","vac","vacuum") else (name.lower() if name else "solv")

def jobname(step_num: int, base: str, cfg) -> str:
    return f"{step_num:02d}{base}_{cfg['FUNCTIONAL']}_{cfg['BASIS']}_{solv_tag(cfg['SOLVENT_MODEL'], cfg['SOLVENT_NAME'])}"

def step_route(step: int, cfg):
    return {1: route_step1, 2: route_step2, 3: route_step3, 4: route_step4, 5: route_step5, 6: route_step6, 7: route_step7}[step](cfg)

def generate_single(base: str, cm: str, coords: List[str], out_path: Path, cfg) -> List[str]:
    steps = cfg.get('MULTI_STEPS', [])
    if not steps:
        steps = [int(cfg['STEP'])]
    
    jobs = []
    # Get state type for titles
    state_type = cfg.get('STATE_TYPE', 'singlet').lower()
    state_label = "Triplet" if state_type == 'triplet' else ("Mixed" if state_type == 'mixed' else "Singlet")
    titles = {
        1: "Step1 GS Opt",
        2: f"Step2 {state_label} Abs",
        3: f"Step3 {state_label} Abs cLR",
        4: f"Step4 {state_label} ES Opt",
        5: "Step5 Density",
        6: f"Step6 {state_label} ES cLR",
        7: f"Step7 {state_label} De-excitation"
    }
    geom_sources = cfg.get('GEOM_SOURCE', {})
    J = lambda k: jobname(k, base, cfg)
    cm_use = cm_override(cm, cfg['CHARGE'], cfg['MULT'])
    
    for step in steps:
        job = J(step)
        source = geom_sources.get(step)
        
        # Step 7 always uses oldchk from Step 6
        if step == 7:
            r = step_route(step, cfg) + " geom=check guess=read"
            save_rwf = cfg.get('SOC_ENABLE', False)
            com = make_com_linked(job, cfg['NPROC'], cfg['MEM'], f"{J(6)}.chk", r, 
                                 f"{titles[step]} {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm_use, save_rwf=save_rwf)
        elif source == 'oldchk_1':
            r = step_route(step, cfg) + " geom=check guess=read"
            save_rwf = cfg.get('SOC_ENABLE', False)
            com = make_com_linked(job, cfg['NPROC'], cfg['MEM'], f"{J(1)}.chk", r, 
                                 f"{titles[step]} {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm_use, save_rwf=save_rwf)
        elif source == 'oldchk_4':
            r = step_route(step, cfg) + " geom=check guess=read"
            save_rwf = cfg.get('SOC_ENABLE', False)
            com = make_com_linked(job, cfg['NPROC'], cfg['MEM'], f"{J(4)}.chk", r, 
                                 f"{titles[step]} {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm_use, save_rwf=save_rwf)
        else:
            # Default: inline coordinates (coords_1, coords_4, or default)
            coords_use = add_redundant_coords(coords.copy(), cfg.get('REDUNDANT_COORDS', ""), step)
            save_rwf = cfg.get('SOC_ENABLE', False)
            com = make_com_inline(job, cfg['NPROC'], cfg['MEM'], step_route(step, cfg), 
                                  f"{titles[step]} {cfg['FUNCTIONAL']}/{cfg['BASIS']}", 
                                  cm_use, coords_use, save_rwf=save_rwf)
        write_lines(out_path / f"{job}.com", com)
        write_lines(out_path / f"{job}.sh", write_sh(job, cfg))
        jobs.append(job)
    
    return jobs

def generate_full(base: str, cm_in: str, coords_in: List[str], out_path: Path, cfg) -> List[str]:
    jobs = []
    J = lambda k: jobname(k, base, cfg)
    cm_use = cm_override(cm_in, cfg['CHARGE'], cfg['MULT'])
    geom_sources = cfg.get('GEOM_SOURCE', {})

    # Step 1 inline
    j01 = J(1)
    coords_01 = add_redundant_coords(coords_in.copy(), cfg.get('REDUNDANT_COORDS', ""), 1)
    save_rwf = cfg.get('SOC_ENABLE', False)
    com01 = make_com_inline(j01, cfg['NPROC'], cfg['MEM'], route_step1(cfg), f"Step1 GS Opt {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm_use, coords_01, save_rwf=save_rwf)
    write_lines(out_path / f"{j01}.com", com01); write_lines(out_path / f"{j01}.sh", write_sh(j01, cfg)); jobs.append(j01)

    def get_geometry_source(step: int) -> Tuple[str, List[str], str]:
        """Returns (cm, coords, oldchk_path)"""
        # Step 7 always uses oldchk from Step 6
        if step == 7:
            return cm_use, None, f"{J(6)}.chk"
        
        source = geom_sources.get(step)
        if source == 'coords_1':
            coords_use = add_redundant_coords(coords_in.copy(), cfg.get('REDUNDANT_COORDS', ""), step)
            return cm_use, coords_use, None
        elif source == 'coords_4':
            coords_use = add_redundant_coords(coords_in.copy(), cfg.get('REDUNDANT_COORDS', ""), step)
            return cm_use, coords_use, None
        elif source == 'oldchk_1':
            return cm_use, None, f"{j01}.chk"
        elif source == 'oldchk_4':
            return cm_use, None, f"{J(4)}.chk"
        else:
            # Default behavior
            if step in set(cfg.get('INLINE_STEPS', []) or []):
                coords_use = add_redundant_coords(coords_in.copy(), cfg.get('REDUNDANT_COORDS', ""), step)
                return cm_use, coords_use, None
            else:
                reader = 1 if step < 5 else 4
                return cm_use, None, f"{J(reader)}.chk"

    # Get state type for titles
    state_type = cfg.get('STATE_TYPE', 'singlet').lower()
    state_label = "Triplet" if state_type == 'triplet' else ("Mixed" if state_type == 'mixed' else "Singlet")
    
    for k, default_reader in [
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 4),
        (6, 4),
        (7, 4),
    ]:
        # Generate title based on step and state type
        if k == 1:
            title = f"Step1 GS Opt {cfg['FUNCTIONAL']}/{cfg['BASIS']}"
        elif k == 5:
            title = "Step5 Density"
        else:
            step_names = {2: "Abs", 3: "Abs cLR", 4: "ES Opt", 6: "ES cLR", 7: "De-excitation"}
            title = f"Step{k} {state_label} {step_names[k]}"
        
        jk = J(k)
        cmk, ck, oldchk = get_geometry_source(k)
        if ck is not None:
            # Use the dynamically generated title
            title_full = f"{title} {cfg['FUNCTIONAL']}/{cfg['BASIS']}" if k != 1 else title
            save_rwf = cfg.get('SOC_ENABLE', False)
            comk = make_com_inline(jk, cfg['NPROC'], cfg['MEM'], step_route(k, cfg), 
                                  title_full, cmk, ck, save_rwf=save_rwf)
        else:
            r = step_route(k, cfg) + " geom=check guess=read"
            save_rwf = cfg.get('SOC_ENABLE', False)
            comk = make_com_linked(jk, cfg['NPROC'], cfg['MEM'], oldchk, r, 
                                  f"{title} {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cmk, save_rwf=save_rwf)
        write_lines(out_path / f"{jk}.com", comk); write_lines(out_path / f"{jk}.sh", write_sh(jk, cfg)); jobs.append(jk)

    return jobs

# ===================== GUI =====================
class EditableCombo(ttk.Frame):
    def __init__(self, parent, values=None):
        super().__init__(parent)
        self.values = list(values or [])
        self.var = tk.StringVar()
        self.combo = ttk.Combobox(self, textvariable=self.var, values=self.values)
        self.combo.pack(fill='x', expand=True)
        self.combo.bind('<KeyRelease>', self._filter)
        self.combo.bind('<FocusOut>', self._add_if_new)
    def _filter(self, _):
        txt = self.var.get()
        if txt:
            filt = [v for v in self.values if txt.lower() in v.lower()]
            self.combo['values'] = filt if filt else self.values
        else:
            self.combo['values'] = self.values
    def _add_if_new(self, _):
        txt = self.var.get().strip()
        if txt and txt not in self.values:
            self.values.append(txt)
            self.combo['values'] = self.values
    def get(self): return self.var.get().strip()
    def set(self, v): self.var.set(v)

class App:
    def __init__(self, root):
        self.root = root
        root.title("Gaussian Steps Generator")
        root.geometry("1200x900")
        self.vars = {}
        self.multi_step_vars = {}
        self.step_route_texts = {}
        self.step_route_frames = {}
        self.step_geom_source_vars = {}
        self._setup_colors()
        self._styles()
        self._create_header()
        self._tabs()
        self._load_prefs()

    def _setup_colors(self):
        """Setup color scheme"""
        self.colors = {
            'bg': '#f5f5f5',
            'card': '#ffffff',
            'text': '#2c3e50',
            'text_light': '#7f8c8d',
            'border': '#e0e0e0',
            'primary': '#4a90e2',
            'secondary': '#7b68ee',
            'accent': '#50c878',
            'hover': '#e8f4f8',
        }

    def _styles(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('Header.TLabel', font=('Segoe UI', 13, 'bold'))
        s.configure('Card.TLabelframe', padding=10)
        s.configure('Modern.TButton', font=('Segoe UI', 9), padding=(8, 4))
        self.root.configure(bg=self.colors['bg'])

    def _create_header(self):
        """Create header"""
        self.header_frame = tk.Frame(self.root, bg=self.colors['bg'], height=50)
        self.header_frame.pack(fill='x', padx=10, pady=5)
        title = tk.Label(self.header_frame, text="🚀 Gaussian Steps Generator", 
                        font=('Segoe UI', 16, 'bold'), bg=self.colors['bg'], 
                        fg=self.colors['primary'])
        title.pack(side='left', padx=10)

    def _tab(self, nb, name):
        f = ttk.Frame(nb); nb.add(f, text=name); return f

    def _section(self, parent, title):
        lf = ttk.Labelframe(parent, text=title, style='Card.TLabelframe')
        lf.pack(fill='x', padx=8, pady=8)
        return lf

    def _tabs(self):
        nb = ttk.Notebook(self.root); nb.pack(fill='both', expand=True, padx=8, pady=8)
        self._tab_main(nb)
        self._tab_advanced(nb)
        self._tab_tict(nb)
        self._tab_generate(nb)

    def _create_card(self, parent, title):
        """Create a styled card frame"""
        card = tk.Frame(parent, bg=self.colors['card'], relief='flat', borderwidth=1,
                       highlightbackground=self.colors['border'], highlightthickness=1)
        if title:
            tk.Label(card, text=title, font=('Segoe UI', 10, 'bold'),
                    bg=self.colors['card'], fg=self.colors['text']).pack(anchor='w', padx=10, pady=(8,5))
        return card

    def _tab_main(self, nb):
        f = tk.Frame(nb, bg=self.colors['bg'])
        nb.add(f, text='⚙️ Main Settings')
        
        # Scrollable container
        canvas = tk.Canvas(f, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(f, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg'])
        
        def update_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        def on_canvas_configure(event):
            canvas_width = event.width
            if canvas.find_all():
                canvas.itemconfig(canvas.find_all()[0], width=canvas_width)
        
        scrollable_frame.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", on_canvas_configure)
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Enable mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Mode and Steps Section
        mode_card = self._create_card(scrollable_frame, 'Mode & Steps')
        mode_card.pack(fill='x', padx=15, pady=8)
        
        mode_row = tk.Frame(mode_card, bg=self.colors['card'])
        mode_row.pack(fill='x', padx=10, pady=(0,10))
        tk.Label(mode_row, text='Mode:', font=('Segoe UI', 10), 
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        
        self.vars['MODE'] = tk.StringVar(value='single')
        mode_buttons_frame = tk.Frame(mode_row, bg=self.colors['card'])
        mode_buttons_frame.pack(side='left')
        
        for label, value in [('Full (1-7)', 'full'), ('Single', 'single'), ('Multiple', 'multiple')]:
            btn = tk.Button(mode_buttons_frame, text=label, font=('Segoe UI', 9, 'bold'),
                          width=10, height=1, bg=self.colors['card'], fg=self.colors['text'],
                          relief='raised', borderwidth=1, command=lambda v=value: self._set_mode(v),
                          cursor='hand2')
            btn.pack(side='left', padx=3)
            setattr(self, f'_mode_btn_{value}', btn)
        # Store mode buttons frame for disabling
        self.mode_buttons_frame = mode_buttons_frame
        self.mode_card = mode_card
        
        step_row = tk.Frame(mode_card, bg=self.colors['card'])
        step_row.pack(fill='x', padx=10, pady=(0,10))
        tk.Label(step_row, text='Steps:', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        
        self.single_step_frame = tk.Frame(step_row, bg=self.colors['card'])
        self.single_step_frame.pack(side='left')
        self.vars['STEP'] = tk.IntVar(value=DEFAULTS['STEP'])
        step_buttons = tk.Frame(self.single_step_frame, bg=self.colors['card'])
        step_buttons.pack(side='left')
        for k in (1,2,3,4,5,6,7):
            btn = tk.Button(step_buttons, text=str(k), width=3, height=1,
                          font=('Segoe UI', 9), bg=self.colors['card'], fg=self.colors['text'],
                          relief='raised', borderwidth=1, command=lambda v=k: self._set_single_step(v),
                          cursor='hand2')
            btn.pack(side='left', padx=2)
            setattr(self, f'_step_btn_{k}', btn)
        
        self.multi_step_frame = tk.Frame(step_row, bg=self.colors['card'])
        self.multi_step_vars = {}
        multi_buttons = tk.Frame(self.multi_step_frame, bg=self.colors['card'])
        multi_buttons.pack(side='left')
        for k in (1,2,3,4,5,6,7):
            var = tk.BooleanVar()
            self.multi_step_vars[k] = var
            btn = tk.Button(multi_buttons, text=str(k), width=3, height=1,
                          font=('Segoe UI', 9), bg=self.colors['card'], fg=self.colors['text'],
                          relief='raised', borderwidth=1, command=lambda v=k: self._toggle_multi_step(v),
                          cursor='hand2')
            btn.pack(side='left', padx=2)
            setattr(self, f'_multi_btn_{k}', btn)
        self.multi_step_frame.pack_forget()
        # Store step row for disabling
        self.step_row = step_row
        self._update_mode_buttons()
        
        # Two-column layout
        content_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        content_frame.pack(fill='both', expand=True, padx=15, pady=5)
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        
        # Left Column
        left_col = tk.Frame(content_frame, bg=self.colors['bg'])
        left_col.grid(row=0, column=0, sticky='nsew', padx=(0,8))
        
        # Input/Output
        io_card = self._create_card(left_col, 'Input/Output Files')
        io_card.pack(fill='x', pady=8)
        
        # Input type selection
        input_type_row = tk.Frame(io_card, bg=self.colors['card'])
        input_type_row.pack(fill='x', padx=10, pady=(0,5))
        tk.Label(input_type_row, text='Input Type:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        if 'INPUT_TYPE' not in self.vars:
            self.vars['INPUT_TYPE'] = tk.StringVar(value=DEFAULTS['INPUT_TYPE'])
        input_type_frame = tk.Frame(input_type_row, bg=self.colors['card'])
        input_type_frame.pack(side='left')
        tk.Radiobutton(input_type_frame, text='.com Files', variable=self.vars['INPUT_TYPE'], value='com',
                      font=('Segoe UI', 9), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text'], command=self._on_input_type_change).pack(side='left', padx=5)
        tk.Radiobutton(input_type_frame, text='.log Files', variable=self.vars['INPUT_TYPE'], value='log',
                      font=('Segoe UI', 9), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text'], command=self._on_input_type_change).pack(side='left', padx=5)
        tk.Radiobutton(input_type_frame, text='SMILES', variable=self.vars['INPUT_TYPE'], value='smiles',
                      font=('Segoe UI', 9), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text'], command=self._on_input_type_change).pack(side='left', padx=5)
        if not RDKIT_AVAILABLE:
            error_text = '(RDKit not available)'
            if RDKIT_ERROR and len(RDKIT_ERROR) < 50:
                error_text = f'(RDKit: {RDKIT_ERROR[:40]}...)'
            tk.Label(input_type_frame, text=error_text, font=('Segoe UI', 7, 'italic'),
                    bg=self.colors['card'], fg='red').pack(side='left', padx=5)
        
        # .com/.log file input row
        input_row = tk.Frame(io_card, bg=self.colors['card'])
        input_row.pack(fill='x', padx=10, pady=(0,8))
        input_row.columnconfigure(0, weight=1)
        self.input_label = tk.Label(input_row, text='Input (.com files or folder):', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text'])
        self.input_label.grid(row=0, column=0, sticky='w', pady=3)
        self.vars['INPUTS'] = tk.StringVar(value=DEFAULTS['INPUTS'])
        self.input_entry = tk.Entry(input_row, textvariable=self.vars['INPUTS'], font=('Segoe UI', 9),
                         bg='white', fg=self.colors['text'], relief='flat', borderwidth=1,
                         highlightthickness=1, highlightbackground=self.colors['border'])
        self.input_entry.grid(row=1, column=0, sticky='ew', padx=(0,5))
        self.browse_btn = tk.Button(input_row, text='Browse', command=self._browse_inputs,
                 font=('Segoe UI', 8), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=10, pady=3, cursor='hand2')
        self.browse_btn.grid(row=1, column=1)
        self.input_row = input_row  # Store reference for show/hide
        
        # SMILES input row
        smiles_row = tk.Frame(io_card, bg=self.colors['card'])
        smiles_row.pack(fill='both', expand=True, padx=10, pady=(0,8))
        smiles_row.columnconfigure(0, weight=1)
        smiles_row.rowconfigure(1, weight=1)
        label_frame = tk.Frame(smiles_row, bg=self.colors['card'])
        label_frame.grid(row=0, column=0, sticky='ew', pady=3)
        tk.Label(label_frame, text='SMILES String(s) - one per line:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left')
        
        # Help button for ChemDraw instructions
        help_btn = tk.Button(label_frame, text='?', font=('Segoe UI', 8, 'bold'),
                           bg=self.colors['primary'], fg='white', width=2, height=1,
                           relief='flat', cursor='hand2', command=self._show_chemdraw_help)
        help_btn.pack(side='right', padx=(5,0))
        
        # Load names from SVG file button (to match with pasted SMILES)
        load_names_btn = tk.Button(label_frame, text='Load Names from SVG', font=('Segoe UI', 8),
                           bg=self.colors['accent'], fg='white',
                           relief='flat', padx=8, pady=2, cursor='hand2',
                           command=self._load_names_from_svg)
        load_names_btn.pack(side='right', padx=(5,0))
        
        help_text = "1. Paste SMILES → 2. Export ChemDraw as SVG → 3. Click 'Load Names from SVG'"
        tk.Label(label_frame, text=help_text, font=('Segoe UI', 7, 'italic'),
                bg=self.colors['card'], fg=self.colors['text_light']).pack(side='right', padx=(10,5))
        self.smiles_text = scrolledtext.ScrolledText(smiles_row, height=6, wrap='none', font=('Consolas', 9),
                         bg='white', fg='black', relief='flat', borderwidth=1,
                         highlightthickness=1, highlightbackground=self.colors['border'])
        self.smiles_text.grid(row=1, column=0, sticky='nsew', pady=3)
        if DEFAULTS['SMILES_INPUT']:
            self.smiles_text.insert('1.0', DEFAULTS['SMILES_INPUT'])
        self.smiles_row = smiles_row
        # Initially show/hide rows based on input type
        if self.vars['INPUT_TYPE'].get() == 'smiles':
            self.input_row.pack_forget()
        else:  # 'com' or 'log'
            self.smiles_row.pack_forget()
            # Update input label based on initial type
            if self.vars['INPUT_TYPE'].get() == 'log':
                self.input_label.config(text='Input (.log files or folder):')
            # Update prefix/suffix labels for SMILES mode
            if hasattr(self, 'prefix_label'):
                self.prefix_label.config(text='Add Prefix:')
            if hasattr(self, 'suffix_label'):
                self.suffix_label.config(text='Add Suffix:')
        
        prefix_row = tk.Frame(io_card, bg=self.colors['card'])
        prefix_row.pack(fill='x', padx=10, pady=(0,5))
        self.prefix_label = tk.Label(prefix_row, text='Remove Prefix:', font=('Segoe UI', 8),
                bg=self.colors['card'], fg=self.colors['text'])
        self.prefix_label.grid(row=0, column=0, sticky='w', padx=(0,5))
        self.vars['REMOVE_PREFIX'] = tk.StringVar(value=DEFAULTS['REMOVE_PREFIX'])
        self.prefix_entry = tk.Entry(prefix_row, textvariable=self.vars['REMOVE_PREFIX'], width=15,
                font=('Segoe UI', 8), bg='white', fg=self.colors['text'],
                relief='flat', borderwidth=1, highlightthickness=1,
                highlightbackground=self.colors['border'])
        self.prefix_entry.grid(row=0, column=1, padx=5)
        self.suffix_label = tk.Label(prefix_row, text='Remove Suffix:', font=('Segoe UI', 8),
                bg=self.colors['card'], fg=self.colors['text'])
        self.suffix_label.grid(row=0, column=2, sticky='w', padx=(10,5))
        self.vars['REMOVE_SUFFIX'] = tk.StringVar(value=DEFAULTS['REMOVE_SUFFIX'])
        self.suffix_entry = tk.Entry(prefix_row, textvariable=self.vars['REMOVE_SUFFIX'], width=15,
                font=('Segoe UI', 8), bg='white', fg=self.colors['text'],
                relief='flat', borderwidth=1, highlightthickness=1,
                highlightbackground=self.colors['border'])
        self.suffix_entry.grid(row=0, column=3)
        
        output_row = tk.Frame(io_card, bg=self.colors['card'])
        output_row.pack(fill='x', padx=10, pady=(0,10))
        output_row.columnconfigure(0, weight=1)
        tk.Label(output_row, text='Output:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=0, column=0, sticky='w', pady=3)
        self.vars['OUT_DIR'] = tk.StringVar(value=DEFAULTS['OUT_DIR'])
        entry2 = tk.Entry(output_row, textvariable=self.vars['OUT_DIR'], font=('Segoe UI', 9),
                         bg='white', fg=self.colors['text'], relief='flat', borderwidth=1,
                         highlightthickness=1, highlightbackground=self.colors['border'])
        entry2.grid(row=1, column=0, sticky='ew', padx=(0,5))
        tk.Button(output_row, text='Browse', command=self._browse_outdir,
                 font=('Segoe UI', 8), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=10, pady=3, cursor='hand2').grid(row=1, column=1)
        
        # Method/Basis
        method_card = self._create_card(left_col, 'Method & Basis Set')
        method_card.pack(fill='x', pady=8)
        method_content = tk.Frame(method_card, bg=self.colors['card'])
        method_content.pack(fill='x', padx=10, pady=(0,10))
        method_content.columnconfigure(1, weight=1)
        
        tk.Label(method_content, text='Functional:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=0, column=0, sticky='w', pady=3)
        self.cb_func = EditableCombo(method_content, ["m062x","b3lyp","wb97xd","cam-b3lyp","pbe0","tpss","bp86","scan"])
        self.cb_func.set(DEFAULTS['FUNCTIONAL'])
        self.cb_func.grid(row=0, column=1, sticky='ew', padx=(10,0), pady=3)
        
        tk.Label(method_content, text='Basis:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=1, column=0, sticky='w', pady=3)
        self.cb_basis = EditableCombo(method_content, ["def2SVP","def2TZVP","6-31G*","6-311+G**","cc-pVDZ","cc-pVTZ","aug-cc-pVDZ"])
        self.cb_basis.set(DEFAULTS['BASIS'])
        self.cb_basis.grid(row=1, column=1, sticky='ew', padx=(10,0), pady=3)
        
        # Solvent
        solvent_card = self._create_card(left_col, 'Solvent Model')
        solvent_card.pack(fill='x', pady=8)
        solvent_content = tk.Frame(solvent_card, bg=self.colors['card'])
        solvent_content.pack(fill='x', padx=10, pady=(0,10))
        solvent_content.columnconfigure(1, weight=1)
        
        tk.Label(solvent_content, text='Model:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=0, column=0, sticky='w', pady=3)
        self.cb_smodel = EditableCombo(solvent_content, ["none","SMD","PCM","IEFPCM","CPCM"])
        self.cb_smodel.set(DEFAULTS['SOLVENT_MODEL'])
        self.cb_smodel.grid(row=0, column=1, sticky='ew', padx=(10,0), pady=3)
        
        tk.Label(solvent_content, text='Solvent:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=1, column=0, sticky='w', pady=3)
        self.cb_sname = EditableCombo(solvent_content, ["DMSO","Water","Acetonitrile","Methanol","Ethanol","DCM","THF","Toluene","Benzene","Acetone","DMF"])
        self.cb_sname.set(DEFAULTS['SOLVENT_NAME'])
        self.cb_sname.grid(row=1, column=1, sticky='ew', padx=(10,0), pady=3)
        
        # Right Column
        right_col = tk.Frame(content_frame, bg=self.colors['bg'])
        right_col.grid(row=0, column=1, sticky='nsew', padx=(8,0))
        
        # TD-DFT
        td_card = self._create_card(right_col, 'TD-DFT Settings')
        td_card.pack(fill='x', pady=8)
        td_content = tk.Frame(td_card, bg=self.colors['card'])
        td_content.pack(fill='x', padx=10, pady=(0,10))
        
        td_row1 = tk.Frame(td_content, bg=self.colors['card'])
        td_row1.pack(fill='x', pady=3)
        tk.Label(td_row1, text='NStates:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        self.vars['TD_NSTATES'] = tk.IntVar(value=DEFAULTS['TD_NSTATES'])
        spin1 = ttk.Spinbox(td_row1, from_=1, to=128, textvariable=self.vars['TD_NSTATES'], width=8)
        spin1.pack(side='left')
        
        tk.Label(td_row1, text='Root:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(20,10))
        self.vars['TD_ROOT'] = tk.IntVar(value=DEFAULTS['TD_ROOT'])
        spin2 = ttk.Spinbox(td_row1, from_=1, to=128, textvariable=self.vars['TD_ROOT'], width=8)
        spin2.pack(side='left')
        
        # State Type selection (Singlet/Triplet/Mixed)
        td_row2 = tk.Frame(td_content, bg=self.colors['card'])
        td_row2.pack(fill='x', pady=3)
        tk.Label(td_row2, text='State Type:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        if 'STATE_TYPE' not in self.vars:
            self.vars['STATE_TYPE'] = tk.StringVar(value=DEFAULTS['STATE_TYPE'])
        state_frame = tk.Frame(td_row2, bg=self.colors['card'])
        state_frame.pack(side='left')
        # Store references to radio buttons for enabling/disabling
        self.singlet_rb = tk.Radiobutton(state_frame, text='Singlet', variable=self.vars['STATE_TYPE'], value='singlet',
                      font=('Segoe UI', 9), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text'])
        self.singlet_rb.pack(side='left', padx=5)
        self.triplet_rb = tk.Radiobutton(state_frame, text='Triplet', variable=self.vars['STATE_TYPE'], value='triplet',
                      font=('Segoe UI', 9), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text'])
        self.triplet_rb.pack(side='left', padx=5)
        self.mixed_rb = tk.Radiobutton(state_frame, text='Mixed (50-50)', variable=self.vars['STATE_TYPE'], value='mixed',
                      font=('Segoe UI', 9), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text'])
        self.mixed_rb.pack(side='left', padx=5)
        
        # SOC (PySOC) preparation checkbox
        self.vars['SOC_ENABLE'] = tk.BooleanVar(value=DEFAULTS['SOC_ENABLE'])
        soc_cb = tk.Checkbutton(td_content, text='Prepare for PySOC (saves RWF, adds 6D 10F GFInput)', 
                      variable=self.vars['SOC_ENABLE'],
                      font=('Segoe UI', 8), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text'],
                      command=self._on_soc_enable_change)
        soc_cb.pack(anchor='w', pady=3)
        # Initialize SOC state on startup
        self.root.after(100, self._on_soc_enable_change)
        # Tooltip/info for SOC - explain two-step process
        soc_info = tk.Label(td_content, 
                           text='Step 1: Check this, then Generate → creates .com/.sh with %Rwf and 6D 10F GFInput\n'
                                'Step 2: After Gaussian jobs complete, use "Generate PySOC Scripts" to create SOC calculation scripts', 
                           font=('Segoe UI', 7, 'italic'), bg=self.colors['card'], fg=self.colors['text_light'],
                           justify='left')
        soc_info.pack(anchor='w', padx=(20,0), pady=(0,5))
        
        self.vars['POP_FULL'] = tk.BooleanVar(value=DEFAULTS['POP_FULL'])
        tk.Checkbutton(td_content, text='pop=(full,orbitals=2)', variable=self.vars['POP_FULL'],
                      font=('Segoe UI', 8), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text']).pack(anchor='w', pady=3)
        self.vars['DISPERSION'] = tk.BooleanVar(value=DEFAULTS['DISPERSION'])
        tk.Checkbutton(td_content, text='EmpiricalDispersion=GD3BJ', variable=self.vars['DISPERSION'],
                      font=('Segoe UI', 8), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text']).pack(anchor='w', pady=3)
        
        # Resources
        res_card = self._create_card(right_col, 'Computational Resources')
        res_card.pack(fill='x', pady=8)
        res_content = tk.Frame(res_card, bg=self.colors['card'])
        res_content.pack(fill='x', padx=10, pady=(0,10))
        res_content.columnconfigure(1, weight=1)
        
        tk.Label(res_content, text='Cores:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=0, column=0, sticky='w', pady=3)
        self.vars['NPROC'] = tk.IntVar(value=DEFAULTS['NPROC'])
        ttk.Spinbox(res_content, from_=1, to=256, textvariable=self.vars['NPROC'], width=10).grid(row=0, column=1, sticky='w', padx=(10,0), pady=3)
        
        tk.Label(res_content, text='Memory:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=1, column=0, sticky='w', pady=3)
        self.vars['MEM'] = tk.StringVar(value=DEFAULTS['MEM'])
        tk.Entry(res_content, textvariable=self.vars['MEM'], font=('Segoe UI', 9),
                bg='white', fg=self.colors['text'], relief='flat', borderwidth=1,
                highlightthickness=1, highlightbackground=self.colors['border']).grid(row=1, column=1, sticky='ew', padx=(10,0), pady=3)
        
        # Scheduler
        sched_card = self._create_card(right_col, 'Scheduler Settings')
        sched_card.pack(fill='x', pady=8)
        sched_content = tk.Frame(sched_card, bg=self.colors['card'])
        sched_content.pack(fill='x', padx=10, pady=(0,10))
        sched_content.columnconfigure(1, weight=1)
        
        tk.Label(sched_content, text='Type:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=0, column=0, sticky='w', pady=3)
        self.cb_sched = EditableCombo(sched_content, ["pbs","slurm","local"])
        self.cb_sched.set(DEFAULTS['SCHEDULER'])
        self.cb_sched.grid(row=0, column=1, sticky='ew', padx=(10,0), pady=3)
        
        tk.Label(sched_content, text='Queue:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=1, column=0, sticky='w', pady=3)
        self.cb_queue = EditableCombo(sched_content, ["normal","express","long","gpu","debug"])
        self.cb_queue.set(DEFAULTS['QUEUE'])
        self.cb_queue.grid(row=1, column=1, sticky='ew', padx=(10,0), pady=3)
        
        tk.Label(sched_content, text='Walltime:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=2, column=0, sticky='w', pady=3)
        self.vars['WALLTIME'] = tk.StringVar(value=DEFAULTS['WALLTIME'])
        tk.Entry(sched_content, textvariable=self.vars['WALLTIME'], font=('Segoe UI', 9),
                bg='white', fg=self.colors['text'], relief='flat', borderwidth=1,
                highlightthickness=1, highlightbackground=self.colors['border']).grid(row=2, column=1, sticky='ew', padx=(10,0), pady=3)
        
        tk.Label(sched_content, text='Project:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=3, column=0, sticky='w', pady=3)
        self.vars['PROJECT'] = tk.StringVar(value=DEFAULTS['PROJECT'])
        tk.Entry(sched_content, textvariable=self.vars['PROJECT'], font=('Segoe UI', 9),
                bg='white', fg=self.colors['text'], relief='flat', borderwidth=1,
                highlightthickness=1, highlightbackground=self.colors['border']).grid(row=3, column=1, sticky='ew', padx=(10,0), pady=3)
        
        tk.Label(sched_content, text='Account:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=4, column=0, sticky='w', pady=3)
        self.vars['ACCOUNT'] = tk.StringVar(value=DEFAULTS['ACCOUNT'])
        tk.Entry(sched_content, textvariable=self.vars['ACCOUNT'], font=('Segoe UI', 9),
                bg='white', fg=self.colors['text'], relief='flat', borderwidth=1,
                highlightthickness=1, highlightbackground=self.colors['border']).grid(row=4, column=1, sticky='ew', padx=(10,0), pady=3)
        
        # Charge/Multiplicity
        cm_card = self._create_card(right_col, 'Charge & Multiplicity')
        cm_card.pack(fill='x', pady=8)
        cm_content = tk.Frame(cm_card, bg=self.colors['card'])
        cm_content.pack(fill='x', padx=10, pady=(0,10))
        
        cm_row = tk.Frame(cm_content, bg=self.colors['card'])
        cm_row.pack(fill='x', pady=3)
        tk.Label(cm_row, text='Charge:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        self.vars['CHARGE'] = tk.StringVar(value='' if DEFAULTS['CHARGE'] is None else str(DEFAULTS['CHARGE']))
        tk.Entry(cm_row, textvariable=self.vars['CHARGE'], width=8,
                font=('Segoe UI', 9), bg='white', fg=self.colors['text'],
                relief='flat', borderwidth=1, highlightthickness=1,
                highlightbackground=self.colors['border']).pack(side='left')
        
        tk.Label(cm_row, text='Multiplicity:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(20,10))
        self.vars['MULT'] = tk.StringVar(value='' if DEFAULTS['MULT'] is None else str(DEFAULTS['MULT']))
        tk.Entry(cm_row, textvariable=self.vars['MULT'], width=8,
                font=('Segoe UI', 9), bg='white', fg=self.colors['text'],
                relief='flat', borderwidth=1, highlightthickness=1,
                highlightbackground=self.colors['border']).pack(side='left')
        
        # Inline Steps
        inline_card = self._create_card(right_col, 'Inline Coordinates')
        inline_card.pack(fill='x', pady=8)
        inline_content = tk.Frame(inline_card, bg=self.colors['card'])
        inline_content.pack(fill='x', padx=10, pady=(0,10))
        tk.Label(inline_content, text='Copy coords for steps:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(anchor='w', pady=3)
        inline_buttons = tk.Frame(inline_content, bg=self.colors['card'])
        inline_buttons.pack(fill='x')
        self.inline_vars = {}
        for k in (2,3,4,5,6,7):
            v = tk.BooleanVar(value=k in (DEFAULTS['INLINE_STEPS'] or []))
            tk.Checkbutton(inline_buttons, text=str(k), variable=v,
                          font=('Segoe UI', 8), bg=self.colors['card'], fg=self.colors['text'],
                          selectcolor=self.colors['card'], activebackground=self.colors['card'],
                          activeforeground=self.colors['text']).pack(side='left', padx=5)
            self.inline_vars[k] = v
        
        # Watermark at the end
        watermark_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        watermark_frame.pack(fill='x', pady=20)
        watermark_label = tk.Label(watermark_frame, text='Designed by Abedi', 
                                   font=('Segoe UI', 8, 'italic'), 
                                   bg=self.colors['bg'], fg=self.colors['text_light'])
        watermark_label.pack(pady=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _set_mode(self, mode):
        """Set mode and update UI"""
        self.vars['MODE'].set(mode)
        # Clear multi-step selections when switching away from multiple mode
        if mode != 'multiple':
            for var in self.multi_step_vars.values():
                var.set(False)
        # Clear single step when switching to full
        if mode == 'full':
            self.vars['STEP'].set(4)  # Reset to default
        self._on_mode_change()
        self._update_mode_buttons()
    
    def _set_single_step(self, step):
        """Set single step and update button appearance"""
        self.vars['STEP'].set(step)
        self._update_step_buttons()
        self._on_step_change()
    
    def _toggle_multi_step(self, step):
        """Toggle multi-step selection and update button appearance"""
        var = self.multi_step_vars[step]
        var.set(not var.get())
        self._update_multi_buttons()
        self._on_step_change()
    
    def _update_mode_buttons(self):
        """Update mode button appearances"""
        mode = self.vars['MODE'].get()
        for opt_value in ['full', 'single', 'multiple']:
            btn = getattr(self, f'_mode_btn_{opt_value}', None)
            if btn:
                if opt_value == mode:
                    btn.config(bg=self.colors['primary'], fg='white', relief='sunken')
                else:
                    btn.config(bg=self.colors['card'], fg=self.colors['text'], relief='raised')
    
    def _update_step_buttons(self):
        """Update single step button appearances"""
        selected = self.vars['STEP'].get()
        for k in range(1, 8):
            btn = getattr(self, f'_step_btn_{k}', None)
            if btn:
                if k == selected:
                    btn.config(bg=self.colors['primary'], fg='white', relief='sunken')
                else:
                    btn.config(bg=self.colors['card'], fg=self.colors['text'], relief='raised')
    
    def _update_multi_buttons(self):
        """Update multi-step button appearances"""
        for k in range(1, 8):
            var = self.multi_step_vars.get(k)
            btn = getattr(self, f'_multi_btn_{k}', None)
            if btn and var:
                if var.get():
                    btn.config(bg=self.colors['accent'], fg='white', relief='sunken')
                else:
                    btn.config(bg=self.colors['card'], fg=self.colors['text'], relief='raised')

    def _tab_advanced(self, nb):
        f = tk.Frame(nb, bg=self.colors['bg'])
        nb.add(f, text='🔧 Advanced')
        
        # Scrollable frame
        canvas = tk.Canvas(f, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(f, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg'])
        
        def update_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        def on_canvas_configure(event):
            canvas_width = event.width
            if canvas.find_all():
                canvas.itemconfig(canvas.find_all()[0], width=canvas_width)
        
        scrollable_frame.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", on_canvas_configure)
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Enable mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Route Editors
        route_card = self._create_card(scrollable_frame, 'Route Editors (for Selected Steps)')
        route_card.pack(fill='x', padx=15, pady=8)
        
        route_header = tk.Frame(route_card, bg=self.colors['card'])
        route_header.pack(fill='x', padx=10, pady=(0,8))
        
        # Detailed explanation
        explanation_frame = tk.Frame(route_card, bg=self.colors['card'])
        explanation_frame.pack(fill='x', padx=10, pady=(0,10))
        
        explanation_text = """Route Editor: This advanced feature allows you to manually customize the Gaussian route card for each computational step. 

The route card is the key directive in Gaussian input files that specifies the computational method, basis set, and job type. Here you can:
• Override auto-generated routes with custom keywords
• Add specialized options (e.g., opt=modredundant, calcfc, tight, etc.)
• Modify route parameters for specific requirements

Technical Notes:
- Routes are displayed for steps you've selected in the Main Settings tab
- Click "Update All Routes" to regenerate routes from current settings
- Manual routes completely override auto-generated routes for that step
- Ensure route syntax follows Gaussian conventions: # method/basis [options]
- Step 7 (De-excitation) automatically uses geometry from Step 6 checkpoint file"""
        
        explanation_label = tk.Label(explanation_frame, text=explanation_text, 
                                    font=('Segoe UI', 8), bg=self.colors['card'], 
                                    fg=self.colors['text'], justify='left', wraplength=750)
        explanation_label.pack(anchor='w', pady=5)
        
        tk.Button(route_header, text='Update All Routes', command=self._update_all_routes,
                 font=('Segoe UI', 9), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=12, pady=4, cursor='hand2').pack(side='right')
        
        # Container for step route editors
        self.step_route_frames = {}
        self.step_route_texts = {}
        self.step_geom_source_vars = {}
        route_container = tk.Frame(route_card, bg=self.colors['card'])
        route_container.pack(fill='x', padx=10, pady=(0,10))
        
        step_titles = {
            1: "Step 1 (GS Opt)", 2: "Step 2 (Abs)", 3: "Step 3 (Abs cLR)",
            4: "Step 4 (ES Opt)", 5: "Step 5 (Density)", 6: "Step 6 (ES cLR)",
            7: "Step 7 (De-excitation)",
        }
        
        for step in range(1, 8):
            step_frame = tk.LabelFrame(route_container, text=step_titles[step],
                                      bg=self.colors['card'], fg=self.colors['text'],
                                      font=('Segoe UI', 9, 'bold'))
            step_frame.pack(fill='x', pady=5, padx=5)
            self.step_route_frames[step] = step_frame
            
            # Geometry source selection
            geom_frame = tk.Frame(step_frame, bg=self.colors['card'])
            geom_frame.pack(fill='x', padx=8, pady=5)
            tk.Label(geom_frame, text='Geometry Source:', font=('Segoe UI', 9),
                    bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,5))
            geom_var = tk.StringVar(value='default')
            self.step_geom_source_vars[step] = geom_var
            
            geom_explanation = {
                'Default': 'Uses workflow default (inline coords or linked checkpoint)',
                'Coords S1': 'Inline coordinates from Step 1 (ground state optimized geometry)',
                'Coords S4': 'Inline coordinates from Step 4 (excited state optimized geometry)',
                'oldchk S1': 'Linked checkpoint from Step 1 (%oldchk with geom=check)',
                'oldchk S4': 'Linked checkpoint from Step 4 (%oldchk with geom=check)'
            }
            
            for label, val in [('Default', 'default'), ('Coords S1', 'coords_1'), ('Coords S4', 'coords_4'),
                              ('oldchk S1', 'oldchk_1'), ('oldchk S4', 'oldchk_4')]:
                tk.Radiobutton(geom_frame, text=label, variable=geom_var, value=val,
                              font=('Segoe UI', 8), bg=self.colors['card'], fg=self.colors['text'],
                              selectcolor=self.colors['card'], activebackground=self.colors['card'],
                              activeforeground=self.colors['text']).pack(side='left', padx=3)
            
            # Route text editor
            route_text = scrolledtext.ScrolledText(step_frame, height=2, wrap='none', font=('Consolas', 9),
                                                  bg='white', fg='black')
            route_text.pack(fill='x', padx=8, pady=(0,5))
            self.step_route_texts[step] = route_text
            step_frame.pack_forget()  # Initially hidden
        
        # Redundant coordinates (Step 4 only)
        redundant_card = self._create_card(scrollable_frame, 'Redundant Coordinates (Step 4 Only)')
        redundant_card.pack(fill='both', expand=True, padx=15, pady=8)
        
        redundant_content = tk.Frame(redundant_card, bg=self.colors['card'])
        redundant_content.pack(fill='both', expand=True, padx=10, pady=(0,10))
        
        redundant_explanation = """Redundant Internal Coordinates: This section allows you to define additional internal coordinate constraints specifically for Step 4 (Excited State Optimization).

Purpose: Redundant coordinates are used to constrain specific geometric parameters during optimization, such as:
• Dihedral angles (D atom1 atom2 atom3 atom4 value F)
• Bond angles (A atom1 atom2 atom3 value F)
• Bond distances (B atom1 atom2 value F)

Technical Details:
- These coordinates are added ONLY to Step 4 input files
- Format: One coordinate per line (e.g., "D 4 5 6 5 F" for a frozen dihedral)
- A blank line is automatically inserted after the main coordinates and before redundant coordinates
- Use "F" suffix to freeze the coordinate, or specify a target value
- Useful for constraining rotatable bonds, fixing angles, or maintaining molecular conformations

Example Format:
D 4 5 6 7 180.0 F    (Freeze dihedral angle at 180 degrees)
A 1 2 3 120.0        (Set angle to 120 degrees)
B 2 3 1.50 F         (Freeze bond length at 1.50 Å)"""
        
        tk.Label(redundant_content, text=redundant_explanation,
                font=('Segoe UI', 8), bg=self.colors['card'], fg=self.colors['text'],
                justify='left', wraplength=750).pack(anchor='w', pady=3)
        self.redundant_text = scrolledtext.ScrolledText(redundant_content, height=8, wrap='word', font=('Consolas', 10),
                                                        bg='white', fg='black')
        self.redundant_text.pack(fill='both', expand=True, pady=5)
        self.redundant_text.insert('1.0', DEFAULTS['REDUNDANT_COORDS'])
        
        # Watermark at the end
        watermark_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        watermark_frame.pack(fill='x', pady=20)
        watermark_label = tk.Label(watermark_frame, text='Designed by Abedi', 
                                   font=('Segoe UI', 8, 'italic'), 
                                   bg=self.colors['bg'], fg=self.colors['text_light'])
        watermark_label.pack(pady=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self._update_visible_routes()
    
    def _tab_tict(self, nb):
        """TICT Rotation Tab"""
        f = tk.Frame(nb, bg=self.colors['bg'])
        nb.add(f, text='🔄 TICT Rotation')
        
        # Scrollable container
        canvas = tk.Canvas(f, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(f, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg'])
        
        def update_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        def on_canvas_configure(event):
            canvas_width = event.width
            if canvas.find_all():
                canvas.itemconfig(canvas.find_all()[0], width=canvas_width)
        
        scrollable_frame.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", on_canvas_configure)
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Enable mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Initialize TICT variables
        if not hasattr(self, 'tict_vars'):
            self.tict_vars = {
                'INPUT_FILE': tk.StringVar(value=""),
                'OUTPUT_DIR': tk.StringVar(value=""),
                'AXIS': tk.StringVar(value="3,10"),
                'BRANCH_A': tk.StringVar(value="11,18,22"),
                'BRANCH_A_STEP': tk.StringVar(value="-8.81"),
                'BRANCH_B': tk.StringVar(value="12-13,19-21,23-26"),
                'BRANCH_B_STEP': tk.StringVar(value="-9.36"),
                'NUM_STEPS': tk.StringVar(value="9"),
            }
            self.tict_output_dir = None
        
        # File Selection Section
        file_card = self._create_card(scrollable_frame, 'Input/Output Files')
        file_card.pack(fill='x', padx=15, pady=8)
        
        file_content = tk.Frame(file_card, bg=self.colors['card'])
        file_content.pack(fill='x', padx=10, pady=10)
        
        # Input file
        input_row = tk.Frame(file_content, bg=self.colors['card'])
        input_row.pack(fill='x', pady=5)
        tk.Label(input_row, text='Input File (.com):', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        tk.Entry(input_row, textvariable=self.tict_vars['INPUT_FILE'], width=60,
                font=('Segoe UI', 9)).pack(side='left', fill='x', expand=True, padx=(0,5))
        tk.Button(input_row, text='Browse...', command=self._tict_browse_input,
                 font=('Segoe UI', 9), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=15, pady=4, cursor='hand2').pack(side='left')
        
        # Output directory
        output_row = tk.Frame(file_content, bg=self.colors['card'])
        output_row.pack(fill='x', pady=5)
        tk.Label(output_row, text='Output Directory:', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        tk.Entry(output_row, textvariable=self.tict_vars['OUTPUT_DIR'], width=60,
                font=('Segoe UI', 9)).pack(side='left', fill='x', expand=True, padx=(0,5))
        tk.Button(output_row, text='Browse...', command=self._tict_browse_output,
                 font=('Segoe UI', 9), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=15, pady=4, cursor='hand2').pack(side='left')
        
        # Rotation Parameters Section
        params_card = self._create_card(scrollable_frame, 'TICT Rotation Parameters (1-Based Atom Indices)')
        params_card.pack(fill='x', padx=15, pady=8)
        
        params_content = tk.Frame(params_card, bg=self.colors['card'])
        params_content.pack(fill='x', padx=10, pady=10)
        
        # Create grid layout for parameters
        params_content.columnconfigure(1, weight=1)
        
        row = 0
        # Rotation Axis
        tk.Label(params_content, text='Rotation Axis (2 atoms, e.g., "3,10"):', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(params_content, textvariable=self.tict_vars['AXIS'], font=('Segoe UI', 9)).grid(row=row, column=1, sticky='ew', padx=5, pady=5)
        row += 1
        
        # Branch A
        tk.Label(params_content, text='Branch A Indices:', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(params_content, textvariable=self.tict_vars['BRANCH_A'], font=('Segoe UI', 9)).grid(row=row, column=1, sticky='ew', padx=5, pady=5)
        row += 1
        
        tk.Label(params_content, text='Branch A Step (degrees):', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(params_content, textvariable=self.tict_vars['BRANCH_A_STEP'], font=('Segoe UI', 9), width=20).grid(row=row, column=1, sticky='w', padx=5, pady=5)
        row += 1
        
        # Branch B
        tk.Label(params_content, text='Branch B Indices:', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(params_content, textvariable=self.tict_vars['BRANCH_B'], font=('Segoe UI', 9)).grid(row=row, column=1, sticky='ew', padx=5, pady=5)
        row += 1
        
        tk.Label(params_content, text='Branch B Step (degrees):', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(params_content, textvariable=self.tict_vars['BRANCH_B_STEP'], font=('Segoe UI', 9), width=20).grid(row=row, column=1, sticky='w', padx=5, pady=5)
        row += 1
        
        # Number of steps
        tk.Label(params_content, text='Number of Steps (0 to N inclusive):', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(params_content, textvariable=self.tict_vars['NUM_STEPS'], font=('Segoe UI', 9), width=20).grid(row=row, column=1, sticky='w', padx=5, pady=5)
        row += 1
        
        # Help text
        help_text = """Help: Enter atom indices using 1-based indexing (as in GaussView).
You can use ranges (e.g., "12-13") and comma-separated lists (e.g., "11,18,22").
Example: "12-13,19-21,23-26" means atoms 12,13,19,20,21,23,24,25,26."""
        tk.Label(params_content, text=help_text, font=('Segoe UI', 8), bg=self.colors['card'],
                fg=self.colors['text_light'], justify='left', wraplength=600).grid(row=row, column=0, columnspan=2, sticky='w', padx=5, pady=(10,5))
        
        # Generate Button
        button_card = self._create_card(scrollable_frame, None)
        button_card.pack(fill='x', padx=15, pady=8)
        button_frame = tk.Frame(button_card, bg=self.colors['card'])
        button_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(button_frame, text='Generate Rotated Geometries', command=self._tict_generate,
                 font=('Segoe UI', 11, 'bold'), bg=self.colors['accent'], fg='white',
                 relief='flat', padx=25, pady=10, cursor='hand2').pack(side='left', padx=5)
        
        if not TICT_AVAILABLE:
            warning_label = tk.Label(button_frame, text='⚠️ TICT module not available', 
                                    font=('Segoe UI', 9), bg=self.colors['card'], fg='red')
            warning_label.pack(side='left', padx=10)
        
        # Status and Log
        log_card = self._create_card(scrollable_frame, 'Status & Log')
        log_card.pack(fill='both', expand=True, padx=15, pady=8)
        log_content = tk.Frame(log_card, bg=self.colors['card'])
        log_content.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.tict_log = scrolledtext.ScrolledText(log_content, wrap='word', font=('Consolas', 10),
                                                  bg='white', fg='black', height=15)
        self.tict_log.pack(fill='both', expand=True)
        self.tict_log.insert('1.0', 'Ready. Select input file and set parameters, then click "Generate Rotated Geometries".\n')
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Watermark
        watermark_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        watermark_frame.pack(fill='x', pady=20)
        watermark_label = tk.Label(watermark_frame, text='Designed by Abedi', 
                                   font=('Segoe UI', 8, 'italic'), 
                                   bg=self.colors['bg'], fg=self.colors['text_light'])
        watermark_label.pack()

    def _tab_generate(self, nb):
        f = tk.Frame(nb, bg=self.colors['bg'])
        nb.add(f, text='🚀 Generate')
        
        # Button bar
        button_card = self._create_card(f, None)
        button_card.pack(fill='x', padx=15, pady=10)
        button_frame = tk.Frame(button_card, bg=self.colors['card'])
        button_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(button_frame, text='Preview', command=self._preview,
                 font=('Segoe UI', 10, 'bold'), bg=self.colors['secondary'], fg='white',
                 relief='flat', padx=20, pady=8, cursor='hand2').pack(side='left', padx=5)
        tk.Button(button_frame, text='Generate', command=self._generate,
                 font=('Segoe UI', 10, 'bold'), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=20, pady=8, cursor='hand2').pack(side='left', padx=5)
        tk.Button(button_frame, text='Save Prefs', command=self._save_prefs,
                 font=('Segoe UI', 9), bg=self.colors['accent'], fg='white',
                 relief='flat', padx=15, pady=8, cursor='hand2').pack(side='left', padx=5)
        tk.Button(button_frame, text='Load Prefs', command=self._load_prefs,
                 font=('Segoe UI', 9), bg=self.colors['accent'], fg='white',
                 relief='flat', padx=15, pady=8, cursor='hand2').pack(side='left', padx=5)
        tk.Button(button_frame, text='Reset to Defaults', command=self._reset_to_defaults,
                 font=('Segoe UI', 9), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=15, pady=8, cursor='hand2').pack(side='left', padx=5)
        tk.Button(button_frame, text='Generate PySOC Scripts', command=self._run_pysoc,
                 font=('Segoe UI', 9), bg=self.colors['accent'], fg='white',
                 relief='flat', padx=15, pady=8, cursor='hand2').pack(side='left', padx=5)
        
        # Status
        status_card = self._create_card(f, None)
        status_card.pack(fill='x', padx=15, pady=(0,10))
        status_frame = tk.Frame(status_card, bg=self.colors['card'])
        status_frame.pack(fill='x', padx=10, pady=8)
        self.status = tk.Label(status_frame, text='Ready', font=('Segoe UI', 10, 'bold'),
                              bg=self.colors['card'], fg=self.colors['primary'])
        self.status.pack(anchor='w')
        
        # Preview
        preview_card = self._create_card(f, 'Preview & Output')
        preview_card.pack(fill='both', expand=True, padx=15, pady=(0,10))
        preview_content = tk.Frame(preview_card, bg=self.colors['card'])
        preview_content.pack(fill='both', expand=True, padx=10, pady=10)
        self.preview = scrolledtext.ScrolledText(preview_content, wrap='word', font=('Consolas', 10),
                                                bg='white', fg='black')
        self.preview.pack(fill='both', expand=True)
        
        # Watermark at the end
        watermark_frame = tk.Frame(f, bg=self.colors['bg'])
        watermark_frame.pack(fill='x', pady=10)
        watermark_label = tk.Label(watermark_frame, text='Designed by Abedi', 
                                   font=('Segoe UI', 8, 'italic'), 
                                   bg=self.colors['bg'], fg=self.colors['text_light'])
        watermark_label.pack()

    # ---- helpers ----
    def _browse_inputs(self):
        input_type = self.vars.get('INPUT_TYPE', tk.StringVar(value='com')).get()
        if input_type == 'log':
            path = filedialog.askdirectory(title='Pick folder with .log files')
        else:  # 'com'
            path = filedialog.askdirectory(title='Pick folder with .com files')
        if path:
            self.vars['INPUTS'].set(str(Path(path)))
    def _browse_outdir(self):
        path = filedialog.askdirectory(title='Select output directory')
        if path:
            self.vars['OUT_DIR'].set(str(Path(path)))
    
    def _tict_browse_input(self):
        """Browse for input .com file for TICT rotation"""
        path = filedialog.askopenfilename(title='Select input .com file', filetypes=[('Gaussian COM files', '*.com'), ('All files', '*.*')])
        if path:
            self.tict_vars['INPUT_FILE'].set(path)
            # Auto-set output directory to same location if not set
            if not self.tict_vars['OUTPUT_DIR'].get():
                self.tict_vars['OUTPUT_DIR'].set(str(Path(path).parent))
    
    def _tict_browse_output(self):
        """Browse for output directory for TICT rotation"""
        path = filedialog.askdirectory(title='Select output directory for rotated geometries')
        if path:
            self.tict_vars['OUTPUT_DIR'].set(path)
    
    def _tict_generate(self):
        """Generate TICT rotated geometries"""
        if not TICT_AVAILABLE:
            messagebox.showerror('TICT Module Error', 'TICT rotation module is not available. Please ensure tict_rotation.py is in the same directory.')
            return
        
        # Clear log
        self.tict_log.delete('1.0', tk.END)
        self.tict_log.insert('1.0', 'Starting TICT rotation generation...\n\n')
        
        # Get inputs
        input_file = self.tict_vars['INPUT_FILE'].get().strip()
        output_dir = self.tict_vars['OUTPUT_DIR'].get().strip()
        
        if not input_file:
            self.tict_log.insert(tk.END, 'ERROR: Please select an input file.\n')
            return
        
        if not os.path.exists(input_file):
            self.tict_log.insert(tk.END, f'ERROR: Input file does not exist: {input_file}\n')
            return
        
        if not output_dir:
            self.tict_log.insert(tk.END, 'ERROR: Please select an output directory.\n')
            return
        
        try:
            axis_str = self.tict_vars['AXIS'].get().strip()
            branch_a_str = self.tict_vars['BRANCH_A'].get().strip()
            branch_a_step = float(self.tict_vars['BRANCH_A_STEP'].get().strip())
            branch_b_str = self.tict_vars['BRANCH_B'].get().strip()
            branch_b_step = float(self.tict_vars['BRANCH_B_STEP'].get().strip())
            num_steps = int(self.tict_vars['NUM_STEPS'].get().strip())
        except ValueError as e:
            self.tict_log.insert(tk.END, f'ERROR: Invalid parameter value: {e}\n')
            return
        
        # Create output directory name
        base_name = Path(input_file).stem
        tict_output_dir = os.path.join(output_dir, f"{base_name}_tict_rotations")
        
        self.tict_log.insert(tk.END, f'Input file: {input_file}\n')
        self.tict_log.insert(tk.END, f'Output directory: {tict_output_dir}\n')
        self.tict_log.insert(tk.END, f'Rotation axis: {axis_str}\n')
        self.tict_log.insert(tk.END, f'Branch A: {branch_a_str}, step: {branch_a_step}°\n')
        self.tict_log.insert(tk.END, f'Branch B: {branch_b_str}, step: {branch_b_step}°\n')
        self.tict_log.insert(tk.END, f'Number of steps: {num_steps}\n\n')
        self.tict_log.update()
        
        # Generate rotations
        success, message, files_created = generate_tict_rotations(
            input_file=input_file,
            output_dir=tict_output_dir,
            axis_str=axis_str,
            branch_a_str=branch_a_str,
            branch_a_step_deg=branch_a_step,
            branch_b_str=branch_b_str,
            branch_b_step_deg=branch_b_step,
            num_steps=num_steps,
            file_format="gaussian"
        )
        
        if success:
            self.tict_log.insert(tk.END, f'\n{message}\n')
            self.tict_log.insert(tk.END, f'\nFiles created:\n')
            for f in files_created[:10]:  # Show first 10 files
                self.tict_log.insert(tk.END, f'  {os.path.basename(f)}\n')
            if len(files_created) > 10:
                self.tict_log.insert(tk.END, f'  ... and {len(files_created) - 10} more files\n')
            self.tict_log.insert(tk.END, f'\n✅ Success! Rotated geometries saved to:\n  {tict_output_dir}\n')
            self.tict_log.insert(tk.END, f'\n💡 Tip: You can now use these rotated geometries in the Main Settings tab\n')
            self.tict_log.insert(tk.END, f'   by setting the Input field to: {tict_output_dir}\n')
            self.tict_output_dir = tict_output_dir
            messagebox.showinfo('TICT Rotation Complete', f'Successfully generated {len(files_created)} rotated geometry files!\n\nSaved to:\n{tict_output_dir}')
        else:
            self.tict_log.insert(tk.END, f'\n❌ ERROR: {message}\n')
            messagebox.showerror('TICT Rotation Error', message)
        
        self.tict_log.see(tk.END)
    
    def _on_input_type_change(self):
        """Show/hide input fields based on input type selection and update labels"""
        input_type = self.vars['INPUT_TYPE'].get()
        if input_type == 'smiles':
            self.input_row.pack_forget()
            self.smiles_row.pack(fill='both', expand=True, padx=10, pady=(0,8))
            # Update prefix/suffix labels for SMILES (add mode)
            if hasattr(self, 'prefix_label'):
                self.prefix_label.config(text='Add Prefix:')
            if hasattr(self, 'suffix_label'):
                self.suffix_label.config(text='Add Suffix:')
        else:  # 'com' or 'log'
            self.smiles_row.pack_forget()
            self.input_row.pack(fill='x', padx=10, pady=(0,8))
            # Update input label based on type
            if hasattr(self, 'input_label'):
                if input_type == 'log':
                    self.input_label.config(text='Input (.log files or folder):')
                else:  # 'com'
                    self.input_label.config(text='Input (.com files or folder):')
            # Update prefix/suffix labels for .com/.log files (remove mode)
            if hasattr(self, 'prefix_label'):
                self.prefix_label.config(text='Remove Prefix:')
            if hasattr(self, 'suffix_label'):
                self.suffix_label.config(text='Remove Suffix:')
    
    def _load_names_from_svg(self):
        """Load names from SVG file (exported from ChemDraw) and match with pasted SMILES"""
        if not XML_AVAILABLE:
            messagebox.showerror('XML Parser Required', 
                               'XML parser is required to extract names from SVG files.')
            return
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title='Select SVG File (exported from ChemDraw)',
            filetypes=[
                ('SVG Files', '*.svg'),
                ('All Files', '*.*')
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Extract just the names from SVG
            names = extract_names_from_svg(Path(file_path))
            
            if not names:
                messagebox.showwarning('No Names Found', 
                                     'No text labels found in the SVG file.\n\n'
                                     'Make sure:\n'
                                     '1. Your structures have labels (like "FLIMBD 1", etc.) in ChemDraw\n'
                                     '2. The SVG was exported from ChemDraw with text visible')
                return
            
            # Get current SMILES from text field
            current_text = self.smiles_text.get('1.0', 'end-1c').strip()
            if not current_text:
                messagebox.showwarning('No SMILES', 
                                     'Please paste your SMILES strings first, then load names to match them.')
                return
            
            # Parse current SMILES (handle period-separated format)
            lines = current_text.split('\n')
            smiles_list = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Handle period-separated format
                if '.' in line and len(line) > 50:
                    parts = line.split('.')
                    for part in parts:
                        part = part.strip()
                        if part and any(c in part for c in '()[]=+-CNOS0123456789#%'):
                            # Check if it already has a name
                            if ':' in part:
                                _, smiles = part.split(':', 1)
                                smiles_list.append(smiles.strip())
                            else:
                                smiles_list.append(part)
                else:
                    # Single SMILES per line
                    if ':' in line:
                        _, smiles = line.split(':', 1)
                        smiles_list.append(smiles.strip())
                    else:
                        smiles_list.append(line)
            
            if len(smiles_list) != len(names):
                result = messagebox.askyesno(
                    'Count Mismatch',
                    f'Found {len(names)} names in file but {len(smiles_list)} SMILES in text field.\n\n'
                    f'Names: {names[:5]}{"..." if len(names) > 5 else ""}\n\n'
                    f'Continue anyway? (Names will be matched by order)'
                )
                if not result:
                    return
            
            # Match names with SMILES
            formatted_lines = []
            for idx, smiles in enumerate(smiles_list):
                name = names[idx] if idx < len(names) else None
                if name:
                    formatted_lines.append(f"{name}:{smiles}")
                else:
                    formatted_lines.append(smiles)
            
            # Update text field
            self.smiles_text.delete('1.0', 'end')
            self.smiles_text.insert('1.0', '\n'.join(formatted_lines))
            
            messagebox.showinfo('Success', 
                              f'Matched {min(len(names), len(smiles_list))} name(s) with SMILES!\n\n'
                              f'Names loaded from: {Path(file_path).name}\n\n'
                              f'Found names: {", ".join(names[:10])}{"..." if len(names) > 10 else ""}')
        
        except Exception as e:
            messagebox.showerror('Error Loading Names', 
                               f'Could not extract names from SVG file:\n\n{str(e)}\n\n'
                               'Make sure:\n'
                               '1. The file is a valid SVG exported from ChemDraw\n'
                               '2. Your structures have visible text labels in ChemDraw\n'
                               '3. Export as SVG: File → Save As → SVG format')
    
    def _show_chemdraw_help(self):
        """Show help dialog for ChemDraw integration"""
        help_text = """How to Use SMILES with Names from ChemDraw:

STEP 1 - Copy SMILES (WORKS GREAT!):
1. In ChemDraw, select ALL structures (Ctrl+A or drag select)
2. Go to: Edit → Copy As → SMILES (or Alt+Ctrl+C)
3. Paste directly into the text field above
   ✓ Multiple SMILES on one line (separated by periods) are AUTO-DETECTED
   ✓ Each structure will be processed separately

STEP 2 - Get Names from SVG (NEW & EASY!):
1. In ChemDraw: File → Save As → SVG format
2. Click "Load Names from SVG" button above
3. Select your SVG file
4. Names will be automatically matched with your SMILES!

ALTERNATIVE - Manual Format:
You can also type manually:
  • Just SMILES: CCO
  • With name: FLIMBD_1:CCO
  • Tab-separated: FLIMBD_1<TAB>CCO
  • Comments: # This is a comment

TIPS:
• SVG extraction uses spatial coordinates for accurate matching
• Names are sorted by position (top→bottom, left→right)
• Empty lines are ignored
• Lines starting with # are comments
• Names are sanitized for filenames automatically"""
        
        help_window = tk.Toplevel(self.root)
        help_window.title("ChemDraw Integration Help")
        help_window.geometry("600x450")
        help_window.configure(bg=self.colors['bg'])
        
        text_widget = scrolledtext.ScrolledText(help_window, wrap='word', 
                                               font=('Segoe UI', 10),
                                               bg='white', fg='black',
                                               padx=10, pady=10)
        text_widget.pack(fill='both', expand=True, padx=10, pady=10)
        text_widget.insert('1.0', help_text)
        text_widget.config(state='disabled')
        
        tk.Button(help_window, text='Close', command=help_window.destroy,
                 font=('Segoe UI', 9), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=20, pady=5, cursor='hand2').pack(pady=10)
    
    def _on_step_change(self):
        """Update visible route editors when step selection changes"""
        if hasattr(self, 'step_route_frames'):
            self._update_visible_routes()
    
    def _update_visible_routes(self):
        """Show route editors for selected steps and update their routes"""
        mode = self.vars['MODE'].get()
        if mode == 'full':
            visible_steps = list(range(1, 8))
        elif mode == 'multiple':
            visible_steps = [k for k, v in self.multi_step_vars.items() if v.get()]
        else:  # single
            visible_steps = [int(self.vars['STEP'].get())]
        
        # Show/hide step frames
        for step in range(1, 8):
            if step in self.step_route_frames:
                if step in visible_steps:
                    self.step_route_frames[step].pack(fill='x', pady=3, padx=3)
                else:
                    self.step_route_frames[step].pack_forget()
        
        # Update routes for visible steps
        self._update_all_routes()
    
    def _on_mode_change(self):
        """Update UI when mode changes"""
        mode = self.vars['MODE'].get()
        if mode == 'single':
            self.single_step_frame.pack(side='left')
            self.multi_step_frame.pack_forget()
            self._update_step_buttons()
        elif mode == 'multiple':
            self.single_step_frame.pack_forget()
            self.multi_step_frame.pack(side='left')
            self._update_multi_buttons()
        else:  # full
            self.single_step_frame.pack_forget()
            self.multi_step_frame.pack_forget()
        
        if hasattr(self, 'step_route_frames'):
            self._update_visible_routes()
    
    def _on_soc_enable_change(self):
        """Handle SOC enable/disable - disable Singlet/Triplet and Mode/Steps when enabled, update routes"""
        soc_enabled = self.vars['SOC_ENABLE'].get()
        
        # Disable/enable state type radio buttons
        if hasattr(self, 'singlet_rb') and hasattr(self, 'triplet_rb') and hasattr(self, 'mixed_rb'):
            if soc_enabled:
                # Disable Singlet and Triplet, force Mixed (50-50)
                self.singlet_rb.config(state='disabled')
                self.triplet_rb.config(state='disabled')
                self.mixed_rb.config(state='normal')
                # Force state type to mixed
                self.vars['STATE_TYPE'].set('mixed')
            else:
                # Enable all radio buttons
                self.singlet_rb.config(state='normal')
                self.triplet_rb.config(state='normal')
                self.mixed_rb.config(state='normal')
        
        # Disable/enable Mode and Steps when PySOC is enabled (PySOC is independent of steps)
        if hasattr(self, 'mode_buttons_frame') and hasattr(self, 'step_row'):
            if soc_enabled:
                # Disable all mode buttons
                for value in ['full', 'single', 'multiple']:
                    btn = getattr(self, f'_mode_btn_{value}', None)
                    if btn:
                        btn.config(state='disabled')
                # Disable all step buttons
                for k in range(1, 8):
                    btn = getattr(self, f'_step_btn_{k}', None)
                    if btn:
                        btn.config(state='disabled')
                    btn = getattr(self, f'_multi_btn_{k}', None)
                    if btn:
                        btn.config(state='disabled')
            else:
                # Enable all mode buttons
                for value in ['full', 'single', 'multiple']:
                    btn = getattr(self, f'_mode_btn_{value}', None)
                    if btn:
                        btn.config(state='normal')
                # Enable all step buttons
                for k in range(1, 8):
                    btn = getattr(self, f'_step_btn_{k}', None)
                    if btn:
                        btn.config(state='normal')
                    btn = getattr(self, f'_multi_btn_{k}', None)
                    if btn:
                        btn.config(state='normal')
        
        # Update all routes to reflect SOC changes
        if hasattr(self, 'step_route_frames'):
            self._update_all_routes()
    
    def _update_all_routes(self):
        """Update all route texts from current settings"""
        cfg_temp = dict(
            FUNCTIONAL=self.cb_func.get() or DEFAULTS['FUNCTIONAL'],
            BASIS=self.cb_basis.get() or DEFAULTS['BASIS'],
            SOLVENT_MODEL=self.cb_smodel.get() or DEFAULTS['SOLVENT_MODEL'],
            SOLVENT_NAME=self.cb_sname.get() or DEFAULTS['SOLVENT_NAME'],
            TD_NSTATES=int(self.vars['TD_NSTATES'].get()),
            TD_ROOT=int(self.vars['TD_ROOT'].get()),
            STATE_TYPE=self.vars['STATE_TYPE'].get() if 'STATE_TYPE' in self.vars else DEFAULTS['STATE_TYPE'],
            POP_FULL=bool(self.vars['POP_FULL'].get()),
            DISPERSION=bool(self.vars['DISPERSION'].get()),
            SOC_ENABLE=bool(self.vars['SOC_ENABLE'].get()) if 'SOC_ENABLE' in self.vars else DEFAULTS['SOC_ENABLE'],
            MANUAL_ROUTES={},
        )
        
        route_funcs = {1: route_step1, 2: route_step2, 3: route_step3, 4: route_step4,
                       5: route_step5, 6: route_step6, 7: route_step7}
        
        for step, route_text in self.step_route_texts.items():
            if step in route_funcs:
                current_route = route_funcs[step](cfg_temp)
                route_text.delete('1.0', 'end')
                route_text.insert('1.0', current_route)

    def _collect(self):
        # charge/mult parse (allow blank = None)
        def _to_int_or_none(s):
            if s is None:
                return None
            s = str(s).strip()
            if s == '' or s.lower() == 'none':
                return None
            try:
                return int(s)
            except (ValueError, TypeError):
                # Silently ignore invalid values, don't show error dialog
                return None
        inline = [k for k,v in self.inline_vars.items() if v.get()]
        multi_steps = [k for k,v in self.multi_step_vars.items() if v.get()]
        
        # Collect manual routes
        manual_routes = {}
        if hasattr(self, 'step_route_texts'):
            for step, route_text in self.step_route_texts.items():
                route = route_text.get('1.0', 'end-1c').strip()
                if route:
                    manual_routes[step] = route
        
        # Collect geometry sources
        geom_sources = {}
        if hasattr(self, 'step_geom_source_vars'):
            for step, var in self.step_geom_source_vars.items():
                source = var.get()
                if source != 'default':
                    geom_sources[step] = source
        
        # Get redundant coordinates
        redundant_coords = ""
        if hasattr(self, 'redundant_text'):
            redundant_coords = self.redundant_text.get('1.0', 'end-1c').strip()
        
        mode = self.vars['MODE'].get()
        
        cfg = dict(
            MODE=mode,  # Store actual mode: 'full', 'single', or 'multiple'
            STEP=int(self.vars['STEP'].get()),
            INPUT_TYPE=self.vars['INPUT_TYPE'].get(),
            INPUTS=self.vars['INPUTS'].get().strip(),
            SMILES_INPUT=self.smiles_text.get('1.0', 'end-1c').strip() if hasattr(self, 'smiles_text') else self.vars.get('SMILES_INPUT', tk.StringVar()).get().strip(),
            OUT_DIR=self.vars['OUT_DIR'].get().strip(),
            FUNCTIONAL=self.cb_func.get() or DEFAULTS['FUNCTIONAL'],
            BASIS=self.cb_basis.get() or DEFAULTS['BASIS'],
            SOLVENT_MODEL=self.cb_smodel.get() or DEFAULTS['SOLVENT_MODEL'],
            SOLVENT_NAME=self.cb_sname.get() or DEFAULTS['SOLVENT_NAME'],
            TD_NSTATES=int(self.vars['TD_NSTATES'].get()),
            TD_ROOT=int(self.vars['TD_ROOT'].get()),
            STATE_TYPE=self.vars['STATE_TYPE'].get() if 'STATE_TYPE' in self.vars else DEFAULTS['STATE_TYPE'],
            POP_FULL=bool(self.vars['POP_FULL'].get()),
            DISPERSION=bool(self.vars['DISPERSION'].get()),
            NPROC=int(self.vars['NPROC'].get()),
            MEM=self.vars['MEM'].get().strip(),
            SCHEDULER=self.cb_sched.get() or DEFAULTS['SCHEDULER'],
            QUEUE=self.cb_queue.get() or DEFAULTS['QUEUE'],
            WALLTIME=self.vars['WALLTIME'].get().strip(),
            PROJECT=self.vars['PROJECT'].get().strip(),
            ACCOUNT=self.vars['ACCOUNT'].get().strip(),
            INLINE_STEPS=inline,
            INLINE_SOURCE_5TO7=4,
            CHARGE=_to_int_or_none(self.vars['CHARGE'].get()),
            MULT=_to_int_or_none(self.vars['MULT'].get()),
            REMOVE_PREFIX=self.vars['REMOVE_PREFIX'].get().strip(),
            REMOVE_SUFFIX=self.vars['REMOVE_SUFFIX'].get().strip(),
            MULTI_STEPS=multi_steps,
            MANUAL_ROUTES=manual_routes,
            REDUNDANT_COORDS=redundant_coords,
            GEOM_SOURCE=geom_sources,
            SOC_ENABLE=bool(self.vars['SOC_ENABLE'].get()) if 'SOC_ENABLE' in self.vars else DEFAULTS['SOC_ENABLE'],
        )
        return cfg

    def _preview(self):
        cfg = self._collect()
        
        # Handle SMILES input
        if cfg['INPUT_TYPE'] == 'smiles':
            smiles_text = cfg['SMILES_INPUT'].strip()
            if not smiles_text:
                self.preview.delete('1.0','end')
                self.preview.insert('1.0','Please enter at least one SMILES string (one per line).')
                self.status.config(text='No SMILES input')
                return
            
            if not RDKIT_AVAILABLE:
                error_msg = "RDKit is required for SMILES input."
                if RDKIT_ERROR:
                    error_msg += f"\n\nError: {RDKIT_ERROR}"
                error_msg += "\n\nInstall with: conda install -c conda-forge rdkit"
                error_msg += "\nOR: pip install rdkit"
                self.preview.delete('1.0','end')
                self.preview.insert('1.0', error_msg)
                self.status.config(text='RDKit not available')
                return
            
            # Parse multiple SMILES
            # ChemDraw exports multiple SMILES on one line separated by periods (.)
            lines = smiles_text.split('\n')
            smiles_data = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Check if this line contains multiple SMILES separated by periods (ChemDraw format)
                if '.' in line and len(line) > 50:  # Likely multiple SMILES on one line
                    # Split by period - ChemDraw separates multiple SMILES with periods
                    parts = line.split('.')
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        # Validate it looks like SMILES (contains typical SMILES characters)
                        if any(c in part for c in '()[]=+-CNOS0123456789#%'):
                            name, smiles = parse_smiles_line(part)
                            if smiles:
                                smiles_data.append((name, smiles))
                else:
                    # Single SMILES per line (normal format)
                    name, smiles = parse_smiles_line(line)
                    if smiles:
                        smiles_data.append((name, smiles))
            
            if not smiles_data:
                self.preview.delete('1.0','end')
                self.preview.insert('1.0','Please enter at least one SMILES string.\nFormat: "name:SMILES" or just "SMILES"\nChemDraw format (period-separated) is supported!')
                self.status.config(text='No SMILES input')
                return
            
            files_data = []
            errors = []
            for idx, (name, smiles) in enumerate(smiles_data[:10], 1):  # Preview max 10 to keep UI responsive
                try:
                    cm, coords, base = smiles_to_coords(smiles, cfg['CHARGE'], cfg['MULT'], name)
                    cm = cm_override(cm, cfg['CHARGE'], cfg['MULT'])
                    
                    # If no custom name was provided, number the molecule
                    if not name:
                        base = f"{base}_{idx}"
                    
                    # Apply prefix/suffix ADDITION for SMILES (not removal)
                    if cfg['REMOVE_PREFIX']:
                        base = cfg['REMOVE_PREFIX'] + base
                    if cfg['REMOVE_SUFFIX']:
                        base = base + cfg['REMOVE_SUFFIX']
                    
                    files_data.append((base, cm, coords))
                except Exception as e:
                    display_name = name if name else smiles[:30]
                    errors.append(f"Entry {idx} ({display_name}...): {str(e)}")
            
            if errors:
                error_msg = "Errors processing SMILES:\n" + "\n".join(errors)
                if files_data:
                    error_msg += f"\n\nSuccessfully processed {len(files_data)} SMILES string(s)."
                self.preview.delete('1.0','end')
                self.preview.insert('1.0', error_msg)
                self.status.config(text=f'SMILES errors - {len(files_data)} OK')
                if not files_data:
                    return
        elif cfg['INPUT_TYPE'] == 'log':
            # Handle .log file input
            files = find_geoms(cfg['INPUTS'], input_type='log')
            if not files:
                self.preview.delete('1.0','end')
                self.preview.insert('1.0','No .log files found for the given folder/glob.')
                self.status.config(text='No inputs')
                return
            files_data = []
            errors = []
            for idx, p in enumerate(files[:3], 1):  # preview max a few inputs to keep UI responsive
                try:
                    cm, coords, base = parse_gaussian_log(p)
                    cm = cm_override(cm, cfg['CHARGE'], cfg['MULT'])
                    # Apply prefix/suffix removal - work with the base name from parse_gaussian_log
                    if cfg['REMOVE_PREFIX'] or cfg['REMOVE_SUFFIX']:
                        # Use the full filename for removal, then extract base name
                        cleaned = remove_prefix_suffix(p.name, cfg['REMOVE_PREFIX'], cfg['REMOVE_SUFFIX'])
                        # Remove extension to get base name
                        if '.' in cleaned:
                            base = cleaned.rsplit('.', 1)[0]
                        else:
                            base = cleaned
                    files_data.append((base, cm, coords))
                except Exception as e:
                    errors.append(f"{p.name}: {str(e)}")
            
            if errors:
                error_msg = "Errors processing log files:\n" + "\n".join(errors)
                if files_data:
                    error_msg += f"\n\nSuccessfully processed {len(files_data)} log file(s)."
                self.preview.delete('1.0','end')
                self.preview.insert('1.0', error_msg)
                self.status.config(text=f'Log errors - {len(files_data)} OK')
                if not files_data:
                    return
        else:  # 'com'
            # Handle .com file input
            files = find_geoms(cfg['INPUTS'], input_type='com')
            if not files:
                self.preview.delete('1.0','end')
                self.preview.insert('1.0','No .com files found for the given folder/glob.')
                self.status.config(text='No inputs')
                return
            files_data = []
            for p in files[:3]:  # preview max a few inputs to keep UI responsive
                cm, coords = extract_cm_coords(read_lines(p))
                cm = cm_override(cm, cfg['CHARGE'], cfg['MULT'])
                base = p.stem
                # Apply prefix/suffix removal
                if cfg['REMOVE_PREFIX'] or cfg['REMOVE_SUFFIX']:
                    base = remove_prefix_suffix(p.name, cfg['REMOVE_PREFIX'], cfg['REMOVE_SUFFIX']).replace('.com', '')
                files_data.append((base, cm, coords))
        
        # PySOC mode: filename-based, no steps
        if cfg.get('SOC_ENABLE', False):
            chunks = []
            for base, cm, coords in files_data:
                # Simple naming: base_functional_basis_solvent_SOC (no step numbers)
                solv_tag_val = solv_tag(cfg['SOLVENT_MODEL'], cfg['SOLVENT_NAME'])
                job = f"{base}_{cfg['FUNCTIONAL']}_{cfg['BASIS']}_{solv_tag_val}_SOC"
                title = f"PySOC {cfg['FUNCTIONAL']}/{cfg['BASIS']}"
                # Use Step 2 route (TD-DFT) for PySOC
                route = route_step2(cfg)
                save_rwf = True  # Always save RWF for PySOC
                lines = make_com_inline(job, cfg['NPROC'], cfg['MEM'], route, title, cm, coords, save_rwf=save_rwf)
                chunks.append(f"### {job}.com\n"+"\n".join(lines))
        else:
            # Normal step-based mode
            if cfg['MODE'] == 'full':
                steps = list(range(1,8))
            elif cfg['MODE'] == 'multiple':
                steps = cfg['MULTI_STEPS'] if cfg['MULTI_STEPS'] else []
            else:
                steps = [cfg['STEP']]
            
            chunks = []
            for base, cm, coords in files_data:
                for k in steps:
                    job = jobname(k, base, cfg)
                    # Generate title based on step and state type
                    state_type = cfg.get('STATE_TYPE', 'singlet').lower()
                    state_label = "Triplet" if state_type == 'triplet' else ("Mixed" if state_type == 'mixed' else "Singlet")
                    if k == 1:
                        title = "Step1 GS Opt"
                    elif k == 5:
                        title = "Step5 Density"
                    else:
                        step_names = {2: "Abs", 3: "Abs cLR", 4: "ES Opt", 6: "ES cLR", 7: "De-excitation"}
                        title = f"Step{k} {state_label} {step_names[k]}"
                    geom_sources = cfg.get('GEOM_SOURCE', {})
                    source = geom_sources.get(k)
                    
                    save_rwf = cfg.get('SOC_ENABLE', False)
                    # Step 7 always uses oldchk from Step 6
                    if k == 7:
                        route = step_route(k, cfg) + " geom=check guess=read"
                        lines = make_com_linked(job, cfg['NPROC'], cfg['MEM'], f"{jobname(6, base, cfg)}.chk", route, f"{title} {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm, save_rwf=save_rwf)
                    elif source and source.startswith('oldchk_'):
                        # Linked geometry
                        oldchk_name = f"{jobname(1 if source == 'oldchk_1' else 4, base, cfg)}.chk"
                        route = step_route(k, cfg) + " geom=check guess=read"
                        lines = make_com_linked(job, cfg['NPROC'], cfg['MEM'], oldchk_name, route, f"{title} {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm, save_rwf=save_rwf)
                    elif cfg['MODE'] == 'full' and k!=1 and (k not in (cfg['INLINE_STEPS'] or [])) and not source:
                        # Default linked
                        route = step_route(k, cfg) + " geom=check guess=read"
                        lines = make_com_linked(job, cfg['NPROC'], cfg['MEM'], f"{jobname(1 if k<5 else 4, base, cfg)}.chk", route, f"{title} {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm, save_rwf=save_rwf)
                    else:
                        # Inline coordinates
                        coords_use = add_redundant_coords(coords.copy(), cfg.get('REDUNDANT_COORDS', ""), k)
                        route = step_route(k, cfg)
                        lines = make_com_inline(job, cfg['NPROC'], cfg['MEM'], route, f"{title} {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm, coords_use, save_rwf=save_rwf)
                    chunks.append(f"### {job}.com\n"+"\n".join(lines))
        self.preview.delete('1.0','end'); self.preview.insert('1.0','\n\n'.join(chunks))
        self.status.config(text=f"Preview OK — {len(files_data)} input(s);")

    def _generate(self):
        cfg = self._collect()
        
        # Handle SMILES input
        if cfg['INPUT_TYPE'] == 'smiles':
            smiles_text = cfg['SMILES_INPUT'].strip()
            if not smiles_text:
                messagebox.showerror('No input', 'Please enter at least one SMILES string (one per line).')
                return
            
            if not RDKIT_AVAILABLE:
                error_msg = "RDKit is required for SMILES input."
                if RDKIT_ERROR:
                    error_msg += f"\n\nError: {RDKIT_ERROR}"
                error_msg += "\n\nInstall with: conda install -c conda-forge rdkit"
                error_msg += "\nOR: pip install rdkit"
                messagebox.showerror('RDKit required', error_msg)
                return
            
            # Parse multiple SMILES
            # ChemDraw exports multiple SMILES on one line separated by periods (.)
            lines = smiles_text.split('\n')
            smiles_data = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Check if this line contains multiple SMILES separated by periods (ChemDraw format)
                if '.' in line and len(line) > 50:  # Likely multiple SMILES on one line
                    # Split by period - ChemDraw separates multiple SMILES with periods
                    parts = line.split('.')
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        # Validate it looks like SMILES (contains typical SMILES characters)
                        if any(c in part for c in '()[]=+-CNOS0123456789#%'):
                            name, smiles = parse_smiles_line(part)
                            if smiles:
                                smiles_data.append((name, smiles))
                else:
                    # Single SMILES per line (normal format)
                    name, smiles = parse_smiles_line(line)
                    if smiles:
                        smiles_data.append((name, smiles))
            
            if not smiles_data:
                messagebox.showerror('No input', 'Please enter at least one SMILES string.\nFormat: "name:SMILES" or just "SMILES"\nChemDraw format (period-separated) is supported!')
                return
            
            files_data = []
            errors = []
            for idx, (name, smiles) in enumerate(smiles_data, 1):
                try:
                    cm, coords, base = smiles_to_coords(smiles, cfg['CHARGE'], cfg['MULT'], name)
                    cm = cm_override(cm, cfg['CHARGE'], cfg['MULT'])
                    
                    # If no custom name was provided, number the molecule
                    if not name:
                        base = f"{base}_{idx}"
                    
                    # Apply prefix/suffix ADDITION for SMILES (not removal)
                    if cfg['REMOVE_PREFIX']:
                        base = cfg['REMOVE_PREFIX'] + base
                    if cfg['REMOVE_SUFFIX']:
                        base = base + cfg['REMOVE_SUFFIX']
                    
                    files_data.append((base, cm, coords))
                except Exception as e:
                    display_name = name if name else smiles[:50]
                    errors.append(f"Entry {idx} ({display_name}...): {str(e)}")
            
            if errors:
                error_msg = f"Errors processing {len(errors)} SMILES string(s):\n" + "\n".join(errors[:5])
                if len(errors) > 5:
                    error_msg += f"\n... and {len(errors) - 5} more errors"
                if files_data:
                    error_msg += f"\n\nSuccessfully processed {len(files_data)} SMILES string(s). Continue anyway?"
                    if not messagebox.askyesno('SMILES Processing Errors', error_msg):
                        return
                else:
                    messagebox.showerror('SMILES Processing Errors', error_msg)
                    return
        elif cfg['INPUT_TYPE'] == 'log':
            # Handle .log file input
            files = find_geoms(cfg['INPUTS'], input_type='log')
            if not files:
                messagebox.showerror('No inputs', 'No .log files found.'); return
            files_data = []
            errors = []
            for p in files:
                try:
                    cm, coords, base = parse_gaussian_log(p)
                    cm = cm_override(cm, cfg['CHARGE'], cfg['MULT'])
                    # Apply prefix/suffix removal - work with the base name from parse_gaussian_log
                    if cfg['REMOVE_PREFIX'] or cfg['REMOVE_SUFFIX']:
                        # Use the full filename for removal, then extract base name
                        cleaned = remove_prefix_suffix(p.name, cfg['REMOVE_PREFIX'], cfg['REMOVE_SUFFIX'])
                        # Remove extension to get base name
                        if '.' in cleaned:
                            base = cleaned.rsplit('.', 1)[0]
                        else:
                            base = cleaned
                    files_data.append((base, cm, coords))
                except Exception as e:
                    errors.append(f"{p.name}: {str(e)}")
            
            if errors:
                error_msg = f"Errors processing {len(errors)} log file(s):\n" + "\n".join(errors[:5])
                if len(errors) > 5:
                    error_msg += f"\n... and {len(errors) - 5} more errors"
                if files_data:
                    error_msg += f"\n\nSuccessfully processed {len(files_data)} log file(s). Continue anyway?"
                    if not messagebox.askyesno('Log Processing Errors', error_msg):
                        return
                else:
                    messagebox.showerror('Log Processing Errors', error_msg)
                    return
        else:  # 'com'
            # Handle .com file input
            files = find_geoms(cfg['INPUTS'], input_type='com')
            if not files:
                messagebox.showerror('No inputs', 'No .com files found.'); return
            files_data = []
            for p in files:
                base = p.stem
                # Apply prefix/suffix removal
                if cfg['REMOVE_PREFIX'] or cfg['REMOVE_SUFFIX']:
                    base = remove_prefix_suffix(p.name, cfg['REMOVE_PREFIX'], cfg['REMOVE_SUFFIX']).replace('.com', '')
                cm, coords = extract_cm_coords(read_lines(p))
                cm = cm_override(cm, cfg['CHARGE'], cfg['MULT'])
                files_data.append((base, cm, coords))
        
        out = Path(cfg['OUT_DIR']); out.mkdir(exist_ok=True)
        submit_all: List[str] = []
        submit_by_step: dict[int, List[str]] = {k: [] for k in range(1,8)}
        formchk_by_step: dict[int, List[str]] = {k: [] for k in range(1,8)}
        
        # PySOC mode: filename-based, no steps
        if cfg.get('SOC_ENABLE', False):
            for base, cm, coords in files_data:
                # Simple naming: base_functional_basis_solvent_SOC (no step numbers)
                solv_tag_val = solv_tag(cfg['SOLVENT_MODEL'], cfg['SOLVENT_NAME'])
                job = f"{base}_{cfg['FUNCTIONAL']}_{cfg['BASIS']}_{solv_tag_val}_SOC"
                title = f"PySOC {cfg['FUNCTIONAL']}/{cfg['BASIS']}"
                # Use Step 2 route (TD-DFT) for PySOC
                route = route_step2(cfg)
                save_rwf = True  # Always save RWF for PySOC
                com = make_com_inline(job, cfg['NPROC'], cfg['MEM'], route, title, cm, coords, save_rwf=save_rwf)
                write_lines(out / f"{job}.com", com)
                write_lines(out / f"{job}.sh", write_sh(job, cfg))
                sched_cmd = ("qsub " if cfg['SCHEDULER']=="pbs" else ("sbatch " if cfg['SCHEDULER']=="slurm" else "bash ")) + f"{job}.sh"
                submit_all.append(sched_cmd)
        else:
            # Normal step-based mode
            if cfg['MODE'] == 'full':
                steps = list(range(1,8))
            elif cfg['MODE'] == 'multiple':
                steps = cfg['MULTI_STEPS'] if cfg['MULTI_STEPS'] else []
            else:
                steps = [cfg['STEP']]

            for base, cm, coords in files_data:
                if cfg['MODE'] == 'full':
                    jobs = generate_full(base, cm, coords, out, cfg)
                else:
                    jobs = generate_single(base, cm, coords, out, cfg)
                for j in jobs:
                    try:
                        step = int(Path(j).name[:2])
                    except Exception:
                        step = None
                    sched_cmd = ("qsub " if cfg['SCHEDULER']=="pbs" else ("sbatch " if cfg['SCHEDULER']=="slurm" else "bash ")) + f"{j}.sh"
                    submit_all.append(sched_cmd)
                    if step and 1<=step<=7:
                        submit_by_step[step].append(sched_cmd)
                        formchk_by_step[step].append(f"formchk {j}.chk")
            # Write helpers
            for k in range(1,8):
                if submit_by_step[k]: write_exec(out / f'{k:02d}sub.sh', submit_by_step[k])
                if formchk_by_step[k]: write_exec(out / f'{k:02d}formchk.sh', formchk_by_step[k])
        
        # Write submit_all.sh for both modes
        write_exec(out / 'submit_all.sh', submit_all)
        msg = f"Generated jobs for {len(files_data)} input(s) → {out.resolve()}"
        self.status.config(text=msg)
        self.preview.delete('1.0','end'); self.preview.insert('1.0', msg + "\n\n" + "\n".join(submit_all[:200]))
        
        # Open the output folder in Windows Explorer
        try:
            if os.name == 'nt':  # Windows
                os.startfile(str(out.resolve()))
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', str(out.resolve())])
            else:  # Linux
                subprocess.run(['xdg-open', str(out.resolve())])
        except Exception as e:
            # If opening fails, just show the message - don't break the flow
            pass
        
        messagebox.showinfo('Done', msg)
    
    def _run_pysoc(self):
        """Generate PySOC submission scripts for log files"""
        # Ask user to select directory containing log files
        log_dir = filedialog.askdirectory(title='Select directory containing Gaussian .log files')
        if not log_dir:
            return
        
        log_dir = Path(log_dir)
        
        # Find all _SOC.log files (PySOC naming convention)
        log_files = sorted(log_dir.glob("*_SOC.log"))
        
        if not log_files:
            messagebox.showwarning('No log files', f'No *_SOC.log files found in {log_dir}\n\nMake sure your log files end with "_SOC.log"')
            return
        
        # Ask for output directory (where to save the scripts)
        output_dir = filedialog.askdirectory(title='Select directory to save PySOC submission scripts')
        if not output_dir:
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create a single master script that loops through all _SOC.log files
        master_script = output_dir / "run_pysoc.sh"
        script_lines = [
            "#!/bin/bash",
            "# PySOC calculation script for all _SOC.log files",
            "# Generated by Gaussian Steps Generator",
            "#",
            "# INSTRUCTIONS:",
            "# 1. Copy this script to the directory containing your *_SOC.log and *.rwf files",
            "# 2. Make it executable: chmod +x run_pysoc.sh",
            "# 3. Run it: ./run_pysoc.sh",
            "",
            "# Loop through every file ending in _SOC.log",
            "for logfile in *_SOC.log; do",
            "    ",
            "    # 1. Get the base name (remove .log)",
            "    # e.g., \"FLIMBD_16_m062x_def2SVP_dmso_SOC\"",
            "    base_name=\"${logfile%.log}\"",
            "    ",
            "    # 2. Define the matching RWF file name",
            "    rwf_file=\"${base_name}.rwf\"",
            "    ",
            "    # 3. Check if the matching RWF file actually exists",
            "    if [ -f \"$rwf_file\" ]; then",
            "        echo \"Processing $logfile ...\"",
            "        ",
            "        # Run PySOC (Text Output)",
            "        pysoc \"$logfile\" --rwf_file \"$rwf_file\" > \"${base_name}_RESULTS.txt\" 2>&1",
            "        ",
            "        # Run PySOC (CSV Output - Easier for Excel)",
            "        pysoc \"$logfile\" --rwf_file \"$rwf_file\" -c > \"${base_name}_RESULTS.csv\" 2>&1",
            "        ",
            "    else",
            "        echo \"WARNING: Missing .rwf file for $logfile. Skipping...\"",
            "    fi",
            "",
            "done",
            "",
            "echo \"All calculations complete.\"",
        ]
        
        write_exec(master_script, script_lines)
        
        # Create a Python script to combine all CSV results into Excel
        analysis_script = output_dir / "combine_pysoc_results.py"
        analysis_script_lines = [
            "#!/usr/bin/env python3",
            '"""',
            "Combine PySOC Results into Excel",
            "Combines all *_RESULTS.csv files into a single Excel file with separate tabs for each molecule.",
            '"""',
            "",
            "import pandas as pd",
            "import glob",
            "import sys",
            "from pathlib import Path",
            "",
            "def combine_pysoc_results():",
            '    """Combine all PySOC CSV results into a single Excel file"""',
            "    ",
            "    # Find all CSV result files",
            '    csv_files = sorted(glob.glob("*_RESULTS.csv"))',
            "    ",
            "    if not csv_files:",
            '        print("No *_RESULTS.csv files found in current directory.")',
            '        print("Make sure you\'ve run run_pysoc.sh first.")',
            "        sys.exit(1)",
            "    ",
            '    print(f"Found {len(csv_files)} result file(s)...")',
            "    ",
            "    # Create Excel writer",
            '    output_file = "PySOC_Combined_Results.xlsx"',
            "    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:",
            "        ",
            "        # Process each CSV file",
            "        for csv_file in csv_files:",
            "            try:",
            "                # Get molecule name from filename (remove _RESULTS.csv)",
            '                molecule_name = Path(csv_file).stem.replace("_RESULTS", "")',
            "                ",
            "                # Read CSV file",
            "                df = pd.read_csv(csv_file)",
            "                ",
            "                # Clean up column names (remove extra spaces)",
            "                df.columns = df.columns.str.strip()",
            "                ",
            "                # Write to Excel sheet (limit sheet name to 31 chars for Excel)",
            "                sheet_name = molecule_name[:31] if len(molecule_name) > 31 else molecule_name",
            "                ",
            "                # Handle duplicate sheet names",
            "                base_sheet_name = sheet_name",
            "                counter = 1",
            "                while sheet_name in [ws.title for ws in writer.book.worksheets]:",
                    '                    sheet_name = f"{base_sheet_name[:28]}_{counter}"',
            "                    counter += 1",
            "                ",
            "                df.to_excel(writer, sheet_name=sheet_name, index=False)",
                '                print(f"  Added {molecule_name} ({len(df)} rows)")',
            "                ",
            "            except Exception as e:",
                '                print(f"  Error processing {csv_file}: {e}")',
            "                continue",
            "        ",
            "        # Create summary sheet",
            "        summary_data = []",
            "        for csv_file in csv_files:",
            "            try:",
            '                molecule_name = Path(csv_file).stem.replace("_RESULTS", "")',
            "                df = pd.read_csv(csv_file)",
            "                ",
            "                # Calculate some basic statistics",
            "                if 'RSS (cm-1)' in df.columns:",
            "                    max_soc = df['RSS (cm-1)'].max()",
            "                    mean_soc = df['RSS (cm-1)'].mean()",
            "                    n_transitions = len(df)",
            "                else:",
            '                    max_soc = "N/A"',
            '                    mean_soc = "N/A"',
            "                    n_transitions = len(df)",
            "                ",
            "                summary_data.append({",
            "                    'Molecule': molecule_name,",
            "                    'N_Transitions': n_transitions,",
            "                    'Max_SOC_cm-1': max_soc,",
            "                    'Mean_SOC_cm-1': mean_soc,",
            "                    'File': csv_file",
            "                })",
            "            except Exception as e:",
            "                summary_data.append({",
            '                    \'Molecule\': Path(csv_file).stem.replace("_RESULTS", ""),',
            "                    'N_Transitions': 'Error',",
            "                    'Max_SOC_cm-1': 'Error',",
            "                    'Mean_SOC_cm-1': 'Error',",
            "                    'File': csv_file",
            "                })",
            "        ",
            "        if summary_data:",
            "            summary_df = pd.DataFrame(summary_data)",
            "            summary_df.to_excel(writer, sheet_name='Summary', index=False)",
            '            print("  Created Summary sheet")',
            "    ",
            '    print(f"\\nSuccessfully created {output_file}")',
            '    print(f"  Total molecules: {len(csv_files)}")',
            '    print(f"  Open {output_file} in Excel to view all results in separate tabs.")',
            "",
            'if __name__ == "__main__":',
            "    combine_pysoc_results()",
        ]
        
        # Write analysis script
        try:
            with open(analysis_script, 'w', encoding='utf-8', newline='\n') as f:
                f.write('\n'.join(analysis_script_lines))
            # Make it executable on Linux
            if os.name != 'nt':
                os.chmod(analysis_script, 0o755)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to write analysis script: {e}')
            return
        
        # Create a bash script to run the analysis
        analysis_bash = output_dir / "combine_results.sh"
        analysis_bash_lines = [
            "#!/bin/bash",
            "# Combine PySOC results into Excel",
            "# Generated by Gaussian Steps Generator",
            "",
            "# Check if Python is available",
            "if ! command -v python3 &> /dev/null; then",
            "    echo 'Error: python3 not found. Please install Python 3.'",
            "    exit 1",
            "fi",
            "",
            "# Check if required packages are installed",
            "python3 -c 'import pandas, openpyxl' 2>/dev/null",
            "if [ $? -ne 0 ]; then",
            "    echo 'Installing required packages...'",
            "    pip3 install pandas openpyxl",
            "fi",
            "",
            "# Run the combination script",
            "python3 combine_pysoc_results.py",
        ]
        
        write_exec(analysis_bash, analysis_bash_lines)
        
        # Show preview of what was created
        preview_text = f"Generated PySOC scripts:\n\n"
        preview_text += f"1. run_pysoc.sh - Run PySOC calculations\n"
        preview_text += f"2. combine_pysoc_results.py - Combine results into Excel\n"
        preview_text += f"3. combine_results.sh - Run the combination script\n\n"
        preview_text += f"Found {len(log_files)} *_SOC.log file(s):\n"
        for log_file in log_files[:20]:  # Show first 20
            preview_text += f"  • {log_file.name}\n"
        if len(log_files) > 20:
            preview_text += f"  ... and {len(log_files) - 20} more\n"
        
        preview_text += f"\nScript location: {output_dir}\n"
        preview_text += f"Log files directory: {log_dir}\n"
        preview_text += f"\nSTEP 1 - Run PySOC calculations on Linux:\n"
        preview_text += f"  Copy run_pysoc.sh to your log directory\n"
        preview_text += f"  chmod +x run_pysoc.sh\n"
        preview_text += f"  ./run_pysoc.sh\n"
        preview_text += f"\nSTEP 2 - Combine results into Excel:\n"
        preview_text += f"  Copy combine_pysoc_results.py and combine_results.sh to log directory\n"
        preview_text += f"  chmod +x combine_results.sh\n"
        preview_text += f"  ./combine_results.sh\n"
        preview_text += f"\nThis will create PySOC_Combined_Results.xlsx with:\n"
        preview_text += f"  • One tab per molecule (with all SOC transitions)\n"
        preview_text += f"  • Summary tab (statistics for all molecules)\n"
        
        # Display in preview window
        self.preview.delete('1.0', 'end')
        self.preview.insert('1.0', preview_text)
        self.status.config(text=f'Generated PySOC script for {len(log_files)} log files → {output_dir}')
        
        # Open output directory
        try:
            if os.name == 'nt':
                os.startfile(str(output_dir.resolve()))
        except:
            pass
        
        messagebox.showinfo('PySOC Scripts Generated', 
                          f'Generated 3 scripts in:\n{output_dir}\n\n'
                          f'1. run_pysoc.sh - Run PySOC calculations\n'
                          f'2. combine_pysoc_results.py - Combine CSV results into Excel\n'
                          f'3. combine_results.sh - Run the combination script\n\n'
                          f'Found {len(log_files)} *_SOC.log file(s)\n\n'
                          f'Transfer scripts to Linux and follow the instructions in the preview.')

    # -------- prefs --------
    def _save_prefs(self):
        cfg = self._collect()
        cfg['lists'] = dict(func=self.cb_func.values, basis=self.cb_basis.values, smodel=self.cb_smodel.values, sname=self.cb_sname.values, sched=self.cb_sched.values, queue=self.cb_queue.values)
        try:
            with open(PREFS_FILE, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2)
            self.status.config(text=f'Saved preferences → {PREFS_FILE}')
        except Exception as e:
            messagebox.showwarning('Save prefs', str(e))
    
    def _reset_to_defaults(self):
        """Reset all fields to default values"""
        if not messagebox.askyesno('Reset to Defaults', 
                                   'This will reset all fields to default values.\n'
                                   'Continue?'):
            return
        
        # Reset all vars to defaults
        for key, var in self.vars.items():
            if key in DEFAULTS:
                default_val = DEFAULTS[key]
                try:
                    if isinstance(var, tk.BooleanVar):
                        var.set(bool(default_val))
                    elif isinstance(var, tk.IntVar):
                        var.set(int(default_val) if default_val is not None else 0)
                    else:
                        var.set(str(default_val) if default_val is not None else '')
                except Exception:
                    pass
        
        # Reset multi-step selections
        if hasattr(self, 'multi_step_vars'):
            for v in self.multi_step_vars.values():
                v.set(False)
        
        # Reset inline steps
        if hasattr(self, 'inline_vars'):
            for v in self.inline_vars.values():
                v.set(False)
        
        # Reset combo boxes to defaults
        if hasattr(self, 'cb_func'):
            self.cb_func.set(DEFAULTS['FUNCTIONAL'])
        if hasattr(self, 'cb_basis'):
            self.cb_basis.set(DEFAULTS['BASIS'])
        if hasattr(self, 'cb_smodel'):
            self.cb_smodel.set(DEFAULTS['SOLVENT_MODEL'])
        if hasattr(self, 'cb_sname'):
            self.cb_sname.set(DEFAULTS['SOLVENT_NAME'])
        if hasattr(self, 'cb_sched'):
            self.cb_sched.set(DEFAULTS['SCHEDULER'])
        if hasattr(self, 'cb_queue'):
            self.cb_queue.set(DEFAULTS['QUEUE'])
        
        # Update mode and step buttons
        self._on_mode_change()
        self._update_mode_buttons()
        
        self.status.config(text='Reset to defaults')
        messagebox.showinfo('Reset Complete', 'All fields have been reset to default values.')
    
    def _load_prefs(self):
        # seed simple vars
        for k,v in DEFAULTS.items():
            if k not in self.vars and not isinstance(v, list):
                if isinstance(v, bool): self.vars[k] = tk.BooleanVar(value=v)
                elif isinstance(v, int): self.vars[k] = tk.IntVar(value=v)
                else: self.vars[k] = tk.StringVar(value=str(v) if v is not None else '')
        if not PREFS_FILE.exists():
            return
        try:
            with open(PREFS_FILE, 'r', encoding='utf-8') as f: cfg = json.load(f)
            def _restore(cb: EditableCombo, key: str):
                vals = cfg.get('lists', {}).get(key, cb.values); cb.values = vals; cb.combo['values'] = vals
            _restore(self.cb_func, 'func'); _restore(self.cb_basis, 'basis'); _restore(self.cb_smodel, 'smodel'); _restore(self.cb_sname, 'sname'); _restore(self.cb_sched, 'sched'); _restore(self.cb_queue, 'queue')
            # scalars - but always use defaults for INPUTS, OUT_DIR, NPROC, MEM, REMOVE_PREFIX, REMOVE_SUFFIX
            # These should always start fresh
            fields_to_always_default = {'INPUTS', 'OUT_DIR', 'NPROC', 'MEM', 'REMOVE_PREFIX', 'REMOVE_SUFFIX'}
            for key, var in self.vars.items():
                if key in cfg and not isinstance(cfg[key], list):
                    # Skip fields that should always use defaults
                    if key in fields_to_always_default:
                        continue
                    try: var.set(cfg[key])
                    except Exception: pass
            # inline steps
            inline = set(cfg.get('INLINE_STEPS', []))
            for k,v in self.inline_vars.items(): v.set(k in inline)
            
            # Restore multi-step selections (only if explicitly saved, otherwise keep all False)
            if hasattr(self, 'multi_step_vars'):
                multi = set(cfg.get('MULTI_STEPS', []))
                # Only restore if MULTI_STEPS was explicitly in the config, otherwise keep all False
                if 'MULTI_STEPS' in cfg:
                    for k,v in self.multi_step_vars.items(): v.set(k in multi)
                else:
                    # Explicitly set all to False if not in saved prefs
                    for v in self.multi_step_vars.values(): v.set(False)
            
            # Restore manual routes
            if hasattr(self, 'step_route_texts'):
                manual_routes = cfg.get('MANUAL_ROUTES', {})
                for step, route_text in self.step_route_texts.items():
                    if step in manual_routes:
                        route_text.delete('1.0', 'end')
                        route_text.insert('1.0', manual_routes[step])
            
            # Restore geometry sources
            if hasattr(self, 'step_geom_source_vars'):
                geom_sources = cfg.get('GEOM_SOURCE', {})
                for step, var in self.step_geom_source_vars.items():
                    if step in geom_sources:
                        var.set(geom_sources[step])
            
            # Restore redundant coordinates
            if hasattr(self, 'redundant_text'):
                redundant = cfg.get('REDUNDANT_COORDS', '')
                self.redundant_text.delete('1.0', 'end')
                self.redundant_text.insert('1.0', redundant)
            
            # Restore input type and SMILES
            if 'INPUT_TYPE' in cfg:
                if 'INPUT_TYPE' in self.vars:
                    self.vars['INPUT_TYPE'].set(cfg['INPUT_TYPE'])
                    self._on_input_type_change()
            if 'SMILES_INPUT' in cfg:
                if hasattr(self, 'smiles_text'):
                    self.smiles_text.delete('1.0', 'end')
                    self.smiles_text.insert('1.0', cfg['SMILES_INPUT'])
                elif 'SMILES_INPUT' in self.vars:
                    self.vars['SMILES_INPUT'].set(cfg['SMILES_INPUT'])
            
            self._update_visible_routes()
            self.status.config(text=f'Loaded preferences from {PREFS_FILE}')
        except Exception as e:
            messagebox.showwarning('Load prefs', f'Could not load preferences: {e}')

if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
