#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Chemical Calculations Step Maker
----------------------------------------
Unified GUI for generating input files for multiple quantum chemistry software packages.

Supported Software:
- Gaussian (G16/G09)
- ORCA

Features:
- Software selector at the top
- Dynamic UI that adapts to selected software
- Full workflow generation for each software
- SMILES input support (Gaussian only)
- Log file parsing (Gaussian only)
- PySOC integration (Gaussian only)
- Scheduler support (PBS/SLURM/Local)

Run:
  python quantum_steps_gui.py
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import List, Sequence, Tuple, Optional, Dict, Any
import json, re, glob, os, stat, subprocess, sys

# Import Gaussian-specific modules
try:
    from gaussian_steps_gui import (
        DEFAULTS as GAUSSIAN_DEFAULTS,
        RDKIT_AVAILABLE, RDKIT_ERROR,
        XML_AVAILABLE,
        natural_key, read_lines, write_lines, write_exec,
        find_geoms, remove_prefix_suffix, add_redundant_coords,
        extract_cm_coords, parse_gaussian_log, parse_smiles_line,
        smiles_to_coords, extract_names_from_svg, build_scrf,
        route_line, cm_override, make_com_inline, make_com_linked,
        pbs_script, slurm_script, local_script, write_sh,
        td_block, pop_kw, disp_kw, scrf, scrf_clr,
        route_step1, route_step2, route_step3, route_step4,
        route_step5, route_step6, route_step7,
        solv_tag, jobname, step_route, generate_single, generate_full,
        EditableCombo as GaussianEditableCombo,
        PREFS_FILE as GAUSSIAN_PREFS_FILE
    )
    GAUSSIAN_AVAILABLE = True
except ImportError as e:
    GAUSSIAN_AVAILABLE = False
    GAUSSIAN_ERROR = str(e)
    print(f"Warning: Could not import Gaussian modules: {e}")

# ORCA-specific imports and functions
try:
    import xml.etree.ElementTree as ET
    ORCA_XML_AVAILABLE = True
except ImportError:
    ORCA_XML_AVAILABLE = False

# Import TICT rotation module
try:
    from tict_rotation import generate_tict_rotations
    TICT_AVAILABLE = True
except ImportError:
    TICT_AVAILABLE = False

# Import Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    GEMINI_ERROR = "google-generativeai not installed. Install with: pip install google-generativeai"

PREFS_FILE = Path.home() / ".quantum_steps_gui.json"
GEMINI_API_KEY_FILE = Path.home() / ".gemini_api_key.txt"

# ===================== ORCA Backend Logic =====================
# ORCA-specific defaults
ORCA_DEFAULTS = dict(
    MODE="single", STEP=4,
    INPUT_TYPE="xyz",  # xyz, com, log, smiles
    INPUTS="", OUT_DIR="ORCA_Jobs",
    SMILES_INPUT="", REMOVE_PREFIX="", REMOVE_SUFFIX="",
    CHARGE=None, MULT=None,
    METHOD="m06-2x", BASIS="def2-TZVP",
    TD_NROOTS=6, TD_IROOT=1, TD_TDA=True, FOLLOW_IROOT=True,
    SOLVENT_MODEL="none", SOLVENT_NAME="DMSO", EPSILON="",
    NPROCS=8, MAXCORE_MB=4000,
    ORCA_PATH="/path/to/orca",
    SCHEDULER="pbs", QUEUE="normal", WALLTIME="24:00:00",
    PROJECT="", ACCOUNT="",
    CUSTOM_KEYWORDS="! m06-2x def2-TZVP TightSCF", CUSTOM_BLOCK=""
)

# ORCA utility functions
def orca_natural_key(s: str):
    parts = re.split(r"(\d+)", s.lower())
    return tuple(int(p) if p.isdigit() else p for p in parts)

def orca_find_inputs(pattern: str, input_type: str = "xyz") -> List[Path]:
    """Find input files based on type: xyz, com, log"""
    path = Path(pattern)
    extensions = {
        "xyz": [".xyz"],
        "com": [".com"],
        "log": [".log"],
    }.get(input_type, [".xyz", ".com"])
    
    if path.exists() and path.is_dir():
        files = []
        for ext in extensions:
            files.extend(path.glob(f"*{ext}"))
        files = sorted(files, key=lambda x: orca_natural_key(x.name))
    else:
        files = [Path(m) for m in glob.glob(pattern)]
        files = sorted([p for p in files if p.is_file() and p.suffix.lower() in extensions],
                       key=lambda x: orca_natural_key(x.name))
    return files

def orca_parse_com(lines: List[str]) -> Tuple[str, List[str]]:
    """Parse Gaussian .com -> ('charge mult', xyz-lines)."""
    empties = [i for i, L in enumerate(lines) if not L.strip()]
    coords = lines[empties[1] + 1 :] if len(empties) >= 2 else lines[:]
    atom_pat = re.compile(r"^[A-Za-z]{1,2}\s+[-\d]")
    cm = "0 1"
    for i, L in enumerate(coords):
        t = L.strip()
        if not t: continue
        if not atom_pat.match(t):
            if re.match(r"^-?\d+\s+-?\d+$", t):
                cm = t
                coords = coords[i + 1 :]
            break
    while coords and not coords[-1].strip():
        coords.pop()
    return cm, coords

def orca_parse_xyz(lines: List[str]) -> Tuple[str, List[str]]:
    return "0 1", [L for L in lines[2:] if L.strip()]

def orca_extract_geom(p: Path, input_type: str = None) -> Tuple[str, List[str]]:
    """Extract geometry from file based on type"""
    if input_type is None:
        input_type = p.suffix.lower().lstrip('.')
    
    if input_type == "xyz" or p.suffix.lower() == ".xyz":
        return orca_parse_xyz(read_lines(p))
    elif input_type == "log" or p.suffix.lower() == ".log":
        # Use Gaussian log parser
        if GAUSSIAN_AVAILABLE:
            try:
                cm, coords, _ = parse_gaussian_log(p)
                return cm, coords
            except Exception as e:
                raise ValueError(f"Error parsing log file {p}: {e}")
        else:
            raise ValueError("Gaussian modules not available for .log file parsing")
    else:  # .com or default
        return orca_parse_com(read_lines(p))

def orca_cm_tuple(cm: str) -> Tuple[int,int]:
    m = re.match(r"^\s*(-?\d+)\s+(-?\d+)\s*$", cm or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0,1)

def orca_ensure_bang(s: str) -> str:
    s = (s or "").strip()
    return s if s.startswith("!") else ("! " + s if s else "!")

def orca_solvent_tokens_and_block(model: str, name: str, epsilon: float|None):
    """Returns (header_tokens: List[str], extra_block_lines: List[str])."""
    m = (model or "").lower().strip()
    if m in ("", "none", "vac", "vacuum"):
        return [], []
    if m == "smd":
        token = f"SMD({name})" if name else "SMD(water)"
        return [token], []
    if m == "cpcm":
        if epsilon is None:
            token = f"CPCM({name})" if name else "CPCM(water)"
            return [token], []
        block = ["%cpcm", f"  epsilon {float(epsilon)}", "end"]
        return ["CPCM"], block
    return [], []

def orca_pal_block(nprocs: int, maxcore_mb: int) -> List[str]:
    return [f"%pal nprocs {nprocs} end", f"%maxcore {maxcore_mb}"]

def orca_xyz_block(cm: str, coords: List[str]) -> List[str]:
    q, m = orca_cm_tuple(cm)
    return ["", f"* xyz {q} {m}", *coords, "*", ""]

def orca_header_line(method: str, basis: str, base_tokens: List[str]) -> str:
    tokens = [method, basis, "TightSCF", *base_tokens]
    return orca_ensure_bang(" ".join(tokens))

def orca_tddft_block(nroots: int, iroot: int|None=None, tda: bool=True, follow: bool=False) -> List[str]:
    lines = ["%tddft", f"  nroots {int(nroots)}"]
    if iroot is not None:
        lines.append(f"  iroot {int(iroot)}")
        if follow:
            lines.append("  followiroot true")
    if tda is False:
        lines.append("  tda false")
    lines.append("end")
    return lines

def orca_make_inp(job: str, method: str, basis: str, nprocs: int, maxcore_mb: int,
             header_tokens: List[str], body_blocks: List[List[str]],
             cm: str, coords: List[str]) -> List[str]:
    lines = [orca_header_line(method, basis, header_tokens), *orca_pal_block(nprocs, maxcore_mb)]
    for blk in body_blocks:
        if blk:
            lines.extend(blk)
    lines.extend(orca_xyz_block(cm, coords))
    if lines and lines[-1].strip():
        lines.append("")
    return lines

def orca_step1_gs_opt(cfg, cm, coords):
    toks, solv_blk = orca_solvent_tokens_and_block(cfg['SOLVENT_MODEL'], cfg['SOLVENT_NAME'], cfg['EPSILON'])
    blocks = [solv_blk]
    return orca_make_inp("JOB", cfg['METHOD'], cfg['BASIS'], cfg['NPROCS'], cfg['MAXCORE_MB'],
                    toks + ["Opt"], blocks, cm, coords)

def orca_step2_abs(cfg, cm, coords):
    toks, solv_blk = orca_solvent_tokens_and_block(cfg['SOLVENT_MODEL'], cfg['SOLVENT_NAME'], cfg['EPSILON'])
    blocks = [solv_blk, orca_tddft_block(cfg['TD_NROOTS'], None, cfg['TD_TDA'], False)]
    return orca_make_inp("JOB", cfg['METHOD'], cfg['BASIS'], cfg['NPROCS'], cfg['MAXCORE_MB'],
                    toks, blocks, cm, coords)

def orca_step4_es_opt(cfg, cm, coords):
    toks, solv_blk = orca_solvent_tokens_and_block(cfg['SOLVENT_MODEL'], cfg['SOLVENT_NAME'], cfg['EPSILON'])
    blocks = [solv_blk, orca_tddft_block(cfg['TD_NROOTS'], cfg['TD_IROOT'], cfg['TD_TDA'], cfg['FOLLOW_IROOT'])]
    return orca_make_inp("JOB", cfg['METHOD'], cfg['BASIS'], cfg['NPROCS'], cfg['MAXCORE_MB'],
                    toks + ["Opt"], blocks, cm, coords)

def orca_step7_deex(cfg, cm, coords):
    toks, solv_blk = orca_solvent_tokens_and_block(cfg['SOLVENT_MODEL'], cfg['SOLVENT_NAME'], cfg['EPSILON'])
    blocks = [solv_blk, orca_tddft_block(cfg['TD_NROOTS'], None, cfg['TD_TDA'], False)]
    return orca_make_inp("JOB", cfg['METHOD'], cfg['BASIS'], cfg['NPROCS'], cfg['MAXCORE_MB'],
                    toks, blocks, cm, coords)

def orca_step9_custom(cfg, cm, coords):
    hdr = orca_ensure_bang(cfg['CUSTOM_KEYWORDS'])
    lines = [hdr, *orca_pal_block(cfg['NPROCS'], cfg['MAXCORE_MB'])]
    if cfg['CUSTOM_BLOCK'].strip():
        lines.append(cfg['CUSTOM_BLOCK'])
    lines.extend(orca_xyz_block(cm, coords))
    if lines and lines[-1].strip():
        lines.append("")
    return lines

def orca_solv_tag(model: str, name: str) -> str:
    return "vac" if (model or "").lower() in ("none", "", "vac", "vacuum") else (name.lower() if name else "solv")

def orca_jobname(step_num: int, base: str, method: str, basis: str, model: str, name: str) -> str:
    return f"{step_num:02d}{base}_{method}_{basis}_{orca_solv_tag(model, name)}"

def orca_pbs_script(job: str, cfg) -> List[str]:
    total_mem_gb = max(1, (cfg['MAXCORE_MB'] * cfg['NPROCS']) // 1024)
    L = ["#!/bin/bash",
         f"#PBS -q {cfg['QUEUE']}",
         f"#PBS -N {job}",
         f"#PBS -l select=1:ncpus={cfg['NPROCS']}:mpiprocs={cfg['NPROCS']}:mem={total_mem_gb}GB"]
    if cfg['WALLTIME']: L.append(f"#PBS -l walltime={cfg['WALLTIME']}")
    if cfg['PROJECT']:  L.append(f"#PBS -P {cfg['PROJECT']}")
    L += [f"#PBS -o {job}.o", f"#PBS -e {job}.e", "cd $PBS_O_WORKDIR",
          f"{cfg['ORCA_PATH']} {job}.inp > {job}.log"]
    return L

def orca_slurm_script(job: str, cfg) -> List[str]:
    total_mem_gb = max(1, (cfg['MAXCORE_MB'] * cfg['NPROCS']) // 1024)
    L = ["#!/bin/bash",
         f"#SBATCH -J {job}",
         f"#SBATCH -p {cfg['QUEUE']}",
         "#SBATCH -N 1",
         f"#SBATCH --ntasks={cfg['NPROCS']}",
         f"#SBATCH --mem={total_mem_gb}G"]
    if cfg['WALLTIME']: L.append(f"#SBATCH -t {cfg['WALLTIME']}")
    if cfg['ACCOUNT']:  L.append(f"#SBATCH -A {cfg['ACCOUNT']}")
    L += [f"#SBATCH -o {job}.out", f"#SBATCH -e {job}.err",
          f"{cfg['ORCA_PATH']} {job}.inp > {job}.log"]
    return L

def orca_local_script(job: str, cfg) -> List[str]:
    return ["#!/bin/bash", f"{cfg['ORCA_PATH']} {job}.inp > {job}.log &"]

def orca_write_sh(job: str, cfg) -> List[str]:
    if cfg['SCHEDULER'] == "pbs":   return orca_pbs_script(job, cfg)
    if cfg['SCHEDULER'] == "slurm": return orca_slurm_script(job, cfg)
    return orca_local_script(job, cfg)

def orca_build_step(step_num: int, cfg, cm: str, coords: List[str]) -> List[str]:
    if step_num == 1: return orca_step1_gs_opt(cfg, cm, coords)
    if step_num == 2: return orca_step2_abs(cfg, cm, coords)
    if step_num == 4: return orca_step4_es_opt(cfg, cm, coords)
    if step_num == 7: return orca_step7_deex(cfg, cm, coords)
    if step_num == 9: return orca_step9_custom(cfg, cm, coords)
    raise ValueError("Unsupported step")

# ===================== Editable Combo Box =====================
class EditableCombo(ttk.Frame):
    """Combobox that supports typing to filter & add new entries."""
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

    def get(self) -> str: return self.var.get().strip()
    def set(self, v: str): self.var.set(v)

# ===================== Main Application =====================
class QuantumStepsApp:
    def __init__(self, root):
        self.root = root
        root.title("🧬 Quantum Chemical Calculations Step Maker")
        root.geometry("1400x950")
        
        self.software = tk.StringVar(value="gaussian")  # "gaussian" or "orca"
        self.vars = {}
        self.cb_method = None
        self.cb_basis = None
        self.cb_smodel = None
        self.cb_sname = None
        self.cb_sched = None
        self.cb_queue = None
        self.txt_ckw = None
        self.txt_cblk = None
        
        # Performance optimization: debouncing for route updates
        self._route_update_job = None
        self._route_cache = {}
        
        self._setup_styles()
        self._create_header()
        self._create_software_selector()
        self._build_tabs()
        self._load_prefs()
        
        # Initialize with Gaussian tabs visible
        self._on_software_change()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook.Tab', padding=[16, 8])
        style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'))
        style.configure('Card.TLabelframe', padding=10)
        style.configure('Modern.TButton', font=('Segoe UI', 9), padding=(8, 4))
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
        self.root.configure(bg=self.colors['bg'])

    def _create_header(self):
        """Create header with title"""
        self.header_frame = tk.Frame(self.root, bg=self.colors['bg'], height=50)
        self.header_frame.pack(fill='x', padx=10, pady=5)
        title = tk.Label(self.header_frame, text="🧬 Quantum Chemical Calculations Step Maker", 
                        font=('Segoe UI', 18, 'bold'), bg=self.colors['bg'], 
                        fg=self.colors['primary'])
        title.pack(side='left', padx=10)

    def _create_software_selector(self):
        """Create software selection frame at the top"""
        selector_frame = tk.Frame(self.root, bg=self.colors['bg'], height=60)
        selector_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(selector_frame, text="Software:", font=('Segoe UI', 12, 'bold'),
                bg=self.colors['bg'], fg=self.colors['text']).pack(side='left', padx=10)
        
        software_frame = tk.Frame(selector_frame, bg=self.colors['bg'])
        software_frame.pack(side='left', padx=10)
        
        for label, value in [("Gaussian", "gaussian"), ("ORCA", "orca")]:
            btn = tk.Radiobutton(software_frame, text=label, variable=self.software, 
                               value=value, font=('Segoe UI', 11),
                               bg=self.colors['bg'], fg=self.colors['text'],
                               selectcolor=self.colors['card'],
                               command=self._on_software_change,
                               indicatoron=True)
            btn.pack(side='left', padx=15)
        
        # Status label
        self.software_status = tk.Label(selector_frame, text="", font=('Segoe UI', 10),
                                       bg=self.colors['bg'], fg=self.colors['text'])
        self.software_status.pack(side='left', padx=20)

    def _on_software_change(self):
        """Handle software selection change"""
        software = self.software.get()
        
        # Update status
        if software == "gaussian":
            if GAUSSIAN_AVAILABLE:
                self.software_status.config(text="✓ Gaussian modules loaded", fg='green')
            else:
                self.software_status.config(text="✗ Gaussian modules not available", fg='red')
        else:  # orca
            self.software_status.config(text="✓ ORCA support active", fg='green')
        
        # Rebuild tabs based on software
        self._rebuild_tabs()

    def _rebuild_tabs(self):
        """Rebuild tabs based on selected software"""
        # Destroy existing notebook
        if hasattr(self, 'notebook'):
            self.notebook.destroy()
        
        # Create new notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Build software-specific tabs
        if self.software.get() == "gaussian":
            self._build_gaussian_tabs()
        else:  # orca
            self._build_orca_tabs()

    def _build_tabs(self):
        """Initial tab building - will be replaced by _rebuild_tabs"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        self._build_gaussian_tabs()

    def _build_gaussian_tabs(self):
        """Build Gaussian-specific tabs by creating a wrapper around Gaussian App"""
        if not GAUSSIAN_AVAILABLE:
            f = ttk.Frame(self.notebook)
            self.notebook.add(f, text="📋 Gaussian Settings")
            error_msg = f"Gaussian modules not available.\n\nError: {GAUSSIAN_ERROR if 'GAUSSIAN_ERROR' in globals() else 'Import failed'}\n\nPlease ensure gaussian_steps_gui.py is in the same directory."
            tk.Label(f, text=error_msg, font=('Segoe UI', 11), 
                    fg='red', justify='left').pack(pady=50, padx=20)
            return
        
        # Create a temporary root to instantiate Gaussian App
        # We'll extract its tabs and add them to our notebook
        try:
            from gaussian_steps_gui import App as GaussianApp
            
            # Create a temporary frame to hold the Gaussian app
            temp_root = tk.Toplevel()
            temp_root.withdraw()  # Hide it
            
            # Create Gaussian app instance
            gaussian_app = GaussianApp(temp_root)
            
            # Extract the notebook from Gaussian app
            # The Gaussian app uses a different structure, so we need to adapt
            # Instead, let's create our own Gaussian tabs using the imported functions
            self._create_gaussian_tabs_direct()
            
            # Clean up temp root
            temp_root.destroy()
            
        except Exception as e:
            # Fallback: create tabs directly
            self._create_gaussian_tabs_direct()
    
    def _create_gaussian_tabs_direct(self):
        """Create Gaussian tabs directly using imported functions - FULL INTEGRATION"""
        # Initialize Gaussian-specific variables if not already done
        if not hasattr(self, 'gaussian_vars_initialized'):
            self.gaussian_vars_initialized = True
            self.multi_step_vars = {}
            self.step_route_texts = {}
            self.step_route_frames = {}
            self.step_geom_source_vars = {}
            self.inline_vars = {}
        
        # Build all Gaussian tabs including TICT and AI Assistant
        self._gaussian_tab_main()
        self._gaussian_tab_advanced()
        self._gaussian_tab_tict()
        self._gaussian_tab_ai_assistant()
        self._gaussian_tab_generate()
    
    def _gaussian_create_card(self, parent, title):
        """Create a styled card frame for Gaussian tabs"""
        card = tk.Frame(parent, bg=self.colors['card'], relief='flat', borderwidth=1,
                       highlightbackground=self.colors['border'], highlightthickness=1)
        if title:
            tk.Label(card, text=title, font=('Segoe UI', 10, 'bold'),
                    bg=self.colors['card'], fg=self.colors['text']).pack(anchor='w', padx=10, pady=(8,5))
        return card
    
    def _orca_create_card(self, parent, title):
        """Create a styled card frame for ORCA tabs (same style as Gaussian)"""
        return self._gaussian_create_card(parent, title)
    
    # ========== Gaussian Tab Methods ==========
    def _gaussian_tab_main(self):
        """Build Gaussian Main Settings tab - FULL INTEGRATION"""
        f = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(f, text='⚙️ Main Settings')
        
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
        
        # Enable mousewheel scrolling - bind to canvas and scrollable frame
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def on_enter(event):
            canvas.focus_set()
        # Bind to both canvas and scrollable frame for better coverage
        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Enter>", on_enter)
        scrollable_frame.bind("<MouseWheel>", on_mousewheel)
        scrollable_frame.bind("<Enter>", on_enter)
        # Also bind to the frame itself
        f.bind("<MouseWheel>", lambda e: on_mousewheel(e))
        f.bind("<Enter>", lambda e: on_enter(e))
        
        # Mode and Steps Section
        mode_card = self._gaussian_create_card(scrollable_frame, 'Mode & Steps')
        mode_card.pack(fill='x', padx=15, pady=8)
        
        mode_row = tk.Frame(mode_card, bg=self.colors['card'])
        mode_row.pack(fill='x', padx=10, pady=(0,10))
        tk.Label(mode_row, text='Mode:', font=('Segoe UI', 10), 
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        
        if 'MODE' not in self.vars:
            self.vars['MODE'] = tk.StringVar(value='single')
        mode_buttons_frame = tk.Frame(mode_row, bg=self.colors['card'])
        mode_buttons_frame.pack(side='left')
        
        for label, value in [('Full (1-7)', 'full'), ('Single', 'single'), ('Multiple', 'multiple')]:
            btn = tk.Button(mode_buttons_frame, text=label, font=('Segoe UI', 9, 'bold'),
                          width=10, height=1, bg=self.colors['card'], fg=self.colors['text'],
                          relief='raised', borderwidth=1, command=lambda v=value: self._gaussian_set_mode(v),
                          cursor='hand2')
            btn.pack(side='left', padx=3)
            setattr(self, f'_mode_btn_{value}', btn)
        self.mode_buttons_frame = mode_buttons_frame
        self.mode_card = mode_card
        
        step_row = tk.Frame(mode_card, bg=self.colors['card'])
        step_row.pack(fill='x', padx=10, pady=(0,10))
        tk.Label(step_row, text='Steps:', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        
        self.single_step_frame = tk.Frame(step_row, bg=self.colors['card'])
        self.single_step_frame.pack(side='left')
        if 'STEP' not in self.vars:
            self.vars['STEP'] = tk.IntVar(value=GAUSSIAN_DEFAULTS['STEP'])
        step_buttons = tk.Frame(self.single_step_frame, bg=self.colors['card'])
        step_buttons.pack(side='left')
        for k in (1,2,3,4,5,6,7):
            btn = tk.Button(step_buttons, text=str(k), width=3, height=1,
                          font=('Segoe UI', 9), bg=self.colors['card'], fg=self.colors['text'],
                          relief='raised', borderwidth=1, command=lambda v=k: self._gaussian_set_single_step(v),
                          cursor='hand2')
            btn.pack(side='left', padx=2)
            setattr(self, f'_step_btn_{k}', btn)
        
        self.multi_step_frame = tk.Frame(step_row, bg=self.colors['card'])
        if not hasattr(self, 'multi_step_vars') or not self.multi_step_vars:
            self.multi_step_vars = {}
        multi_buttons = tk.Frame(self.multi_step_frame, bg=self.colors['card'])
        multi_buttons.pack(side='left')
        for k in (1,2,3,4,5,6,7):
            if k not in self.multi_step_vars:
                var = tk.BooleanVar()
                self.multi_step_vars[k] = var
            btn = tk.Button(multi_buttons, text=str(k), width=3, height=1,
                          font=('Segoe UI', 9), bg=self.colors['card'], fg=self.colors['text'],
                          relief='raised', borderwidth=1, command=lambda v=k: self._gaussian_toggle_multi_step(v),
                          cursor='hand2')
            btn.pack(side='left', padx=2)
            setattr(self, f'_multi_btn_{k}', btn)
        self.multi_step_frame.pack_forget()
        self.step_row = step_row
        self._gaussian_update_mode_buttons()
        
        # Two-column layout
        content_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        content_frame.pack(fill='both', expand=True, padx=15, pady=5)
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        
        # Left Column
        left_col = tk.Frame(content_frame, bg=self.colors['bg'])
        left_col.grid(row=0, column=0, sticky='nsew', padx=(0,8))
        
        # Input/Output
        io_card = self._gaussian_create_card(left_col, 'Input/Output Files')
        io_card.pack(fill='x', pady=8)
        
        # Input type selection
        input_type_row = tk.Frame(io_card, bg=self.colors['card'])
        input_type_row.pack(fill='x', padx=10, pady=(0,5))
        tk.Label(input_type_row, text='Input Type:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        if 'INPUT_TYPE' not in self.vars:
            self.vars['INPUT_TYPE'] = tk.StringVar(value=GAUSSIAN_DEFAULTS['INPUT_TYPE'])
        input_type_frame = tk.Frame(input_type_row, bg=self.colors['card'])
        input_type_frame.pack(side='left')
        tk.Radiobutton(input_type_frame, text='.com Files', variable=self.vars['INPUT_TYPE'], value='com',
                      font=('Segoe UI', 9), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text'], command=self._gaussian_on_input_type_change).pack(side='left', padx=5)
        tk.Radiobutton(input_type_frame, text='.log Files', variable=self.vars['INPUT_TYPE'], value='log',
                      font=('Segoe UI', 9), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text'], command=self._gaussian_on_input_type_change).pack(side='left', padx=5)
        tk.Radiobutton(input_type_frame, text='SMILES', variable=self.vars['INPUT_TYPE'], value='smiles',
                      font=('Segoe UI', 9), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text'], command=self._gaussian_on_input_type_change).pack(side='left', padx=5)
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
        self.gaussian_input_label = tk.Label(input_row, text='Input (.com files or folder):', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text'])
        self.gaussian_input_label.grid(row=0, column=0, sticky='w', pady=3)
        if 'INPUTS' not in self.vars:
            self.vars['INPUTS'] = tk.StringVar(value=GAUSSIAN_DEFAULTS['INPUTS'])
        self.gaussian_input_entry = tk.Entry(input_row, textvariable=self.vars['INPUTS'], font=('Segoe UI', 9),
                         bg='white', fg=self.colors['text'], relief='flat', borderwidth=1,
                         highlightthickness=1, highlightbackground=self.colors['border'])
        self.gaussian_input_entry.grid(row=1, column=0, sticky='ew', padx=(0,5))
        self.gaussian_browse_btn = tk.Button(input_row, text='Browse', command=self._gaussian_browse_inputs,
                 font=('Segoe UI', 8), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=10, pady=3, cursor='hand2')
        self.gaussian_browse_btn.grid(row=1, column=1)
        self.gaussian_input_row = input_row
        
        # SMILES input row
        smiles_row = tk.Frame(io_card, bg=self.colors['card'])
        smiles_row.pack(fill='both', expand=True, padx=10, pady=(0,8))
        smiles_row.columnconfigure(0, weight=1)
        smiles_row.rowconfigure(1, weight=1)
        label_frame = tk.Frame(smiles_row, bg=self.colors['card'])
        label_frame.grid(row=0, column=0, sticky='ew', pady=3)
        tk.Label(label_frame, text='SMILES String(s) - one per line:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left')
        
        help_btn = tk.Button(label_frame, text='?', font=('Segoe UI', 8, 'bold'),
                           bg=self.colors['primary'], fg='white', width=2, height=1,
                           relief='flat', cursor='hand2', command=self._gaussian_show_chemdraw_help)
        help_btn.pack(side='right', padx=(5,0))
        
        load_names_btn = tk.Button(label_frame, text='Load Names from SVG', font=('Segoe UI', 8),
                           bg=self.colors['accent'], fg='white',
                           relief='flat', padx=8, pady=2, cursor='hand2',
                           command=self._gaussian_load_names_from_svg)
        load_names_btn.pack(side='right', padx=(5,0))
        
        help_text = "1. Paste SMILES → 2. Export ChemDraw as SVG → 3. Click 'Load Names from SVG'"
        tk.Label(label_frame, text=help_text, font=('Segoe UI', 7, 'italic'),
                bg=self.colors['card'], fg=self.colors['text_light']).pack(side='right', padx=(10,5))
        self.gaussian_smiles_text = scrolledtext.ScrolledText(smiles_row, height=6, wrap='none', font=('Consolas', 9),
                         bg='white', fg='black', relief='flat', borderwidth=1,
                         highlightthickness=1, highlightbackground=self.colors['border'])
        self.gaussian_smiles_text.grid(row=1, column=0, sticky='nsew', pady=3)
        if GAUSSIAN_DEFAULTS.get('SMILES_INPUT'):
            self.gaussian_smiles_text.insert('1.0', GAUSSIAN_DEFAULTS['SMILES_INPUT'])
        self.gaussian_smiles_row = smiles_row
        
        # Initially show/hide rows based on input type
        if self.vars['INPUT_TYPE'].get() == 'smiles':
            self.gaussian_input_row.pack_forget()
        else:
            self.gaussian_smiles_row.pack_forget()
            if self.vars['INPUT_TYPE'].get() == 'log':
                self.gaussian_input_label.config(text='Input (.log files or folder):')
        
        prefix_row = tk.Frame(io_card, bg=self.colors['card'])
        prefix_row.pack(fill='x', padx=10, pady=(0,5))
        self.gaussian_prefix_label = tk.Label(prefix_row, text='Remove Prefix:', font=('Segoe UI', 8),
                bg=self.colors['card'], fg=self.colors['text'])
        self.gaussian_prefix_label.grid(row=0, column=0, sticky='w', padx=(0,5))
        if 'REMOVE_PREFIX' not in self.vars:
            self.vars['REMOVE_PREFIX'] = tk.StringVar(value=GAUSSIAN_DEFAULTS['REMOVE_PREFIX'])
        self.gaussian_prefix_entry = tk.Entry(prefix_row, textvariable=self.vars['REMOVE_PREFIX'], width=15,
                font=('Segoe UI', 8), bg='white', fg=self.colors['text'],
                relief='flat', borderwidth=1, highlightthickness=1,
                highlightbackground=self.colors['border'])
        self.gaussian_prefix_entry.grid(row=0, column=1, padx=5)
        self.gaussian_suffix_label = tk.Label(prefix_row, text='Remove Suffix:', font=('Segoe UI', 8),
                bg=self.colors['card'], fg=self.colors['text'])
        self.gaussian_suffix_label.grid(row=0, column=2, sticky='w', padx=(10,5))
        if 'REMOVE_SUFFIX' not in self.vars:
            self.vars['REMOVE_SUFFIX'] = tk.StringVar(value=GAUSSIAN_DEFAULTS['REMOVE_SUFFIX'])
        self.gaussian_suffix_entry = tk.Entry(prefix_row, textvariable=self.vars['REMOVE_SUFFIX'], width=15,
                font=('Segoe UI', 8), bg='white', fg=self.colors['text'],
                relief='flat', borderwidth=1, highlightthickness=1,
                highlightbackground=self.colors['border'])
        self.gaussian_suffix_entry.grid(row=0, column=3)
        
        output_row = tk.Frame(io_card, bg=self.colors['card'])
        output_row.pack(fill='x', padx=10, pady=(0,10))
        output_row.columnconfigure(0, weight=1)
        tk.Label(output_row, text='Output:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=0, column=0, sticky='w', pady=3)
        if 'OUT_DIR' not in self.vars:
            self.vars['OUT_DIR'] = tk.StringVar(value=GAUSSIAN_DEFAULTS['OUT_DIR'])
        entry2 = tk.Entry(output_row, textvariable=self.vars['OUT_DIR'], font=('Segoe UI', 9),
                         bg='white', fg=self.colors['text'], relief='flat', borderwidth=1,
                         highlightthickness=1, highlightbackground=self.colors['border'])
        entry2.grid(row=1, column=0, sticky='ew', padx=(0,5))
        tk.Button(output_row, text='Browse', command=self._gaussian_browse_outdir,
                 font=('Segoe UI', 8), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=10, pady=3, cursor='hand2').grid(row=1, column=1)
        
        # Help Section - Workflow Steps (in left column) - Enhanced with colors
        help_card = self._gaussian_create_card(left_col, '📚 Workflow Steps Guide')
        help_card.pack(fill='x', pady=8)
        help_content = tk.Frame(help_card, bg=self.colors['card'])
        help_content.pack(fill='x', padx=10, pady=(0,10))
        
        # Colorful step descriptions
        steps_data = [
            (1, "Ground State Optimization", "Geometry optimization in the ground state", "#2E7D32"),
            (2, "Vertical Excitation", "Vertical excitation at the same geometry (Franck-Condon state)", "#1976D2"),
            (3, "cLR Correction", "Correction of the vertical excitation energy using corrected Linear Response (cLR) theory", "#7B1FA2"),
            (4, "Excited State Optimization", "Geometry optimization in the excited state", "#F57C00"),
            (5, "Density Calculation", "Density calculations at the optimized excited state geometry", "#C2185B"),
            (6, "cLR Correction (ES)", "cLR correction of excited state energy", "#7B1FA2"),
            (7, "Ground State at ES Geometry", "Ground state energy calculation at the excited state geometry", "#2E7D32"),
        ]
        
        for step_num, step_name, step_desc, color in steps_data:
            step_frame = tk.Frame(help_content, bg=self.colors['card'])
            step_frame.pack(fill='x', pady=2)
            
            step_num_label = tk.Label(step_frame, text=f"Step {step_num}", 
                                     font=('Segoe UI', 9, 'bold'),
                                     bg=color, fg='white',
                                     width=8, anchor='center', padx=5, pady=2)
            step_num_label.pack(side='left', padx=(0,8))
            
            step_text = f"{step_name}: {step_desc}"
            step_desc_label = tk.Label(step_frame, text=step_text,
                                      font=('Segoe UI', 8),
                                      bg=self.colors['card'], fg=self.colors['text'],
                                      justify='left', wraplength=300, anchor='w')
            step_desc_label.pack(side='left', fill='x', expand=True)
        
        # Method/Basis
        method_card = self._gaussian_create_card(left_col, 'Method & Basis Set')
        method_card.pack(fill='x', pady=8)
        method_content = tk.Frame(method_card, bg=self.colors['card'])
        method_content.pack(fill='x', padx=10, pady=(0,10))
        method_content.columnconfigure(1, weight=1)
        
        tk.Label(method_content, text='Functional:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=0, column=0, sticky='w', pady=3)
        # Always create new widget to avoid parent frame issues
        if hasattr(self, 'cb_func') and self.cb_func is not None:
            try:
                self.cb_func.destroy()
            except:
                pass
        self.cb_func = GaussianEditableCombo(method_content, ["m062x","b3lyp","wb97xd","cam-b3lyp","pbe0","tpss","bp86","scan"])
        self.cb_func.set(GAUSSIAN_DEFAULTS['FUNCTIONAL'])
        self.cb_func.grid(row=0, column=1, sticky='ew', padx=(10,0), pady=3)
        
        tk.Label(method_content, text='Basis:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=1, column=0, sticky='w', pady=3)
        # Always create new widget to avoid parent frame issues
        if hasattr(self, 'cb_basis') and self.cb_basis is not None:
            try:
                self.cb_basis.destroy()
            except:
                pass
        self.cb_basis = GaussianEditableCombo(method_content, ["def2SVP","def2TZVP","6-31G*","6-311+G**","cc-pVDZ","cc-pVTZ","aug-cc-pVDZ"])
        self.cb_basis.set(GAUSSIAN_DEFAULTS['BASIS'])
        self.cb_basis.grid(row=1, column=1, sticky='ew', padx=(10,0), pady=3)
        
        # Solvent
        solvent_card = self._gaussian_create_card(left_col, 'Solvent Model')
        solvent_card.pack(fill='x', pady=8)
        solvent_content = tk.Frame(solvent_card, bg=self.colors['card'])
        solvent_content.pack(fill='x', padx=10, pady=(0,10))
        solvent_content.columnconfigure(1, weight=1)
        
        tk.Label(solvent_content, text='Model:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=0, column=0, sticky='w', pady=3)
        # Always create new widget to avoid parent frame issues
        if hasattr(self, 'cb_smodel') and self.cb_smodel is not None:
            try:
                self.cb_smodel.destroy()
            except:
                pass
        self.cb_smodel = GaussianEditableCombo(solvent_content, ["none","SMD","PCM","IEFPCM","CPCM"])
        self.cb_smodel.set(GAUSSIAN_DEFAULTS['SOLVENT_MODEL'])
        self.cb_smodel.grid(row=0, column=1, sticky='ew', padx=(10,0), pady=3)
        
        tk.Label(solvent_content, text='Solvent:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=1, column=0, sticky='w', pady=3)
        # Always create new widget to avoid parent frame issues
        if hasattr(self, 'cb_sname') and self.cb_sname is not None:
            try:
                self.cb_sname.destroy()
            except:
                pass
        self.cb_sname = GaussianEditableCombo(solvent_content, ["DMSO","Water","Acetonitrile","Methanol","Ethanol","DCM","THF","Toluene","Benzene","Acetone","DMF"])
        self.cb_sname.set(GAUSSIAN_DEFAULTS['SOLVENT_NAME'])
        self.cb_sname.grid(row=1, column=1, sticky='ew', padx=(10,0), pady=3)
        
        # Right Column
        right_col = tk.Frame(content_frame, bg=self.colors['bg'])
        right_col.grid(row=0, column=1, sticky='nsew', padx=(8,0))
        
        # TD-DFT
        td_card = self._gaussian_create_card(right_col, 'TD-DFT Settings')
        td_card.pack(fill='x', pady=8)
        td_content = tk.Frame(td_card, bg=self.colors['card'])
        td_content.pack(fill='x', padx=10, pady=(0,10))
        
        td_row1 = tk.Frame(td_content, bg=self.colors['card'])
        td_row1.pack(fill='x', pady=3)
        tk.Label(td_row1, text='NStates:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        if 'TD_NSTATES' not in self.vars:
            self.vars['TD_NSTATES'] = tk.IntVar(value=GAUSSIAN_DEFAULTS['TD_NSTATES'])
        spin1 = ttk.Spinbox(td_row1, from_=1, to=128, textvariable=self.vars['TD_NSTATES'], width=8)
        spin1.pack(side='left')
        
        tk.Label(td_row1, text='Root:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(20,10))
        if 'TD_ROOT' not in self.vars:
            self.vars['TD_ROOT'] = tk.IntVar(value=GAUSSIAN_DEFAULTS['TD_ROOT'])
        spin2 = ttk.Spinbox(td_row1, from_=1, to=128, textvariable=self.vars['TD_ROOT'], width=8)
        spin2.pack(side='left')
        
        # State Type selection
        td_row2 = tk.Frame(td_content, bg=self.colors['card'])
        td_row2.pack(fill='x', pady=3)
        tk.Label(td_row2, text='State Type:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        if 'STATE_TYPE' not in self.vars:
            self.vars['STATE_TYPE'] = tk.StringVar(value=GAUSSIAN_DEFAULTS['STATE_TYPE'])
        state_frame = tk.Frame(td_row2, bg=self.colors['card'])
        state_frame.pack(side='left')
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
        if 'SOC_ENABLE' not in self.vars:
            self.vars['SOC_ENABLE'] = tk.BooleanVar(value=GAUSSIAN_DEFAULTS['SOC_ENABLE'])
        soc_cb = tk.Checkbutton(td_content, text='Prepare for PySOC (saves RWF, adds 6D 10F GFInput)', 
                      variable=self.vars['SOC_ENABLE'],
                      font=('Segoe UI', 8), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text'],
                      command=self._gaussian_on_soc_enable_change)
        soc_cb.pack(anchor='w', pady=3)
        self.root.after(100, self._gaussian_on_soc_enable_change)
        
        soc_info = tk.Label(td_content, 
                           text='Step 1: Check this, then Generate → creates .com/.sh with %Rwf and 6D 10F GFInput\n'
                                'Step 2: After Gaussian jobs complete, use "Generate PySOC Scripts" to create SOC calculation scripts', 
                           font=('Segoe UI', 7, 'italic'), bg=self.colors['card'], fg=self.colors['text_light'],
                           justify='left')
        soc_info.pack(anchor='w', padx=(20,0), pady=(0,5))
        
        if 'POP_FULL' not in self.vars:
            self.vars['POP_FULL'] = tk.BooleanVar(value=GAUSSIAN_DEFAULTS['POP_FULL'])
        tk.Checkbutton(td_content, text='pop=(full,orbitals=2)', variable=self.vars['POP_FULL'],
                      font=('Segoe UI', 8), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text']).pack(anchor='w', pady=3)
        if 'DISPERSION' not in self.vars:
            self.vars['DISPERSION'] = tk.BooleanVar(value=GAUSSIAN_DEFAULTS['DISPERSION'])
        tk.Checkbutton(td_content, text='EmpiricalDispersion=GD3BJ', variable=self.vars['DISPERSION'],
                      font=('Segoe UI', 8), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text']).pack(anchor='w', pady=3)
        
        # Resources
        res_card = self._gaussian_create_card(right_col, 'Computational Resources')
        res_card.pack(fill='x', pady=8)
        res_content = tk.Frame(res_card, bg=self.colors['card'])
        res_content.pack(fill='x', padx=10, pady=(0,10))
        res_content.columnconfigure(1, weight=1)
        
        tk.Label(res_content, text='Cores:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=0, column=0, sticky='w', pady=3)
        if 'NPROC' not in self.vars:
            self.vars['NPROC'] = tk.IntVar(value=GAUSSIAN_DEFAULTS['NPROC'])
        ttk.Spinbox(res_content, from_=1, to=256, textvariable=self.vars['NPROC'], width=10).grid(row=0, column=1, sticky='w', padx=(10,0), pady=3)
        
        tk.Label(res_content, text='Memory:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=1, column=0, sticky='w', pady=3)
        if 'MEM' not in self.vars:
            self.vars['MEM'] = tk.StringVar(value=GAUSSIAN_DEFAULTS['MEM'])
        tk.Entry(res_content, textvariable=self.vars['MEM'], font=('Segoe UI', 9),
                bg='white', fg=self.colors['text'], relief='flat', borderwidth=1,
                highlightthickness=1, highlightbackground=self.colors['border']).grid(row=1, column=1, sticky='ew', padx=(10,0), pady=3)
        
        # Scheduler
        sched_card = self._gaussian_create_card(right_col, 'Scheduler Settings')
        sched_card.pack(fill='x', pady=8)
        sched_content = tk.Frame(sched_card, bg=self.colors['card'])
        sched_content.pack(fill='x', padx=10, pady=(0,10))
        sched_content.columnconfigure(1, weight=1)
        
        tk.Label(sched_content, text='Type:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=0, column=0, sticky='w', pady=3)
        # Always create new widget to avoid parent frame issues
        if hasattr(self, 'cb_sched') and self.cb_sched is not None:
            try:
                self.cb_sched.destroy()
            except:
                pass
        self.cb_sched = GaussianEditableCombo(sched_content, ["pbs","slurm","local"])
        self.cb_sched.set(GAUSSIAN_DEFAULTS['SCHEDULER'])
        self.cb_sched.grid(row=0, column=1, sticky='ew', padx=(10,0), pady=3)
        
        tk.Label(sched_content, text='Queue:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=1, column=0, sticky='w', pady=3)
        # Always create new widget to avoid parent frame issues
        if hasattr(self, 'cb_queue') and self.cb_queue is not None:
            try:
                self.cb_queue.destroy()
            except:
                pass
        self.cb_queue = GaussianEditableCombo(sched_content, ["normal","express","long","gpu","debug"])
        self.cb_queue.set(GAUSSIAN_DEFAULTS['QUEUE'])
        self.cb_queue.grid(row=1, column=1, sticky='ew', padx=(10,0), pady=3)
        
        tk.Label(sched_content, text='Walltime:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=2, column=0, sticky='w', pady=3)
        if 'WALLTIME' not in self.vars:
            self.vars['WALLTIME'] = tk.StringVar(value=GAUSSIAN_DEFAULTS['WALLTIME'])
        tk.Entry(sched_content, textvariable=self.vars['WALLTIME'], font=('Segoe UI', 9),
                bg='white', fg=self.colors['text'], relief='flat', borderwidth=1,
                highlightthickness=1, highlightbackground=self.colors['border']).grid(row=2, column=1, sticky='ew', padx=(10,0), pady=3)
        
        tk.Label(sched_content, text='Project:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=3, column=0, sticky='w', pady=3)
        if 'PROJECT' not in self.vars:
            self.vars['PROJECT'] = tk.StringVar(value=GAUSSIAN_DEFAULTS['PROJECT'])
        tk.Entry(sched_content, textvariable=self.vars['PROJECT'], font=('Segoe UI', 9),
                bg='white', fg=self.colors['text'], relief='flat', borderwidth=1,
                highlightthickness=1, highlightbackground=self.colors['border']).grid(row=3, column=1, sticky='ew', padx=(10,0), pady=3)
        
        tk.Label(sched_content, text='Account:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=4, column=0, sticky='w', pady=3)
        if 'ACCOUNT' not in self.vars:
            self.vars['ACCOUNT'] = tk.StringVar(value=GAUSSIAN_DEFAULTS['ACCOUNT'])
        tk.Entry(sched_content, textvariable=self.vars['ACCOUNT'], font=('Segoe UI', 9),
                bg='white', fg=self.colors['text'], relief='flat', borderwidth=1,
                highlightthickness=1, highlightbackground=self.colors['border']).grid(row=4, column=1, sticky='ew', padx=(10,0), pady=3)
        
        # Charge/Multiplicity
        cm_card = self._gaussian_create_card(right_col, 'Charge & Multiplicity')
        cm_card.pack(fill='x', pady=8)
        cm_content = tk.Frame(cm_card, bg=self.colors['card'])
        cm_content.pack(fill='x', padx=10, pady=(0,10))
        
        cm_row = tk.Frame(cm_content, bg=self.colors['card'])
        cm_row.pack(fill='x', pady=3)
        tk.Label(cm_row, text='Charge:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        if 'CHARGE' not in self.vars:
            self.vars['CHARGE'] = tk.StringVar(value='' if GAUSSIAN_DEFAULTS['CHARGE'] is None else str(GAUSSIAN_DEFAULTS['CHARGE']))
        tk.Entry(cm_row, textvariable=self.vars['CHARGE'], width=8,
                font=('Segoe UI', 9), bg='white', fg=self.colors['text'],
                relief='flat', borderwidth=1, highlightthickness=1,
                highlightbackground=self.colors['border']).pack(side='left')
        
        tk.Label(cm_row, text='Multiplicity:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(20,10))
        if 'MULT' not in self.vars:
            self.vars['MULT'] = tk.StringVar(value='' if GAUSSIAN_DEFAULTS['MULT'] is None else str(GAUSSIAN_DEFAULTS['MULT']))
        tk.Entry(cm_row, textvariable=self.vars['MULT'], width=8,
                font=('Segoe UI', 9), bg='white', fg=self.colors['text'],
                relief='flat', borderwidth=1, highlightthickness=1,
                highlightbackground=self.colors['border']).pack(side='left')
        
        # Inline Steps
        inline_card = self._gaussian_create_card(right_col, 'Inline Coordinates')
        inline_card.pack(fill='x', pady=8)
        inline_content = tk.Frame(inline_card, bg=self.colors['card'])
        inline_content.pack(fill='x', padx=10, pady=(0,10))
        tk.Label(inline_content, text='Copy coords for steps:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(anchor='w', pady=3)
        inline_buttons = tk.Frame(inline_content, bg=self.colors['card'])
        inline_buttons.pack(fill='x')
        if not hasattr(self, 'inline_vars') or not self.inline_vars:
            self.inline_vars = {}
        for k in (2,3,4,5,6,7):
            if k not in self.inline_vars:
                v = tk.BooleanVar(value=k in (GAUSSIAN_DEFAULTS.get('INLINE_STEPS') or []))
                self.inline_vars[k] = v
            tk.Checkbutton(inline_buttons, text=str(k), variable=self.inline_vars[k],
                          font=('Segoe UI', 8), bg=self.colors['card'], fg=self.colors['text'],
                          selectcolor=self.colors['card'], activebackground=self.colors['card'],
                          activeforeground=self.colors['text']).pack(side='left', padx=5)
        
        # Watermark
        watermark_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        watermark_frame.pack(fill='x', pady=20)
        watermark_label = tk.Label(watermark_frame, text='Designed by Abedi', 
                                   font=('Segoe UI', 8, 'italic'), 
                                   bg=self.colors['bg'], fg=self.colors['text_light'])
        watermark_label.pack(pady=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _build_orca_tabs(self):
        """Build ORCA-specific tabs"""
        try:
            self._tab_orca_mode()
            self._tab_orca_method()
            self._tab_orca_solvent()
            self._tab_orca_tddft()
            self._tab_orca_resources()
            self._tab_orca_scheduler()
            self._tab_orca_custom()
            self._tab_orca_tict()
            self._tab_orca_ai_assistant()
            self._tab_orca_generate()
        except Exception as e:
            import traceback
            error_msg = f"Error building ORCA tabs: {str(e)}\n\n{traceback.format_exc()}"
            messagebox.showerror("Error", error_msg)
            raise

    # ========== ORCA Tabs ==========
    def _tab_orca_mode(self):
        """Build ORCA Mode & IO tab - matching Gaussian style"""
        f = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(f, text='⚙️ Mode & IO')
        
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
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def on_enter(event):
            canvas.focus_set()
        # Bind to both canvas and scrollable frame for better coverage
        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Enter>", on_enter)
        scrollable_frame.bind("<MouseWheel>", on_mousewheel)
        scrollable_frame.bind("<Enter>", on_enter)
        # Also bind to the frame itself
        f.bind("<MouseWheel>", lambda e: on_mousewheel(e))
        f.bind("<Enter>", lambda e: on_enter(e))
        
        # Mode & Steps Section
        mode_card = self._orca_create_card(scrollable_frame, 'Mode & Steps')
        mode_card.pack(fill='x', padx=15, pady=8)
        
        mode_row = tk.Frame(mode_card, bg=self.colors['card'])
        mode_row.pack(fill='x', padx=10, pady=(0,10))
        tk.Label(mode_row, text='Mode:', font=('Segoe UI', 10), 
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        
        if 'MODE' not in self.vars:
            self.vars['MODE'] = tk.StringVar(value=ORCA_DEFAULTS['MODE'])
        mode_buttons_frame = tk.Frame(mode_row, bg=self.colors['card'])
        mode_buttons_frame.pack(side='left')
        
        for label, value in [('Full (1,2,4,7,9)', 'full'), ('Single', 'single')]:
            btn = tk.Button(mode_buttons_frame, text=label, font=('Segoe UI', 9, 'bold'),
                          width=12, height=1, bg=self.colors['card'], fg=self.colors['text'],
                          relief='raised', borderwidth=1, command=lambda v=value: self._orca_set_mode(v),
                          cursor='hand2')
            btn.pack(side='left', padx=3)
            setattr(self, f'_orca_mode_btn_{value}', btn)
        
        step_row = tk.Frame(mode_card, bg=self.colors['card'])
        step_row.pack(fill='x', padx=10, pady=(0,10))
        tk.Label(step_row, text='Steps:', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        
        if 'STEP' not in self.vars:
            self.vars['STEP'] = tk.IntVar(value=ORCA_DEFAULTS['STEP'])
        step_buttons = tk.Frame(step_row, bg=self.colors['card'])
        step_buttons.pack(side='left')
        for k in (1,2,4,7,9):
            btn = tk.Button(step_buttons, text=str(k), width=3, height=1,
                          font=('Segoe UI', 9), bg=self.colors['card'], fg=self.colors['text'],
                          relief='raised', borderwidth=1, command=lambda v=k: self._orca_set_step(v),
                          cursor='hand2')
            btn.pack(side='left', padx=2)
            setattr(self, f'_orca_step_btn_{k}', btn)
        self._orca_update_mode_buttons()
        self._orca_update_step_buttons()
        
        # Input/Output
        io_card = self._orca_create_card(scrollable_frame, 'Input/Output Files')
        io_card.pack(fill='x', padx=15, pady=8)
        
        # Input Type Selection
        input_type_frame = tk.Frame(io_card, bg=self.colors['card'])
        input_type_frame.pack(fill='x', padx=10, pady=(0,8))
        tk.Label(input_type_frame, text='Input Type:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        if 'ORCA_INPUT_TYPE' not in self.vars:
            self.vars['ORCA_INPUT_TYPE'] = tk.StringVar(value=ORCA_DEFAULTS['INPUT_TYPE'])
        tk.Radiobutton(input_type_frame, text='.xyz Files', variable=self.vars['ORCA_INPUT_TYPE'], value='xyz',
                      font=('Segoe UI', 9), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text'],
                      command=self._orca_on_input_type_change).pack(side='left', padx=5)
        tk.Radiobutton(input_type_frame, text='.com Files', variable=self.vars['ORCA_INPUT_TYPE'], value='com',
                      font=('Segoe UI', 9), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text'],
                      command=self._orca_on_input_type_change).pack(side='left', padx=5)
        tk.Radiobutton(input_type_frame, text='.log Files', variable=self.vars['ORCA_INPUT_TYPE'], value='log',
                      font=('Segoe UI', 9), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text'],
                      command=self._orca_on_input_type_change).pack(side='left', padx=5)
        tk.Radiobutton(input_type_frame, text='SMILES', variable=self.vars['ORCA_INPUT_TYPE'], value='smiles',
                      font=('Segoe UI', 9), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text'],
                      command=self._orca_on_input_type_change).pack(side='left', padx=5)
        
        # File Input Row (for xyz, com, log)
        self.orca_input_row = tk.Frame(io_card, bg=self.colors['card'])
        self.orca_input_row.pack(fill='x', padx=10, pady=(0,8))
        self.orca_input_row.columnconfigure(0, weight=1)
        self.orca_input_label = tk.Label(self.orca_input_row, text='Input (.xyz files or folder):', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text'])
        self.orca_input_label.grid(row=0, column=0, sticky='w', pady=3)
        if 'INPUTS' not in self.vars:
            self.vars['INPUTS'] = tk.StringVar(value=ORCA_DEFAULTS['INPUTS'])
        self.orca_input_entry = tk.Entry(self.orca_input_row, textvariable=self.vars['INPUTS'], font=('Segoe UI', 9),
                         bg='white', fg=self.colors['text'], relief='flat', borderwidth=1,
                         highlightthickness=1, highlightbackground=self.colors['border'])
        self.orca_input_entry.grid(row=1, column=0, sticky='ew', padx=(0,5))
        tk.Button(self.orca_input_row, text='Browse', command=self._orca_browse_inputs,
                 font=('Segoe UI', 8), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=10, pady=3, cursor='hand2').grid(row=1, column=1)
        
        # SMILES Input Row
        self.orca_smiles_row = tk.Frame(io_card, bg=self.colors['card'])
        self.orca_smiles_row.pack_forget()  # Initially hidden
        self.orca_smiles_row.columnconfigure(0, weight=1)
        smiles_label_frame = tk.Frame(self.orca_smiles_row, bg=self.colors['card'])
        smiles_label_frame.pack(fill='x', pady=(0,3))
        tk.Label(smiles_label_frame, text='SMILES Input:', font=('Segoe UI', 9, 'bold'),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left')
        help_btn = tk.Button(smiles_label_frame, text='ChemDraw Help', font=('Segoe UI', 8),
                           bg=self.colors['primary'], fg='white',
                           relief='flat', padx=8, pady=2, cursor='hand2',
                           command=self._orca_show_chemdraw_help)
        help_btn.pack(side='right', padx=(5,0))
        load_names_btn = tk.Button(smiles_label_frame, text='Load Names from SVG', font=('Segoe UI', 8),
                           bg=self.colors['accent'], fg='white',
                           relief='flat', padx=8, pady=2, cursor='hand2',
                           command=self._orca_load_names_from_svg)
        load_names_btn.pack(side='right', padx=(5,0))
        help_text = "1. Paste SMILES → 2. Export ChemDraw as SVG → 3. Click 'Load Names from SVG'"
        tk.Label(smiles_label_frame, text=help_text, font=('Segoe UI', 7, 'italic'),
                bg=self.colors['card'], fg=self.colors['text_light']).pack(side='right', padx=(10,5))
        # Always create a new ScrolledText widget
        if hasattr(self, 'orca_smiles_text'):
            try:
                self.orca_smiles_text.destroy()
            except:
                pass
        self.orca_smiles_text = scrolledtext.ScrolledText(self.orca_smiles_row, height=6, wrap='none', font=('Consolas', 9),
                     bg='white', fg='black', relief='flat', borderwidth=1,
                     highlightthickness=1, highlightbackground=self.colors['border'])
        self.orca_smiles_text.pack(fill='x', pady=3)
        if ORCA_DEFAULTS.get('SMILES_INPUT'):
            self.orca_smiles_text.insert('1.0', ORCA_DEFAULTS['SMILES_INPUT'])
        
        # Prefix/Suffix for SMILES
        prefix_suffix_frame = tk.Frame(self.orca_smiles_row, bg=self.colors['card'])
        prefix_suffix_frame.pack(fill='x', pady=3)
        tk.Label(prefix_suffix_frame, text='Add Prefix:', font=('Segoe UI', 8),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,5))
        if 'REMOVE_PREFIX' not in self.vars:
            self.vars['REMOVE_PREFIX'] = tk.StringVar(value=ORCA_DEFAULTS.get('REMOVE_PREFIX', ''))
        tk.Entry(prefix_suffix_frame, textvariable=self.vars['REMOVE_PREFIX'], width=15,
                font=('Segoe UI', 8), bg='white', fg=self.colors['text'],
                relief='flat', borderwidth=1, highlightthickness=1,
                highlightbackground=self.colors['border']).pack(side='left', padx=(0,15))
        tk.Label(prefix_suffix_frame, text='Add Suffix:', font=('Segoe UI', 8),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,5))
        if 'REMOVE_SUFFIX' not in self.vars:
            self.vars['REMOVE_SUFFIX'] = tk.StringVar(value=ORCA_DEFAULTS.get('REMOVE_SUFFIX', ''))
        tk.Entry(prefix_suffix_frame, textvariable=self.vars['REMOVE_SUFFIX'], width=15,
                font=('Segoe UI', 8), bg='white', fg=self.colors['text'],
                relief='flat', borderwidth=1, highlightthickness=1,
                highlightbackground=self.colors['border']).pack(side='left')
        
        # Charge/Multiplicity for SMILES
        cm_row = tk.Frame(self.orca_smiles_row, bg=self.colors['card'])
        cm_row.pack(fill='x', pady=3)
        tk.Label(cm_row, text='Charge:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        if 'CHARGE' not in self.vars:
            self.vars['CHARGE'] = tk.StringVar(value='' if ORCA_DEFAULTS.get('CHARGE') is None else str(ORCA_DEFAULTS['CHARGE']))
        tk.Entry(cm_row, textvariable=self.vars['CHARGE'], width=8,
                font=('Segoe UI', 9), bg='white', fg=self.colors['text'],
                relief='flat', borderwidth=1, highlightthickness=1,
                highlightbackground=self.colors['border']).pack(side='left')
        tk.Label(cm_row, text='Multiplicity:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(20,10))
        if 'MULT' not in self.vars:
            self.vars['MULT'] = tk.StringVar(value='' if ORCA_DEFAULTS.get('MULT') is None else str(ORCA_DEFAULTS['MULT']))
        tk.Entry(cm_row, textvariable=self.vars['MULT'], width=8,
                font=('Segoe UI', 9), bg='white', fg=self.colors['text'],
                relief='flat', borderwidth=1, highlightthickness=1,
                highlightbackground=self.colors['border']).pack(side='left')
        
        # Initially show/hide based on input type
        try:
            self._orca_on_input_type_change()
        except Exception as e:
            # If there's an error, just show the default (xyz input)
            pass
        
        output_row = tk.Frame(io_card, bg=self.colors['card'])
        output_row.pack(fill='x', padx=10, pady=(0,10))
        output_row.columnconfigure(0, weight=1)
        tk.Label(output_row, text='Output:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=0, column=0, sticky='w', pady=3)
        if 'OUT_DIR' not in self.vars:
            self.vars['OUT_DIR'] = tk.StringVar(value=ORCA_DEFAULTS['OUT_DIR'])
        entry2 = tk.Entry(output_row, textvariable=self.vars['OUT_DIR'], font=('Segoe UI', 9),
                         bg='white', fg=self.colors['text'], relief='flat', borderwidth=1,
                         highlightthickness=1, highlightbackground=self.colors['border'])
        entry2.grid(row=1, column=0, sticky='ew', padx=(0,5))
        tk.Button(output_row, text='Browse', command=self._browse_outdir,
                 font=('Segoe UI', 8), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=10, pady=3, cursor='hand2').grid(row=1, column=1)
        
        # Help Section - Workflow Steps Guide - Enhanced with colors
        help_card = self._orca_create_card(scrollable_frame, '📚 Workflow Steps Guide')
        help_card.pack(fill='x', padx=15, pady=8)
        help_content = tk.Frame(help_card, bg=self.colors['card'])
        help_content.pack(fill='x', padx=10, pady=(0,10))
        
        # Colorful step descriptions for ORCA
        orca_steps_data = [
            (1, "Ground State Optimization", "Geometry optimization in the ground state", "#2E7D32"),
            (2, "Vertical Excitation", "Vertical excitation at the same geometry (Franck-Condon state)", "#1976D2"),
            (4, "Excited State Optimization", "Geometry optimization in the excited state", "#F57C00"),
            (7, "Ground State at ES Geometry", "Ground state energy calculation at the excited state geometry", "#2E7D32"),
            (9, "Custom Step", "Manual/custom calculation step (user-defined)", "#616161"),
        ]
        
        for step_num, step_name, step_desc, color in orca_steps_data:
            step_frame = tk.Frame(help_content, bg=self.colors['card'])
            step_frame.pack(fill='x', pady=2)
            
            step_num_label = tk.Label(step_frame, text=f"Step {step_num}", 
                                     font=('Segoe UI', 9, 'bold'),
                                     bg=color, fg='white',
                                     width=8, anchor='center', padx=5, pady=2)
            step_num_label.pack(side='left', padx=(0,8))
            
            step_text = f"{step_name}: {step_desc}"
            step_desc_label = tk.Label(step_frame, text=step_text,
                                      font=('Segoe UI', 8),
                                      bg=self.colors['card'], fg=self.colors['text'],
                                      justify='left', wraplength=800, anchor='w')
            step_desc_label.pack(side='left', fill='x', expand=True)
        
        # Mode info
        mode_info_frame = tk.Frame(help_content, bg=self.colors['card'])
        mode_info_frame.pack(fill='x', pady=(5,0))
        mode_info_label = tk.Label(mode_info_frame, 
                                   text="Full Mode: Generates steps 1, 2, 4, 7, 9 | Single Mode: Generate one selected step",
                                   font=('Segoe UI', 8, 'italic'),
                                   bg=self.colors['card'], fg=self.colors['text_light'],
                                   justify='left', wraplength=800)
        mode_info_label.pack(anchor='w', padx=(0,5), pady=2)
        
        # Watermark
        # Help Section - Input Types Help
        help_card = self._orca_create_card(scrollable_frame, '📚 Input Types Help')
        help_card.pack(fill='x', padx=15, pady=8)
        help_content = tk.Frame(help_card, bg=self.colors['card'])
        help_content.pack(fill='x', padx=10, pady=(0,10))
        
        help_text = """Input Files: Select a folder containing .com or .xyz files, or use a glob pattern
Output Directory: Where all generated .inp and .sh files will be saved

Supported input formats:
• .com files (Gaussian input format)
• .xyz files (XYZ coordinate format)

The application will automatically extract geometry and charge/multiplicity from input files."""
        
        help_label = tk.Label(help_content, text=help_text, 
                             font=('Segoe UI', 9), bg=self.colors['card'], 
                             fg=self.colors['text'], justify='left', wraplength=900)
        help_label.pack(anchor='w', padx=5, pady=5)
        
        watermark_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        watermark_frame.pack(fill='x', pady=20)
        watermark_label = tk.Label(watermark_frame, text='Designed by Abedi', 
                                   font=('Segoe UI', 8, 'italic'), 
                                   bg=self.colors['bg'], fg=self.colors['text_light'])
        watermark_label.pack(pady=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _orca_set_mode(self, mode):
        """Set ORCA mode"""
        self.vars['MODE'].set(mode)
        self._orca_update_mode_buttons()
    
    def _orca_set_step(self, step):
        """Set ORCA step"""
        self.vars['STEP'].set(step)
        self._orca_update_step_buttons()
    
    def _orca_update_mode_buttons(self):
        """Update ORCA mode button appearances"""
        mode = self.vars['MODE'].get()
        for opt_value in ['full', 'single']:
            btn = getattr(self, f'_orca_mode_btn_{opt_value}', None)
            if btn:
                if opt_value == mode:
                    btn.config(bg=self.colors['primary'], fg='white', relief='sunken')
                else:
                    btn.config(bg=self.colors['card'], fg=self.colors['text'], relief='raised')
    
    def _orca_update_step_buttons(self):
        """Update ORCA step button appearances"""
        selected = self.vars['STEP'].get()
        for k in (1,2,4,7,9):
            btn = getattr(self, f'_orca_step_btn_{k}', None)
            if btn:
                if k == selected:
                    btn.config(bg=self.colors['primary'], fg='white', relief='sunken')
                else:
                    btn.config(bg=self.colors['card'], fg=self.colors['text'], relief='raised')

    def _tab_orca_method(self):
        """Build ORCA Method & Basis tab - matching Gaussian style"""
        f = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(f, text='🧪 Method & Basis')
        
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
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def on_enter(event):
            canvas.focus_set()
        # Bind to both canvas and scrollable frame for better coverage
        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Enter>", on_enter)
        scrollable_frame.bind("<MouseWheel>", on_mousewheel)
        scrollable_frame.bind("<Enter>", on_enter)
        # Also bind to the frame itself
        f.bind("<MouseWheel>", lambda e: on_mousewheel(e))
        f.bind("<Enter>", lambda e: on_enter(e))
        
        method_card = self._orca_create_card(scrollable_frame, 'Method & Basis Set')
        method_card.pack(fill='x', padx=15, pady=8)
        method_content = tk.Frame(method_card, bg=self.colors['card'])
        method_content.pack(fill='x', padx=10, pady=(0,10))
        method_content.columnconfigure(1, weight=1)
        
        tk.Label(method_content, text='Functional / Method:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=0, column=0, sticky='w', pady=3)
        if hasattr(self, 'cb_method') and self.cb_method is not None:
            try:
                self.cb_method.destroy()
            except:
                pass
        self.cb_method = EditableCombo(method_content, values=["m06-2x","b3lyp","wb97x-d","cam-b3lyp","pbe0","tpss","bp86","scan","r2scan"])
        self.cb_method.set(ORCA_DEFAULTS['METHOD'])
        self.cb_method.grid(row=0, column=1, sticky='ew', padx=(10,0), pady=3)
        
        tk.Label(method_content, text='Basis Set:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=1, column=0, sticky='w', pady=3)
        if hasattr(self, 'cb_basis') and self.cb_basis is not None:
            try:
                self.cb_basis.destroy()
            except:
                pass
        self.cb_basis = EditableCombo(method_content, values=["def2-TZVP","def2-SVP","def2-TZVPP","def2-QZVP","cc-pVDZ","cc-pVTZ","6-31G*","6-311+G**","aug-cc-pVDZ","aug-cc-pVTZ"])
        self.cb_basis.set(ORCA_DEFAULTS['BASIS'])
        self.cb_basis.grid(row=1, column=1, sticky='ew', padx=(10,0), pady=3)
        
        # Help Section - Method & Basis Help
        help_card = self._orca_create_card(scrollable_frame, '📚 Method & Basis Help')
        help_card.pack(fill='x', padx=15, pady=8)
        help_content = tk.Frame(help_card, bg=self.colors['card'])
        help_content.pack(fill='x', padx=10, pady=(0,10))
        
        help_text = """Functional/Method: Select the density functional or method for your calculation
• Common choices: m06-2x, b3lyp, wb97x-d, cam-b3lyp, pbe0, tpss, scan, r2scan
• You can type any custom functional name

Basis Set: Select the basis set for your calculation
• Common choices: def2-TZVP, def2-SVP, cc-pVDZ, cc-pVTZ, aug-cc-pVDZ
• You can type any custom basis set name

Both fields support autocomplete - start typing to filter options, or type a new value to add it."""
        
        help_label = tk.Label(help_content, text=help_text, 
                             font=('Segoe UI', 9), bg=self.colors['card'], 
                             fg=self.colors['text'], justify='left', wraplength=900)
        help_label.pack(anchor='w', padx=5, pady=5)
        
        watermark_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        watermark_frame.pack(fill='x', pady=20)
        watermark_label = tk.Label(watermark_frame, text='Designed by Abedi', 
                                   font=('Segoe UI', 8, 'italic'), 
                                   bg=self.colors['bg'], fg=self.colors['text_light'])
        watermark_label.pack(pady=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _tab_orca_solvent(self):
        """Build ORCA Solvent tab - matching Gaussian style"""
        f = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(f, text='🌊 Solvent')
        
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
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def on_enter(event):
            canvas.focus_set()
        # Bind to both canvas and scrollable frame for better coverage
        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Enter>", on_enter)
        scrollable_frame.bind("<MouseWheel>", on_mousewheel)
        scrollable_frame.bind("<Enter>", on_enter)
        # Also bind to the frame itself
        f.bind("<MouseWheel>", lambda e: on_mousewheel(e))
        f.bind("<Enter>", lambda e: on_enter(e))
        
        solvent_card = self._orca_create_card(scrollable_frame, 'Solvent Model')
        solvent_card.pack(fill='x', padx=15, pady=8)
        solvent_content = tk.Frame(solvent_card, bg=self.colors['card'])
        solvent_content.pack(fill='x', padx=10, pady=(0,10))
        solvent_content.columnconfigure(1, weight=1)
        
        tk.Label(solvent_content, text='Model:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=0, column=0, sticky='w', pady=3)
        if hasattr(self, 'cb_smodel') and self.cb_smodel is not None:
            try:
                self.cb_smodel.destroy()
            except:
                pass
        self.cb_smodel = EditableCombo(solvent_content, values=["none","SMD","CPCM"])
        self.cb_smodel.set(ORCA_DEFAULTS['SOLVENT_MODEL'])
        self.cb_smodel.grid(row=0, column=1, sticky='ew', padx=(10,0), pady=3)
        
        tk.Label(solvent_content, text='Solvent Name:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=1, column=0, sticky='w', pady=3)
        if hasattr(self, 'cb_sname') and self.cb_sname is not None:
            try:
                self.cb_sname.destroy()
            except:
                pass
        self.cb_sname = EditableCombo(solvent_content, values=["DMSO","Water","Acetonitrile","Methanol","Ethanol","DCM","THF","Toluene","Benzene","Acetone","DMF"])
        self.cb_sname.set(ORCA_DEFAULTS['SOLVENT_NAME'])
        self.cb_sname.grid(row=1, column=1, sticky='ew', padx=(10,0), pady=3)
        
        tk.Label(solvent_content, text='ε (CPCM epsilon, optional):', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=2, column=0, sticky='w', pady=3)
        if 'EPSILON' not in self.vars:
            self.vars['EPSILON'] = tk.StringVar(value=ORCA_DEFAULTS['EPSILON'])
        tk.Entry(solvent_content, textvariable=self.vars['EPSILON'], font=('Segoe UI', 9),
                bg='white', fg=self.colors['text'], relief='flat', borderwidth=1,
                highlightthickness=1, highlightbackground=self.colors['border']).grid(row=2, column=1, sticky='ew', padx=(10,0), pady=3)
        
        # Help Section - Solvent Help
        help_card = self._orca_create_card(scrollable_frame, '📚 Solvent Model Help')
        help_card.pack(fill='x', padx=15, pady=8)
        help_content = tk.Frame(help_card, bg=self.colors['card'])
        help_content.pack(fill='x', padx=10, pady=(0,10))
        
        help_text = """Solvent Model: Choose how to model solvation effects
• none: Gas-phase calculation (no solvent)
• SMD: Solvation Model based on Density (recommended for most cases)
• CPCM: Conductor-like Polarizable Continuum Model

Solvent Name: Select the solvent for SMD/CPCM calculations
• Common solvents: DMSO, Water, Acetonitrile, Methanol, Ethanol, DCM, THF, Toluene, Benzene, Acetone, DMF

ε (Epsilon): Optional - For CPCM, you can specify a custom dielectric constant
• Leave blank to use the named solvent's default value
• Useful for custom solvent environments"""
        
        help_label = tk.Label(help_content, text=help_text, 
                             font=('Segoe UI', 9), bg=self.colors['card'], 
                             fg=self.colors['text'], justify='left', wraplength=900)
        help_label.pack(anchor='w', padx=5, pady=5)
        
        watermark_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        watermark_frame.pack(fill='x', pady=20)
        watermark_label = tk.Label(watermark_frame, text='Designed by Abedi', 
                                   font=('Segoe UI', 8, 'italic'), 
                                   bg=self.colors['bg'], fg=self.colors['text_light'])
        watermark_label.pack(pady=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _tab_orca_tddft(self):
        """Build ORCA TD-DFT tab - matching Gaussian style"""
        f = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(f, text='✨ TD-DFT')
        
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
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def on_enter(event):
            canvas.focus_set()
        # Bind to both canvas and scrollable frame for better coverage
        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Enter>", on_enter)
        scrollable_frame.bind("<MouseWheel>", on_mousewheel)
        scrollable_frame.bind("<Enter>", on_enter)
        # Also bind to the frame itself
        f.bind("<MouseWheel>", lambda e: on_mousewheel(e))
        f.bind("<Enter>", lambda e: on_enter(e))
        
        td_card = self._orca_create_card(scrollable_frame, 'TD-DFT Settings')
        td_card.pack(fill='x', padx=15, pady=8)
        td_content = tk.Frame(td_card, bg=self.colors['card'])
        td_content.pack(fill='x', padx=10, pady=(0,10))
        
        td_row1 = tk.Frame(td_content, bg=self.colors['card'])
        td_row1.pack(fill='x', pady=3)
        tk.Label(td_row1, text='NRoots:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        if 'TD_NROOTS' not in self.vars:
            self.vars['TD_NROOTS'] = tk.IntVar(value=ORCA_DEFAULTS['TD_NROOTS'])
        spin1 = ttk.Spinbox(td_row1, from_=1, to=128, textvariable=self.vars['TD_NROOTS'], width=8)
        spin1.pack(side='left')
        
        tk.Label(td_row1, text='IRoot (for ES opt):', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(20,10))
        if 'TD_IROOT' not in self.vars:
            self.vars['TD_IROOT'] = tk.IntVar(value=ORCA_DEFAULTS['TD_IROOT'])
        spin2 = ttk.Spinbox(td_row1, from_=1, to=128, textvariable=self.vars['TD_IROOT'], width=8)
        spin2.pack(side='left')
        
        if 'TD_TDA' not in self.vars:
            self.vars['TD_TDA'] = tk.BooleanVar(value=ORCA_DEFAULTS['TD_TDA'])
        tk.Checkbutton(td_content, text='Use TDA (default in ORCA)', variable=self.vars['TD_TDA'],
                      font=('Segoe UI', 8), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text']).pack(anchor='w', pady=3)
        
        if 'FOLLOW_IROOT' not in self.vars:
            self.vars['FOLLOW_IROOT'] = tk.BooleanVar(value=ORCA_DEFAULTS['FOLLOW_IROOT'])
        tk.Checkbutton(td_content, text='Follow iroot during ES opt', variable=self.vars['FOLLOW_IROOT'],
                      font=('Segoe UI', 8), bg=self.colors['card'], fg=self.colors['text'],
                      selectcolor=self.colors['card'], activebackground=self.colors['card'],
                      activeforeground=self.colors['text']).pack(anchor='w', pady=3)
        
        # Help Section - TD-DFT Help
        help_card = self._orca_create_card(scrollable_frame, '📚 TD-DFT Help')
        help_card.pack(fill='x', padx=15, pady=8)
        help_content = tk.Frame(help_card, bg=self.colors['card'])
        help_content.pack(fill='x', padx=10, pady=(0,10))
        
        help_text = """TD-DFT (Time-Dependent Density Functional Theory) Settings:

NRoots: Number of excited states to calculate
• Typically 3-10 states for absorption spectra
• More states = longer calculation time

IRoot: Which excited state to optimize (for Step 4 - ES Optimization)
• Usually 1 (first excited state)
• Set to the state you want to optimize

Use TDA: Tamm-Dancoff Approximation (default in ORCA)
• Faster calculation, good for most cases
• Uncheck for full TD-DFT (slower but more accurate)

Follow IRoot: During excited state optimization, follow the specified root
• Prevents root flipping during optimization
• Recommended for excited state geometry optimizations"""
        
        help_label = tk.Label(help_content, text=help_text, 
                             font=('Segoe UI', 9), bg=self.colors['card'], 
                             fg=self.colors['text'], justify='left', wraplength=900)
        help_label.pack(anchor='w', padx=5, pady=5)
        
        watermark_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        watermark_frame.pack(fill='x', pady=20)
        watermark_label = tk.Label(watermark_frame, text='Designed by Abedi', 
                                   font=('Segoe UI', 8, 'italic'), 
                                   bg=self.colors['bg'], fg=self.colors['text_light'])
        watermark_label.pack(pady=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _tab_orca_resources(self):
        """Build ORCA Resources tab - matching Gaussian style"""
        f = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(f, text='💻 Resources')
        
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
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def on_enter(event):
            canvas.focus_set()
        # Bind to both canvas and scrollable frame for better coverage
        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Enter>", on_enter)
        scrollable_frame.bind("<MouseWheel>", on_mousewheel)
        scrollable_frame.bind("<Enter>", on_enter)
        # Also bind to the frame itself
        f.bind("<MouseWheel>", lambda e: on_mousewheel(e))
        f.bind("<Enter>", lambda e: on_enter(e))
        
        res_card = self._orca_create_card(scrollable_frame, 'Computational Resources')
        res_card.pack(fill='x', padx=15, pady=8)
        res_content = tk.Frame(res_card, bg=self.colors['card'])
        res_content.pack(fill='x', padx=10, pady=(0,10))
        res_content.columnconfigure(1, weight=1)
        
        tk.Label(res_content, text='Cores:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=0, column=0, sticky='w', pady=3)
        if 'NPROCS' not in self.vars:
            self.vars['NPROCS'] = tk.IntVar(value=ORCA_DEFAULTS['NPROCS'])
        ttk.Spinbox(res_content, from_=1, to=256, textvariable=self.vars['NPROCS'], width=10).grid(row=0, column=1, sticky='w', padx=(10,0), pady=3)
        
        tk.Label(res_content, text='Maxcore per core (MB):', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=1, column=0, sticky='w', pady=3)
        if 'MAXCORE_MB' not in self.vars:
            self.vars['MAXCORE_MB'] = tk.IntVar(value=ORCA_DEFAULTS['MAXCORE_MB'])
        ttk.Spinbox(res_content, from_=500, to=128000, increment=500, textvariable=self.vars['MAXCORE_MB'], width=10).grid(row=1, column=1, sticky='w', padx=(10,0), pady=3)
        
        tk.Label(res_content, text='ORCA Executable:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=2, column=0, sticky='w', pady=3)
        if 'ORCA_PATH' not in self.vars:
            self.vars['ORCA_PATH'] = tk.StringVar(value=ORCA_DEFAULTS['ORCA_PATH'])
        path_row = tk.Frame(res_content, bg=self.colors['card'])
        path_row.grid(row=2, column=1, sticky='ew', padx=(10,0), pady=3)
        path_row.columnconfigure(0, weight=1)
        tk.Entry(path_row, textvariable=self.vars['ORCA_PATH'], font=('Segoe UI', 9),
                bg='white', fg=self.colors['text'], relief='flat', borderwidth=1,
                highlightthickness=1, highlightbackground=self.colors['border']).grid(row=0, column=0, sticky='ew', padx=(0,5))
        tk.Button(path_row, text='Browse', command=self._browse_orca,
                 font=('Segoe UI', 8), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=10, pady=3, cursor='hand2').grid(row=0, column=1)
        
        # Help Section - Resources Help
        help_card = self._orca_create_card(scrollable_frame, '📚 Computational Resources Help')
        help_card.pack(fill='x', padx=15, pady=8)
        help_content = tk.Frame(help_card, bg=self.colors['card'])
        help_content.pack(fill='x', padx=10, pady=(0,10))
        
        help_text = """Computational Resources Settings:

Cores: Number of CPU cores to use for parallel calculation
• Match this to your cluster node's core count
• Typical values: 16, 32, 64, 128

Maxcore per core (MB): Memory per core in megabytes
• Total memory = Cores × Maxcore
• Example: 64 cores × 2000 MB = 128 GB total
• Typical values: 1000-4000 MB per core

ORCA Executable: Path to the ORCA program
• Default: "orca" (assumes ORCA is in PATH)
• Use full path if ORCA is not in PATH: /path/to/orca/orca
• On clusters, this is usually just "orca" """
        
        help_label = tk.Label(help_content, text=help_text, 
                             font=('Segoe UI', 9), bg=self.colors['card'], 
                             fg=self.colors['text'], justify='left', wraplength=900)
        help_label.pack(anchor='w', padx=5, pady=5)
        
        watermark_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        watermark_frame.pack(fill='x', pady=20)
        watermark_label = tk.Label(watermark_frame, text='Designed by Abedi', 
                                   font=('Segoe UI', 8, 'italic'), 
                                   bg=self.colors['bg'], fg=self.colors['text_light'])
        watermark_label.pack(pady=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _tab_orca_scheduler(self):
        """Build ORCA Scheduler tab - matching Gaussian style"""
        f = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(f, text='📊 Scheduler')
        
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
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def on_enter(event):
            canvas.focus_set()
        # Bind to both canvas and scrollable frame for better coverage
        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Enter>", on_enter)
        scrollable_frame.bind("<MouseWheel>", on_mousewheel)
        scrollable_frame.bind("<Enter>", on_enter)
        # Also bind to the frame itself
        f.bind("<MouseWheel>", lambda e: on_mousewheel(e))
        f.bind("<Enter>", lambda e: on_enter(e))
        
        sched_card = self._orca_create_card(scrollable_frame, 'Scheduler Settings')
        sched_card.pack(fill='x', padx=15, pady=8)
        sched_content = tk.Frame(sched_card, bg=self.colors['card'])
        sched_content.pack(fill='x', padx=10, pady=(0,10))
        sched_content.columnconfigure(1, weight=1)
        
        tk.Label(sched_content, text='Type:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=0, column=0, sticky='w', pady=3)
        if hasattr(self, 'cb_sched') and self.cb_sched is not None:
            try:
                self.cb_sched.destroy()
            except:
                pass
        self.cb_sched = EditableCombo(sched_content, values=["pbs","slurm","local"])
        self.cb_sched.set(ORCA_DEFAULTS['SCHEDULER'])
        self.cb_sched.grid(row=0, column=1, sticky='ew', padx=(10,0), pady=3)
        
        tk.Label(sched_content, text='Queue:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=1, column=0, sticky='w', pady=3)
        if hasattr(self, 'cb_queue') and self.cb_queue is not None:
            try:
                self.cb_queue.destroy()
            except:
                pass
        self.cb_queue = EditableCombo(sched_content, values=["normal","express","long","gpu","debug"])
        self.cb_queue.set(ORCA_DEFAULTS['QUEUE'])
        self.cb_queue.grid(row=1, column=1, sticky='ew', padx=(10,0), pady=3)
        
        tk.Label(sched_content, text='Walltime:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=2, column=0, sticky='w', pady=3)
        if 'WALLTIME' not in self.vars:
            self.vars['WALLTIME'] = tk.StringVar(value=ORCA_DEFAULTS['WALLTIME'])
        tk.Entry(sched_content, textvariable=self.vars['WALLTIME'], font=('Segoe UI', 9),
                bg='white', fg=self.colors['text'], relief='flat', borderwidth=1,
                highlightthickness=1, highlightbackground=self.colors['border']).grid(row=2, column=1, sticky='ew', padx=(10,0), pady=3)
        
        tk.Label(sched_content, text='Project:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=3, column=0, sticky='w', pady=3)
        if 'PROJECT' not in self.vars:
            self.vars['PROJECT'] = tk.StringVar(value=ORCA_DEFAULTS['PROJECT'])
        tk.Entry(sched_content, textvariable=self.vars['PROJECT'], font=('Segoe UI', 9),
                bg='white', fg=self.colors['text'], relief='flat', borderwidth=1,
                highlightthickness=1, highlightbackground=self.colors['border']).grid(row=3, column=1, sticky='ew', padx=(10,0), pady=3)
        
        tk.Label(sched_content, text='Account:', font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=4, column=0, sticky='w', pady=3)
        if 'ACCOUNT' not in self.vars:
            self.vars['ACCOUNT'] = tk.StringVar(value=ORCA_DEFAULTS['ACCOUNT'])
        tk.Entry(sched_content, textvariable=self.vars['ACCOUNT'], font=('Segoe UI', 9),
                bg='white', fg=self.colors['text'], relief='flat', borderwidth=1,
                highlightthickness=1, highlightbackground=self.colors['border']).grid(row=4, column=1, sticky='ew', padx=(10,0), pady=3)
        
        # Help Section - Scheduler Help
        help_card = self._orca_create_card(scrollable_frame, '📚 Scheduler Help')
        help_card.pack(fill='x', padx=15, pady=8)
        help_content = tk.Frame(help_card, bg=self.colors['card'])
        help_content.pack(fill='x', padx=10, pady=(0,10))
        
        help_text = """Scheduler Settings for Job Submission:

Scheduler Type: Choose your cluster's job scheduler
• PBS: Portable Batch System (qsub)
• SLURM: Simple Linux Utility for Resource Management (sbatch)
• local: Run jobs locally (bash)

Queue/Partition: Select the queue or partition name
• Common: normal, express, long, gpu, debug
• Check your cluster's available queues

Walltime: Maximum time for the job (HH:MM:SS format)
• Example: "24:00:00" = 24 hours
• Typical: 12-48 hours for geometry optimizations

Project (PBS): Project ID for PBS systems (-P flag)
Account (SLURM): Account name for SLURM systems (-A flag)"""
        
        help_label = tk.Label(help_content, text=help_text, 
                             font=('Segoe UI', 9), bg=self.colors['card'], 
                             fg=self.colors['text'], justify='left', wraplength=900)
        help_label.pack(anchor='w', padx=5, pady=5)
        
        watermark_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        watermark_frame.pack(fill='x', pady=20)
        watermark_label = tk.Label(watermark_frame, text='Designed by Abedi', 
                                   font=('Segoe UI', 8, 'italic'), 
                                   bg=self.colors['bg'], fg=self.colors['text_light'])
        watermark_label.pack(pady=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _tab_orca_custom(self):
        """Build ORCA Custom Step tab - matching Gaussian style"""
        f = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(f, text='🛠️ Step 9')
        
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
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def on_enter(event):
            canvas.focus_set()
        # Bind to both canvas and scrollable frame for better coverage
        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Enter>", on_enter)
        scrollable_frame.bind("<MouseWheel>", on_mousewheel)
        scrollable_frame.bind("<Enter>", on_enter)
        # Also bind to the frame itself
        f.bind("<MouseWheel>", lambda e: on_mousewheel(e))
        f.bind("<Enter>", lambda e: on_enter(e))
        
        custom_card = self._orca_create_card(scrollable_frame, 'Custom Step (Step 9)')
        custom_card.pack(fill='both', expand=True, padx=15, pady=8)
        custom_content = tk.Frame(custom_card, bg=self.colors['card'])
        custom_content.pack(fill='both', expand=True, padx=10, pady=(0,10))
        
        tk.Label(custom_content, text="Custom header line (with or without '!'):", font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(anchor='w', pady=(0,5))
        if hasattr(self, 'txt_ckw') and self.txt_ckw is not None:
            try:
                self.txt_ckw.destroy()
            except:
                pass
        self.txt_ckw = scrolledtext.ScrolledText(custom_content, height=3, wrap='word', font=('Consolas', 10),
                                                bg='white', fg='black')
        self.txt_ckw.pack(fill='x', padx=4, pady=(0,10))
        self.txt_ckw.insert('1.0', ORCA_DEFAULTS['CUSTOM_KEYWORDS'])
        
        tk.Label(custom_content, text="Extra % blocks (optional):", font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['text']).pack(anchor='w', pady=(0,5))
        if hasattr(self, 'txt_cblk') and self.txt_cblk is not None:
            try:
                self.txt_cblk.destroy()
            except:
                pass
        self.txt_cblk = scrolledtext.ScrolledText(custom_content, height=6, wrap='word', font=('Consolas', 10),
                                                  bg='white', fg='black')
        self.txt_cblk.pack(fill='both', expand=True, padx=4, pady=4)
        if ORCA_DEFAULTS['CUSTOM_BLOCK']:
            self.txt_cblk.insert('1.0', ORCA_DEFAULTS['CUSTOM_BLOCK'])
        
        # Help Section - Custom Step Help
        help_card = self._orca_create_card(scrollable_frame, '📚 Custom Step (Step 9) Help')
        help_card.pack(fill='x', padx=15, pady=8)
        help_content = tk.Frame(help_card, bg=self.colors['card'])
        help_content.pack(fill='x', padx=10, pady=(0,10))
        
        help_text = """Custom Step (Step 9): Create a manual/custom ORCA calculation

Custom Header Line: ORCA keywords for your custom calculation
• Include the '!' at the start if needed: ! B3LYP def2-TZVP Opt
• Or omit it: B3LYP def2-TZVP Opt
• Examples: "! CCSD(T) def2-TZVP", "! MP2 def2-SVP Freq"

Extra % Blocks: Additional ORCA input blocks (optional)
• Use for special settings like:
  %pal nprocs 64 end
  %maxcore 2000 end
  %geom constraints end
• Leave blank if not needed

This step allows you to add any custom ORCA calculation to your workflow."""
        
        help_label = tk.Label(help_content, text=help_text, 
                             font=('Segoe UI', 9), bg=self.colors['card'], 
                             fg=self.colors['text'], justify='left', wraplength=900)
        help_label.pack(anchor='w', padx=5, pady=5)
        
        watermark_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        watermark_frame.pack(fill='x', pady=20)
        watermark_label = tk.Label(watermark_frame, text='Designed by Abedi', 
                                   font=('Segoe UI', 8, 'italic'), 
                                   bg=self.colors['bg'], fg=self.colors['text_light'])
        watermark_label.pack(pady=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _tab_orca_tict(self):
        """ORCA TICT Rotation Tab"""
        f = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(f, text='🔄 TICT Rotation')
        
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
        
        # Initialize TICT variables for ORCA
        if not hasattr(self, 'tict_vars_orca'):
            self.tict_vars_orca = {
                'INPUT_FILE': tk.StringVar(value=""),
                'OUTPUT_DIR': tk.StringVar(value=""),
                'AXIS': tk.StringVar(value="3,10"),
                'BRANCH_A': tk.StringVar(value="11,18,22"),
                'BRANCH_A_STEP': tk.StringVar(value="-8.81"),
                'BRANCH_B': tk.StringVar(value="12-13,19-21,23-26"),
                'BRANCH_B_STEP': tk.StringVar(value="-9.36"),
                'NUM_STEPS': tk.StringVar(value="9"),
            }
        
        # File Selection Section
        file_card = self._orca_create_card(scrollable_frame, 'Input/Output Files')
        file_card.pack(fill='x', padx=15, pady=8)
        
        file_content = tk.Frame(file_card, bg=self.colors['card'])
        file_content.pack(fill='x', padx=10, pady=10)
        
        # Input file (ORCA supports .xyz and .inp)
        input_row = tk.Frame(file_content, bg=self.colors['card'])
        input_row.pack(fill='x', pady=5)
        tk.Label(input_row, text='Input File (.xyz or .inp):', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        tk.Entry(input_row, textvariable=self.tict_vars_orca['INPUT_FILE'], width=60,
                font=('Segoe UI', 9)).pack(side='left', fill='x', expand=True, padx=(0,5))
        tk.Button(input_row, text='Browse...', command=lambda: self._tict_browse_input_orca(),
                 font=('Segoe UI', 9), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=15, pady=4, cursor='hand2').pack(side='left')
        
        # Output directory
        output_row = tk.Frame(file_content, bg=self.colors['card'])
        output_row.pack(fill='x', pady=5)
        tk.Label(output_row, text='Output Directory:', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        tk.Entry(output_row, textvariable=self.tict_vars_orca['OUTPUT_DIR'], width=60,
                font=('Segoe UI', 9)).pack(side='left', fill='x', expand=True, padx=(0,5))
        tk.Button(output_row, text='Browse...', command=lambda: self._tict_browse_output_orca(),
                 font=('Segoe UI', 9), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=15, pady=4, cursor='hand2').pack(side='left')
        
        # Rotation Parameters Section (same as Gaussian)
        params_card = self._orca_create_card(scrollable_frame, 'TICT Rotation Parameters (1-Based Atom Indices)')
        params_card.pack(fill='x', padx=15, pady=8)
        
        params_content = tk.Frame(params_card, bg=self.colors['card'])
        params_content.pack(fill='x', padx=10, pady=10)
        
        params_content.columnconfigure(1, weight=1)
        
        row = 0
        tk.Label(params_content, text='Rotation Axis (2 atoms, e.g., "3,10"):', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(params_content, textvariable=self.tict_vars_orca['AXIS'], font=('Segoe UI', 9)).grid(row=row, column=1, sticky='ew', padx=5, pady=5)
        row += 1
        
        tk.Label(params_content, text='Branch A Indices:', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(params_content, textvariable=self.tict_vars_orca['BRANCH_A'], font=('Segoe UI', 9)).grid(row=row, column=1, sticky='ew', padx=5, pady=5)
        row += 1
        
        tk.Label(params_content, text='Branch A Step (degrees):', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(params_content, textvariable=self.tict_vars_orca['BRANCH_A_STEP'], font=('Segoe UI', 9), width=20).grid(row=row, column=1, sticky='w', padx=5, pady=5)
        row += 1
        
        tk.Label(params_content, text='Branch B Indices:', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(params_content, textvariable=self.tict_vars_orca['BRANCH_B'], font=('Segoe UI', 9)).grid(row=row, column=1, sticky='ew', padx=5, pady=5)
        row += 1
        
        tk.Label(params_content, text='Branch B Step (degrees):', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(params_content, textvariable=self.tict_vars_orca['BRANCH_B_STEP'], font=('Segoe UI', 9), width=20).grid(row=row, column=1, sticky='w', padx=5, pady=5)
        row += 1
        
        tk.Label(params_content, text='Number of Steps (0 to N inclusive):', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(params_content, textvariable=self.tict_vars_orca['NUM_STEPS'], font=('Segoe UI', 9), width=20).grid(row=row, column=1, sticky='w', padx=5, pady=5)
        row += 1
        
        help_text = """Help: Enter atom indices using 1-based indexing.
You can use ranges (e.g., "12-13") and comma-separated lists (e.g., "11,18,22").
Example: "12-13,19-21,23-26" means atoms 12,13,19,20,21,23,24,25,26."""
        tk.Label(params_content, text=help_text, font=('Segoe UI', 8), bg=self.colors['card'],
                fg=self.colors['text_light'], justify='left', wraplength=600).grid(row=row, column=0, columnspan=2, sticky='w', padx=5, pady=(10,5))
        
        # Generate Button
        button_card = self._orca_create_card(scrollable_frame, None)
        button_card.pack(fill='x', padx=15, pady=8)
        button_frame = tk.Frame(button_card, bg=self.colors['card'])
        button_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(button_frame, text='Generate Rotated Geometries (ORCA .xyz)', command=lambda: self._tict_generate_orca(),
                 font=('Segoe UI', 11, 'bold'), bg=self.colors['accent'], fg='white',
                 relief='flat', padx=25, pady=10, cursor='hand2').pack(side='left', padx=5)
        
        if not TICT_AVAILABLE:
            warning_label = tk.Label(button_frame, text='⚠️ TICT module not available', 
                                    font=('Segoe UI', 9), bg=self.colors['card'], fg='red')
            warning_label.pack(side='left', padx=10)
        
        # Status and Log
        log_card = self._orca_create_card(scrollable_frame, 'Status & Log')
        log_card.pack(fill='both', expand=True, padx=15, pady=8)
        log_content = tk.Frame(log_card, bg=self.colors['card'])
        log_content.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.tict_log_orca = scrolledtext.ScrolledText(log_content, wrap='word', font=('Consolas', 10),
                                                  bg='white', fg='black', height=15)
        self.tict_log_orca.pack(fill='both', expand=True)
        self.tict_log_orca.insert('1.0', 'Ready. Select input file and set parameters, then click "Generate Rotated Geometries".\n')
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Watermark
        watermark_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        watermark_frame.pack(fill='x', pady=20)
        watermark_label = tk.Label(watermark_frame, text='Designed by Abedi', 
                                   font=('Segoe UI', 8, 'italic'), 
                                   bg=self.colors['bg'], fg=self.colors['text_light'])
        watermark_label.pack()
    
    def _tab_orca_ai_assistant(self):
        """ORCA AI Assistant Tab with Gemini Pro"""
        f = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(f, text='🤖 AI Assistant')
        
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
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Settings Section
        settings_card = self._orca_create_card(scrollable_frame, 'AI Settings')
        settings_card.pack(fill='x', padx=15, pady=8)
        
        settings_content = tk.Frame(settings_card, bg=self.colors['card'])
        settings_content.pack(fill='x', padx=10, pady=10)
        
        # Check Ollama status
        try:
            from ai_assistant import OLLAMA_AVAILABLE, GEMINI_AVAILABLE
            ollama_status = "✓ Available" if OLLAMA_AVAILABLE else "✗ Not installed"
            gemini_status = "✓ Available" if GEMINI_AVAILABLE else "✗ Not installed"
        except:
            ollama_status = "Unknown"
            gemini_status = "Unknown"
        
        status_row = tk.Frame(settings_content, bg=self.colors['card'])
        status_row.pack(fill='x', pady=5)
        tk.Label(status_row, text='Ollama (Free, Local):', font=('Segoe UI', 10, 'bold'),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        tk.Label(status_row, text=ollama_status, font=('Segoe UI', 10),
                bg=self.colors['card'], fg='green' if '✓' in ollama_status else 'red').pack(side='left', padx=(0,20))
        
        if not OLLAMA_AVAILABLE:
            import platform
            if platform.system() == 'Darwin':  # macOS
                install_text = 'Install: Download from https://ollama.com/download or: brew install ollama'
            else:  # Linux
                install_text = 'Install: curl -fsSL https://ollama.com/install.sh | sh'
            install_label = tk.Label(status_row, text=install_text, 
                                    font=('Segoe UI', 8), bg=self.colors['card'], fg=self.colors['text_light'])
            install_label.pack(side='left', padx=10)
        
        # Gemini API Key (optional)
        gemini_row = tk.Frame(settings_content, bg=self.colors['card'])
        gemini_row.pack(fill='x', pady=5)
        tk.Label(gemini_row, text='Gemini API Key (Optional):', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        
        if not hasattr(self, 'gemini_api_key_var_orca'):
            self.gemini_api_key_var_orca = tk.StringVar()
            try:
                from ai_assistant import load_gemini_api_key
                existing_key = load_gemini_api_key()
                if existing_key:
                    self.gemini_api_key_var_orca.set(existing_key)
            except:
                pass
        
        api_entry = tk.Entry(gemini_row, textvariable=self.gemini_api_key_var_orca, show='*', width=40, font=('Segoe UI', 9))
        api_entry.pack(side='left', fill='x', expand=True, padx=(0,5))
        tk.Button(gemini_row, text='Save', command=lambda: self._save_gemini_key(api_entry.get()),
                 font=('Segoe UI', 9), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=15, pady=4, cursor='hand2').pack(side='left')
        tk.Label(gemini_row, text='Get free key: https://makersuite.google.com/app/apikey', 
                font=('Segoe UI', 8), bg=self.colors['card'], fg=self.colors['text_light']).pack(side='left', padx=10)
        
        # Chat Interface
        chat_card = self._orca_create_card(scrollable_frame, 'Chat with AI Assistant')
        chat_card.pack(fill='both', expand=True, padx=15, pady=8)
        
        chat_content = tk.Frame(chat_card, bg=self.colors['card'])
        chat_content.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Chat display area
        chat_display = scrolledtext.ScrolledText(chat_content, wrap='word', font=('Segoe UI', 10),
                                                  bg='white', fg='black', height=20, state=tk.DISABLED)
        chat_display.pack(fill='both', expand=True, pady=(0,10))
        chat_display.tag_config('user', foreground='blue', font=('Segoe UI', 10, 'bold'))
        chat_display.tag_config('assistant', foreground='green', font=('Segoe UI', 10))
        chat_display.tag_config('error', foreground='red', font=('Segoe UI', 10))
        
        # Initial message
        chat_display.config(state=tk.NORMAL)
        chat_display.insert('1.0', '🤖 AI Assistant: Hello! I can help you set up ORCA quantum chemistry calculations.\n\n'
                                   'Tell me what you want to calculate, and I\'ll guide you through the process.\n\n'
                                   'Examples:\n'
                                   '- "I want to optimize a benzene molecule in DMSO"\n'
                                   '- "Calculate excited states for a TICT molecule"\n'
                                   '- "Set up a full workflow for fluorescence calculations"\n\n', 'assistant')
        chat_display.config(state=tk.DISABLED)
        
        # Input area
        input_frame = tk.Frame(chat_content, bg=self.colors['card'])
        input_frame.pack(fill='x')
        
        input_entry = tk.Entry(input_frame, font=('Segoe UI', 10))
        input_entry.pack(side='left', fill='x', expand=True, padx=(0,5))
        input_entry.bind('<Return>', lambda e: self._send_ai_message(input_entry, chat_display, 'orca'))
        
        button_frame = tk.Frame(input_frame, bg=self.colors['card'])
        button_frame.pack(side='left')
        
        tk.Button(button_frame, text='Send', command=lambda: self._send_ai_message(input_entry, chat_display, 'orca'),
                 font=('Segoe UI', 9, 'bold'), bg=self.colors['accent'], fg='white',
                 relief='flat', padx=20, pady=5, cursor='hand2').pack(side='left', padx=2)
        tk.Button(button_frame, text='Generate Files', command=lambda: self._generate_files_from_ai_conversation(chat_display, 'orca'),
                 font=('Segoe UI', 9, 'bold'), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=15, pady=5, cursor='hand2').pack(side='left', padx=2)
        tk.Button(button_frame, text='Clear', command=lambda: self._clear_ai_chat(chat_display),
                 font=('Segoe UI', 9), bg=self.colors['secondary'], fg='white',
                 relief='flat', padx=15, pady=5, cursor='hand2').pack(side='left', padx=2)
        
        # Store references
        self.orca_chat_display = chat_display
        self.orca_chat_input = input_entry
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Watermark
        watermark_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        watermark_frame.pack(fill='x', pady=20)
        watermark_label = tk.Label(watermark_frame, text='Powered by Ollama (Free) / Gemini | Designed by Abedi', 
                                   font=('Segoe UI', 8, 'italic'), 
                                   bg=self.colors['bg'], fg=self.colors['text_light'])
        watermark_label.pack()
    
    def _tab_orca_generate(self):
        """Build ORCA Generate tab - matching Gaussian style"""
        f = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(f, text='🚀 Generate')
        
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
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def on_enter(event):
            canvas.focus_set()
        # Bind to both canvas and scrollable frame for better coverage
        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Enter>", on_enter)
        scrollable_frame.bind("<MouseWheel>", on_mousewheel)
        scrollable_frame.bind("<Enter>", on_enter)
        # Also bind to the frame itself
        f.bind("<MouseWheel>", lambda e: on_mousewheel(e))
        f.bind("<Enter>", lambda e: on_enter(e))
        
        button_card = self._orca_create_card(scrollable_frame, None)
        button_card.pack(fill='x', padx=15, pady=10)
        button_frame = tk.Frame(button_card, bg=self.colors['card'])
        button_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(button_frame, text='Preview', command=self._orca_preview,
                 font=('Segoe UI', 10, 'bold'), bg=self.colors['secondary'], fg='white',
                 relief='flat', padx=20, pady=8, cursor='hand2').pack(side='left', padx=5)
        tk.Button(button_frame, text='Generate', command=self._orca_generate,
                 font=('Segoe UI', 10, 'bold'), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=20, pady=8, cursor='hand2').pack(side='left', padx=5)
        tk.Button(button_frame, text='Save Prefs', command=self._save_prefs,
                 font=('Segoe UI', 9), bg=self.colors['accent'], fg='white',
                 relief='flat', padx=15, pady=8, cursor='hand2').pack(side='left', padx=5)
        tk.Button(button_frame, text='Load Prefs', command=self._load_prefs,
                 font=('Segoe UI', 9), bg=self.colors['accent'], fg='white',
                 relief='flat', padx=15, pady=8, cursor='hand2').pack(side='left', padx=5)
        
        status_card = self._orca_create_card(scrollable_frame, None)
        status_card.pack(fill='x', padx=15, pady=(0,10))
        status_frame = tk.Frame(status_card, bg=self.colors['card'])
        status_frame.pack(fill='x', padx=10, pady=8)
        if not hasattr(self, 'orca_status'):
            self.orca_status = tk.Label(status_frame, text='Ready', font=('Segoe UI', 10, 'bold'),
                                  bg=self.colors['card'], fg=self.colors['primary'])
            self.orca_status.pack(anchor='w')
        self.status = self.orca_status  # Alias for compatibility
        
        preview_card = self._orca_create_card(scrollable_frame, 'Preview & Output')
        preview_card.pack(fill='both', expand=True, padx=15, pady=(0,10))
        preview_content = tk.Frame(preview_card, bg=self.colors['card'])
        preview_content.pack(fill='both', expand=True, padx=10, pady=10)
        if not hasattr(self, 'orca_preview'):
            self.orca_preview = scrolledtext.ScrolledText(preview_content, wrap='word', font=('Consolas', 10),
                                                    bg='white', fg='black')
            self.orca_preview.pack(fill='both', expand=True)
        self.preview = self.orca_preview  # Alias for compatibility
        
        # Help Section - Generation Help
        help_card = self._orca_create_card(scrollable_frame, '📚 Generation Help')
        help_card.pack(fill='x', padx=15, pady=(0,10))
        help_content = tk.Frame(help_card, bg=self.colors['card'])
        help_content.pack(fill='x', padx=10, pady=(0,10))
        
        help_text = """Preview: Shows a preview of the ORCA input files that will be generated | Generate: Creates all .inp and .sh files in the output directory | Save Prefs: Saves all current settings to a preferences file | Load Prefs: Loads previously saved settings | After generation, the output folder will automatically open. All generated files are ready for submission to your cluster scheduler."""
        
        help_label = tk.Label(help_content, text=help_text, 
                             font=('Segoe UI', 9), bg=self.colors['card'], 
                             fg=self.colors['text'], justify='left', wraplength=900)
        help_label.pack(anchor='w', padx=5, pady=5)
        
        watermark_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        watermark_frame.pack(fill='x', pady=10)
        watermark_label = tk.Label(watermark_frame, text='Designed by Abedi', 
                                   font=('Segoe UI', 8, 'italic'), 
                                   bg=self.colors['bg'], fg=self.colors['text_light'])
        watermark_label.pack()
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    # ========== Helper Methods ==========
    def _setup_scrollable_frame(self, parent_frame):
        """Helper method to create optimized scrollable frame with canvas"""
        canvas = tk.Canvas(parent_frame, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg'])
        
        # Optimized scroll region update with throttling
        _scroll_update_pending = [False]
        def update_scroll_region(event=None):
            if not _scroll_update_pending[0]:
                _scroll_update_pending[0] = True
                canvas.after_idle(lambda: _do_scroll_update())
        
        def _do_scroll_update():
            canvas.configure(scrollregion=canvas.bbox("all"))
            _scroll_update_pending[0] = False
        
        def on_canvas_configure(event):
            canvas_width = event.width
            items = canvas.find_all()
            if items:
                canvas.itemconfig(items[0], width=canvas_width)
            update_scroll_region()
        
        scrollable_frame.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", on_canvas_configure)
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Optimized mousewheel: single binding
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        def on_enter(event):
            canvas.focus_set()
        
        # Bind to canvas and parent frame only
        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Enter>", on_enter)
        parent_frame.bind("<MouseWheel>", on_mousewheel)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        return canvas, scrollable_frame
    def _browse_inputs(self):
        paths = filedialog.askdirectory(title="Pick folder with .com/.xyz")
        if paths:
            self.vars['INPUTS'].set(str(Path(paths)))

    def _browse_outdir(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d: self.vars['OUT_DIR'].set(d)

    def _browse_orca(self):
        p = filedialog.askopenfilename(title="Select ORCA executable")
        if p: self.vars['ORCA_PATH'].set(p)

    def _collect_orca_config(self):
        """Collect ORCA configuration from GUI - updated to support all input types"""
        def _to_int_or_none(s):
            if s is None:
                return None
            s = str(s).strip()
            if s == '' or s.lower() == 'none':
                return None
            try:
                return int(s)
            except (ValueError, TypeError):
                return None
        
        eps_raw = self.vars.get('EPSILON', tk.StringVar(value='')).get().strip()
        epsilon = None
        if eps_raw:
            try:
                epsilon = float(eps_raw)
            except Exception:
                epsilon = None
        
        input_type = self.vars.get('ORCA_INPUT_TYPE', tk.StringVar(value='xyz')).get()
        smiles_input = ''
        if input_type == 'smiles' and hasattr(self, 'orca_smiles_text'):
            smiles_input = self.orca_smiles_text.get('1.0', 'end-1c').strip()
        
        # Get widget values - ensure we get actual values, not empty strings that fall back to defaults
        method_val = self.cb_method.get().strip() if hasattr(self, 'cb_method') and self.cb_method else ''
        basis_val = self.cb_basis.get().strip() if hasattr(self, 'cb_basis') and self.cb_basis else ''
        solvent_model_val = self.cb_smodel.get().strip() if hasattr(self, 'cb_smodel') and self.cb_smodel else ''
        solvent_name_val = self.cb_sname.get().strip() if hasattr(self, 'cb_sname') and self.cb_sname else ''
        scheduler_val = self.cb_sched.get().strip() if hasattr(self, 'cb_sched') and self.cb_sched else ''
        queue_val = self.cb_queue.get().strip() if hasattr(self, 'cb_queue') and self.cb_queue else ''
        
        return dict(
            MODE=self.vars['MODE'].get(),
            STEP=int(self.vars['STEP'].get()),
            INPUT_TYPE=input_type,
            INPUTS=self.vars['INPUTS'].get().strip(),
            SMILES_INPUT=smiles_input,
            OUT_DIR=self.vars['OUT_DIR'].get().strip(),
            METHOD=method_val if method_val else ORCA_DEFAULTS['METHOD'],
            BASIS=basis_val if basis_val else ORCA_DEFAULTS['BASIS'],
            TD_NROOTS=int(self.vars['TD_NROOTS'].get()),
            TD_IROOT=int(self.vars['TD_IROOT'].get()),
            TD_TDA=bool(self.vars['TD_TDA'].get()),
            FOLLOW_IROOT=bool(self.vars['FOLLOW_IROOT'].get()),
            SOLVENT_MODEL=solvent_model_val if solvent_model_val else ORCA_DEFAULTS['SOLVENT_MODEL'],
            SOLVENT_NAME=solvent_name_val if solvent_name_val else ORCA_DEFAULTS['SOLVENT_NAME'],
            EPSILON=epsilon,
            NPROCS=int(self.vars['NPROCS'].get()),
            MAXCORE_MB=int(self.vars['MAXCORE_MB'].get()),
            ORCA_PATH=self.vars['ORCA_PATH'].get().strip(),
            SCHEDULER=scheduler_val if scheduler_val else ORCA_DEFAULTS['SCHEDULER'],
            QUEUE=queue_val if queue_val else ORCA_DEFAULTS['QUEUE'],
            WALLTIME=self.vars['WALLTIME'].get().strip(),
            PROJECT=self.vars.get('PROJECT', tk.StringVar(value='')).get().strip(),
            ACCOUNT=self.vars.get('ACCOUNT', tk.StringVar(value='')).get().strip(),
            CUSTOM_KEYWORDS=self.txt_ckw.get('1.0', 'end-1c').strip() if hasattr(self, 'txt_ckw') else ORCA_DEFAULTS['CUSTOM_KEYWORDS'],
            CUSTOM_BLOCK=self.txt_cblk.get('1.0', 'end-1c').strip() if hasattr(self, 'txt_cblk') else ORCA_DEFAULTS['CUSTOM_BLOCK'],
            CHARGE=_to_int_or_none(self.vars.get('CHARGE', tk.StringVar(value='')).get()),
            MULT=_to_int_or_none(self.vars.get('MULT', tk.StringVar(value='')).get()),
            REMOVE_PREFIX=self.vars.get('REMOVE_PREFIX', tk.StringVar(value='')).get().strip(),
            REMOVE_SUFFIX=self.vars.get('REMOVE_SUFFIX', tk.StringVar(value='')).get().strip(),
        )
    

    def _orca_preview(self):
        """Preview ORCA input files - supports xyz, com, log, and SMILES"""
        cfg = self._collect_orca_config()
        input_type = cfg.get('INPUT_TYPE', 'xyz')
        
        files_data = []
        
        if input_type == 'smiles':
            smiles_text = cfg['SMILES_INPUT'].strip()
            if not smiles_text:
                self.preview.delete("1.0","end")
                self.preview.insert("1.0","Please enter at least one SMILES string.")
                self.status.config(text="No SMILES input")
                return
            
            if not RDKIT_AVAILABLE:
                error_msg = "RDKit is required for SMILES input."
                if RDKIT_ERROR:
                    error_msg += f"\n\nError: {RDKIT_ERROR}"
                self.preview.delete("1.0","end")
                self.preview.insert("1.0", error_msg)
                self.status.config(text="RDKit not available")
                return
            
            lines = smiles_text.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parsed = parse_smiles_line(line)
                if parsed:
                    smiles, name, charge, mult = parsed
                    cm, coords, base_name = smiles_to_coords(
                        smiles, 
                        charge or cfg.get('CHARGE'),
                        mult or cfg.get('MULT'),
                        name
                    )
                    
                    # Apply prefix/suffix
                    prefix = cfg.get('REMOVE_PREFIX', '').strip()
                    suffix = cfg.get('REMOVE_SUFFIX', '').strip()
                    if name:
                        final_name = name
                        if prefix:
                            final_name = prefix + final_name
                        if suffix:
                            final_name = final_name + suffix
                    else:
                        final_name = f"molecule_{i+1}"
                        if prefix:
                            final_name = prefix + final_name
                        if suffix:
                            final_name = final_name + suffix
                    
                    files_data.append((final_name, cm, coords))
        else:
            files = orca_find_inputs(cfg['INPUTS'], input_type)
            if not files:
                ext_name = { 'xyz': '.xyz', 'com': '.com', 'log': '.log' }.get(input_type, 'files')
                self.preview.delete("1.0","end")
                self.preview.insert("1.0", f"No {ext_name} found for the given folder/glob.")
                self.status.config(text="No inputs")
                return
            
            for p in files:
                try:
                    cm, coords = orca_extract_geom(p, input_type)
                    base_name = p.stem
                    # Remove prefix/suffix if specified
                    prefix = cfg.get('REMOVE_PREFIX', '').strip()
                    suffix = cfg.get('REMOVE_SUFFIX', '').strip()
                    if prefix or suffix:
                        base_name = remove_prefix_suffix(base_name, prefix, suffix)
                    files_data.append((base_name, cm, coords))
                except Exception as e:
                    self.preview.delete("1.0","end")
                    self.preview.insert("1.0", f"Error processing {p.name}: {str(e)}")
                    self.status.config(text=f"Error: {p.name}")
                    return
        
        if not files_data:
            self.preview.delete("1.0","end")
            self.preview.insert("1.0","No valid inputs found.")
            self.status.config(text="No inputs")
            return

        steps_full = [1,2,4,7,9]
        chosen = steps_full if cfg['MODE']=="full" else [cfg['STEP']]

        chunks = []
        for base_name, cm, coords in files_data[:3]:  # Preview first 3
            for k in chosen:
                job = orca_jobname(k, base_name, cfg['METHOD'], cfg['BASIS'], cfg['SOLVENT_MODEL'], cfg['SOLVENT_NAME'])
                inp = orca_build_step(k, cfg, cm, coords)
                chunks.append(f"### {job}.inp\n" + "\n".join(inp))
        
        preview_text = "\n\n".join(chunks)
        if len(files_data) > 3:
            preview_text += f"\n\n... and {len(files_data) - 3} more input(s)"
        
        self.preview.delete("1.0","end")
        self.preview.insert("1.0", preview_text)
        self.status.config(text=f"Preview OK — {len(files_data)} input(s), steps={chosen}")

    def _orca_generate(self):
        """Generate ORCA input files - supports xyz, com, log, and SMILES"""
        cfg = self._collect_orca_config()
        input_type = cfg.get('INPUT_TYPE', 'xyz')
        
        files_data = []
        
        if input_type == 'smiles':
            smiles_text = cfg['SMILES_INPUT'].strip()
            if not smiles_text:
                messagebox.showerror('No input', 'Please enter at least one SMILES string.')
                return
            
            if not RDKIT_AVAILABLE:
                error_msg = "RDKit is required for SMILES input."
                if RDKIT_ERROR:
                    error_msg += f"\n\nError: {RDKIT_ERROR}"
                messagebox.showerror('RDKit required', error_msg)
                return
            
            lines = smiles_text.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parsed = parse_smiles_line(line)
                if parsed:
                    smiles, name, charge, mult = parsed
                    cm, coords, base_name = smiles_to_coords(
                        smiles, 
                        charge or cfg.get('CHARGE'),
                        mult or cfg.get('MULT'),
                        name
                    )
                    
                    # Apply prefix/suffix
                    prefix = cfg.get('REMOVE_PREFIX', '').strip()
                    suffix = cfg.get('REMOVE_SUFFIX', '').strip()
                    if name:
                        final_name = name
                        if prefix:
                            final_name = prefix + final_name
                        if suffix:
                            final_name = final_name + suffix
                    else:
                        final_name = f"molecule_{i+1}"
                        if prefix:
                            final_name = prefix + final_name
                        if suffix:
                            final_name = final_name + suffix
                    
                    files_data.append((final_name, cm, coords))
        else:
            files = orca_find_inputs(cfg['INPUTS'], input_type)
            if not files:
                ext_name = { 'xyz': '.xyz', 'com': '.com', 'log': '.log' }.get(input_type, 'files')
                messagebox.showerror("No inputs", f"No {ext_name} found.")
                return
            
            for p in files:
                try:
                    cm, coords = orca_extract_geom(p, input_type)
                    base_name = p.stem
                    # Remove prefix/suffix if specified
                    prefix = cfg.get('REMOVE_PREFIX', '').strip()
                    suffix = cfg.get('REMOVE_SUFFIX', '').strip()
                    if prefix or suffix:
                        base_name = remove_prefix_suffix(base_name, prefix, suffix)
                    files_data.append((base_name, cm, coords))
                except Exception as e:
                    messagebox.showerror("Error", f"Error processing {p.name}:\n{str(e)}")
                    return
        
        if not files_data:
            messagebox.showerror("No inputs", "No valid inputs found.")
            return
        
        out_path = Path(cfg['OUT_DIR'])
        out_path.mkdir(exist_ok=True)

        steps_full = [1,2,4,7,9]
        chosen = steps_full if cfg['MODE']=="full" else [cfg['STEP']]

        submit_lines: List[str] = []
        generated = []
        for base_name, cm, coords in files_data:
            for k in chosen:
                job = orca_jobname(k, base_name, cfg['METHOD'], cfg['BASIS'], cfg['SOLVENT_MODEL'], cfg['SOLVENT_NAME'])
                inp = orca_build_step(k, cfg, cm, coords)
                write_lines(out_path / f"{job}.inp", inp)
                write_exec(out_path / f"{job}.sh", orca_write_sh(job, cfg))
                if cfg['SCHEDULER']=="pbs":
                    submit_lines.append(f"qsub {job}.sh")
                elif cfg['SCHEDULER']=="slurm":
                    submit_lines.append(f"sbatch {job}.sh")
                else:
                    submit_lines.append(f"bash {job}.sh")
                generated.append(job)

        write_exec(out_path / "submit_all.sh", submit_lines)
        summary = f"Generated {len(generated)} jobs → {out_path.resolve()}"
        self.status.config(text=summary)
        self.preview.delete("1.0","end")
        self.preview.insert("1.0", summary + "\n\n" + "\n".join(f"- {j}" for j in generated[:200]))
        messagebox.showinfo("Done", summary)
        
        # Open output folder
        try:
            if sys.platform == "win32":
                os.startfile(out_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(out_path)])
            else:
                subprocess.run(["xdg-open", str(out_path)])
        except Exception:
            pass

    def _save_prefs(self):
        """Save preferences"""
        prefs = {
            'software': self.software.get(),
        }
        
        if self.software.get() == "orca":
            cfg = self._collect_orca_config()
            prefs.update(cfg)
            prefs['lists'] = dict(
                method=self.cb_method.values if self.cb_method else [],
                basis=self.cb_basis.values if self.cb_basis else [],
                smodel=self.cb_smodel.values if self.cb_smodel else [],
                sname=self.cb_sname.values if self.cb_sname else [],
                sched=self.cb_sched.values if self.cb_sched else [],
                queue=self.cb_queue.values if self.cb_queue else []
            )
        
        try:
            with open(PREFS_FILE, "w", encoding="utf-8") as f:
                json.dump(prefs, f, indent=2)
            self.status.config(text=f"Saved preferences → {PREFS_FILE}")
        except Exception as e:
            messagebox.showwarning("Save prefs", str(e))

    def _load_prefs(self):
        """Load preferences"""
        if not PREFS_FILE.exists():
            return
        
        try:
            with open(PREFS_FILE, "r", encoding="utf-8") as f:
                prefs = json.load(f)
            
            # Restore software selection
            if 'software' in prefs:
                self.software.set(prefs['software'])
                self._on_software_change()
            
            # Restore ORCA settings if applicable
            if self.software.get() == "orca" and self.cb_method:
                def _restore(cb: EditableCombo, key: str):
                    if cb and key in prefs.get('lists', {}):
                        vals = prefs['lists'][key]
                        cb.values = vals
                        cb.combo['values'] = vals
                
                _restore(self.cb_method, 'method')
                _restore(self.cb_basis, 'basis')
                _restore(self.cb_smodel, 'smodel')
                _restore(self.cb_sname, 'sname')
                _restore(self.cb_sched, 'sched')
                _restore(self.cb_queue, 'queue')
                
                # Restore scalar values
                for k in ['MODE', 'STEP', 'INPUTS', 'OUT_DIR', 'WALLTIME', 'PROJECT', 
                          'ACCOUNT', 'ORCA_PATH', 'TD_NROOTS', 'TD_IROOT', 'TD_TDA', 
                          'FOLLOW_IROOT', 'NPROCS', 'MAXCORE_MB', 'EPSILON']:
                    if k in prefs and k in self.vars:
                        try:
                            self.vars[k].set(prefs[k])
                        except Exception:
                            pass
                
                if 'METHOD' in prefs and self.cb_method:
                    self.cb_method.set(prefs['METHOD'])
                if 'BASIS' in prefs and self.cb_basis:
                    self.cb_basis.set(prefs['BASIS'])
                if 'SOLVENT_MODEL' in prefs and self.cb_smodel:
                    self.cb_smodel.set(prefs['SOLVENT_MODEL'])
                if 'SOLVENT_NAME' in prefs and self.cb_sname:
                    self.cb_sname.set(prefs['SOLVENT_NAME'])
                if 'SCHEDULER' in prefs and self.cb_sched:
                    self.cb_sched.set(prefs['SCHEDULER'])
                if 'QUEUE' in prefs and self.cb_queue:
                    self.cb_queue.set(prefs['QUEUE'])
                if 'CUSTOM_KEYWORDS' in prefs and self.txt_ckw:
                    self.txt_ckw.delete("1.0","end")
                    self.txt_ckw.insert("1.0", prefs['CUSTOM_KEYWORDS'])
                if 'CUSTOM_BLOCK' in prefs and self.txt_cblk:
                    self.txt_cblk.delete("1.0","end")
                    self.txt_cblk.insert("1.0", prefs['CUSTOM_BLOCK'])
                
                self.status.config(text=f"Loaded preferences from {PREFS_FILE}")
        except Exception as e:
            messagebox.showwarning("Load prefs", f"Could not load preferences: {e}")
    
    # ========== Gaussian Helper Methods ==========
    def _gaussian_set_mode(self, mode):
        """Set mode and update UI"""
        self.vars['MODE'].set(mode)
        if mode != 'multiple':
            for var in self.multi_step_vars.values():
                var.set(False)
        if mode == 'full':
            self.vars['STEP'].set(4)
        self._gaussian_on_mode_change()
        self._gaussian_update_mode_buttons()
    
    def _gaussian_set_single_step(self, step):
        """Set single step and update button appearance"""
        self.vars['STEP'].set(step)
        self._gaussian_update_step_buttons()
        self._gaussian_on_step_change()
    
    def _gaussian_toggle_multi_step(self, step):
        """Toggle multi-step selection"""
        var = self.multi_step_vars[step]
        var.set(not var.get())
        self._gaussian_update_multi_buttons()
        self._gaussian_on_step_change()
    
    def _gaussian_update_mode_buttons(self):
        """Update mode button appearances"""
        mode = self.vars['MODE'].get()
        for opt_value in ['full', 'single', 'multiple']:
            btn = getattr(self, f'_mode_btn_{opt_value}', None)
            if btn:
                if opt_value == mode:
                    btn.config(bg=self.colors['primary'], fg='white', relief='sunken')
                else:
                    btn.config(bg=self.colors['card'], fg=self.colors['text'], relief='raised')
    
    def _gaussian_update_step_buttons(self):
        """Update single step button appearances"""
        selected = self.vars['STEP'].get()
        for k in range(1, 8):
            btn = getattr(self, f'_step_btn_{k}', None)
            if btn:
                if k == selected:
                    btn.config(bg=self.colors['primary'], fg='white', relief='sunken')
                else:
                    btn.config(bg=self.colors['card'], fg=self.colors['text'], relief='raised')
    
    def _gaussian_update_multi_buttons(self):
        """Update multi-step button appearances"""
        for k in range(1, 8):
            var = self.multi_step_vars.get(k)
            btn = getattr(self, f'_multi_btn_{k}', None)
            if btn and var:
                if var.get():
                    btn.config(bg=self.colors['accent'], fg='white', relief='sunken')
                else:
                    btn.config(bg=self.colors['card'], fg=self.colors['text'], relief='raised')
    
    def _gaussian_on_mode_change(self):
        """Update UI when mode changes"""
        mode = self.vars['MODE'].get()
        if mode == 'single':
            self.single_step_frame.pack(side='left')
            self.multi_step_frame.pack_forget()
            self._gaussian_update_step_buttons()
        elif mode == 'multiple':
            self.single_step_frame.pack_forget()
            self.multi_step_frame.pack(side='left')
            self._gaussian_update_multi_buttons()
        else:  # full
            self.single_step_frame.pack_forget()
            self.multi_step_frame.pack_forget()
        if hasattr(self, 'step_route_frames'):
            self._gaussian_update_visible_routes(immediate=True)
    
    def _gaussian_on_step_change(self):
        """Update visible route editors when step selection changes"""
        if hasattr(self, 'step_route_frames'):
            self._gaussian_update_visible_routes(immediate=True)
    
    def _gaussian_update_visible_routes(self, immediate=False):
        """Show route editors for selected steps and update their routes"""
        mode = self.vars['MODE'].get()
        if mode == 'full':
            visible_steps = list(range(1, 8))
        elif mode == 'multiple':
            visible_steps = [k for k, v in self.multi_step_vars.items() if v.get()]
        else:  # single
            visible_steps = [int(self.vars['STEP'].get())]
        
        for step in range(1, 8):
            if step in self.step_route_frames:
                if step in visible_steps:
                    self.step_route_frames[step].pack(fill='x', pady=3, padx=3)
                else:
                    self.step_route_frames[step].pack_forget()
        
        self._gaussian_update_all_routes(immediate=immediate)
    
    def _gaussian_update_all_routes(self, immediate=False):
        """Update all route texts from current settings with debouncing"""
        # Cancel any pending update
        if self._route_update_job is not None:
            self.root.after_cancel(self._route_update_job)
            self._route_update_job = None
        
        def do_update():
            # Get widget values - ensure we get actual values, not empty strings that fall back to defaults
            functional_val = self.cb_func.get().strip() if hasattr(self, 'cb_func') and self.cb_func else ''
            basis_val = self.cb_basis.get().strip() if hasattr(self, 'cb_basis') and self.cb_basis else ''
            solvent_model_val = self.cb_smodel.get().strip() if hasattr(self, 'cb_smodel') and self.cb_smodel else ''
            solvent_name_val = self.cb_sname.get().strip() if hasattr(self, 'cb_sname') and self.cb_sname else ''
            
            cfg_temp = dict(
                FUNCTIONAL=functional_val if functional_val else GAUSSIAN_DEFAULTS['FUNCTIONAL'],
                BASIS=basis_val if basis_val else GAUSSIAN_DEFAULTS['BASIS'],
                SOLVENT_MODEL=solvent_model_val if solvent_model_val else GAUSSIAN_DEFAULTS['SOLVENT_MODEL'],
                SOLVENT_NAME=solvent_name_val if solvent_name_val else GAUSSIAN_DEFAULTS['SOLVENT_NAME'],
                TD_NSTATES=int(self.vars['TD_NSTATES'].get()),
                TD_ROOT=int(self.vars['TD_ROOT'].get()),
                STATE_TYPE=self.vars['STATE_TYPE'].get() if 'STATE_TYPE' in self.vars else GAUSSIAN_DEFAULTS['STATE_TYPE'],
                POP_FULL=bool(self.vars['POP_FULL'].get()),
                DISPERSION=bool(self.vars['DISPERSION'].get()),
                SOC_ENABLE=bool(self.vars['SOC_ENABLE'].get()) if 'SOC_ENABLE' in self.vars else GAUSSIAN_DEFAULTS['SOC_ENABLE'],
                MANUAL_ROUTES={},
            )
            
            # Create cache key - convert all values to hashable types
            cache_key = (
                cfg_temp['FUNCTIONAL'],
                cfg_temp['BASIS'],
                cfg_temp['SOLVENT_MODEL'],
                cfg_temp['SOLVENT_NAME'],
                cfg_temp['TD_NSTATES'],
                cfg_temp['TD_ROOT'],
                cfg_temp['STATE_TYPE'],
                cfg_temp['POP_FULL'],
                cfg_temp['DISPERSION'],
                cfg_temp['SOC_ENABLE'],
            )
            
            if cache_key in self._route_cache:
                cached_routes = self._route_cache[cache_key]
            else:
                route_funcs = {1: route_step1, 2: route_step2, 3: route_step3, 4: route_step4,
                               5: route_step5, 6: route_step6, 7: route_step7}
                cached_routes = {}
                for step, func in route_funcs.items():
                    cached_routes[step] = func(cfg_temp)
                # Cache only last 5 configs to avoid memory bloat
                if len(self._route_cache) > 5:
                    self._route_cache.pop(next(iter(self._route_cache)))
                self._route_cache[cache_key] = cached_routes
            
            # Update only visible route texts
            for step, route_text in self.step_route_texts.items():
                if step in cached_routes:
                    current_route = cached_routes[step]
                    # Only update if content changed
                    existing = route_text.get('1.0', 'end-1c')
                    if existing != current_route:
                        route_text.delete('1.0', 'end')
                        route_text.insert('1.0', current_route)
            
            self._route_update_job = None
        
        if immediate:
            do_update()
        else:
            # Debounce: wait 150ms before updating
            self._route_update_job = self.root.after(150, do_update)
    
    def _gaussian_on_input_type_change(self):
        """Show/hide input fields based on input type selection"""
        input_type = self.vars['INPUT_TYPE'].get()
        if input_type == 'smiles':
            self.gaussian_input_row.pack_forget()
            self.gaussian_smiles_row.pack(fill='both', expand=True, padx=10, pady=(0,8))
            self.gaussian_prefix_label.config(text='Add Prefix:')
            self.gaussian_suffix_label.config(text='Add Suffix:')
        else:
            self.gaussian_smiles_row.pack_forget()
            self.gaussian_input_row.pack(fill='x', padx=10, pady=(0,8))
            if input_type == 'log':
                self.gaussian_input_label.config(text='Input (.log files or folder):')
            else:
                self.gaussian_input_label.config(text='Input (.com files or folder):')
            self.gaussian_prefix_label.config(text='Remove Prefix:')
            self.gaussian_suffix_label.config(text='Remove Suffix:')
    
    def _gaussian_on_soc_enable_change(self):
        """Handle SOC enable/disable"""
        soc_enabled = self.vars['SOC_ENABLE'].get()
        
        if hasattr(self, 'singlet_rb') and hasattr(self, 'triplet_rb') and hasattr(self, 'mixed_rb'):
            if soc_enabled:
                self.singlet_rb.config(state='disabled')
                self.triplet_rb.config(state='disabled')
                self.mixed_rb.config(state='normal')
                self.vars['STATE_TYPE'].set('mixed')
            else:
                self.singlet_rb.config(state='normal')
                self.triplet_rb.config(state='normal')
                self.mixed_rb.config(state='normal')
        
        if hasattr(self, 'mode_buttons_frame') and hasattr(self, 'step_row'):
            if soc_enabled:
                for value in ['full', 'single', 'multiple']:
                    btn = getattr(self, f'_mode_btn_{value}', None)
                    if btn:
                        btn.config(state='disabled')
                for k in range(1, 8):
                    btn = getattr(self, f'_step_btn_{k}', None)
                    if btn:
                        btn.config(state='disabled')
                    btn = getattr(self, f'_multi_btn_{k}', None)
                    if btn:
                        btn.config(state='disabled')
            else:
                for value in ['full', 'single', 'multiple']:
                    btn = getattr(self, f'_mode_btn_{value}', None)
                    if btn:
                        btn.config(state='normal')
                for k in range(1, 8):
                    btn = getattr(self, f'_step_btn_{k}', None)
                    if btn:
                        btn.config(state='normal')
                    btn = getattr(self, f'_multi_btn_{k}', None)
                    if btn:
                        btn.config(state='normal')
        
        if hasattr(self, 'step_route_frames'):
            self._gaussian_update_all_routes(immediate=False)
    
    def _gaussian_browse_inputs(self):
        input_type = self.vars.get('INPUT_TYPE', tk.StringVar(value='com')).get()
        if input_type == 'log':
            path = filedialog.askdirectory(title='Pick folder with .log files')
        else:
            path = filedialog.askdirectory(title='Pick folder with .com files')
        if path:
            self.vars['INPUTS'].set(str(Path(path)))
    
    def _gaussian_browse_outdir(self):
        path = filedialog.askdirectory(title='Select output directory')
        if path:
            self.vars['OUT_DIR'].set(str(Path(path)))
    
    def _gaussian_show_chemdraw_help(self):
        """Show help dialog for ChemDraw integration"""
        help_text = """How to Use SMILES with Names from ChemDraw:

STEP 1 - Copy SMILES:
1. In ChemDraw, select ALL structures (Ctrl+A or drag select)
2. Go to: Edit → Copy As → SMILES (or Alt+Ctrl+C)
3. Paste directly into the text field above
   ✓ Multiple SMILES on one line (separated by periods) are AUTO-DETECTED

STEP 2 - Get Names from SVG:
1. In ChemDraw: File → Save As → SVG format
2. Click "Load Names from SVG" button above
3. Select your SVG file
4. Names will be automatically matched with your SMILES!

ALTERNATIVE - Manual Format:
You can also type manually:
  • Just SMILES: CCO
  • With name: FLIMBD_1:CCO
  • Tab-separated: FLIMBD_1<TAB>CCO
  • Comments: # This is a comment"""
        
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
    
    def _gaussian_load_names_from_svg(self):
        """Load names from SVG file"""
        if not XML_AVAILABLE:
            messagebox.showerror('XML Parser Required', 'XML parser is required to extract names from SVG files.')
            return
        
        file_path = filedialog.askopenfilename(
            title='Select SVG File (exported from ChemDraw)',
            filetypes=[('SVG Files', '*.svg'), ('All Files', '*.*')]
        )
        
        if not file_path:
            return
        
        try:
            names = extract_names_from_svg(Path(file_path))
            if not names:
                messagebox.showwarning('No Names Found', 'No text labels found in the SVG file.')
                return
            
            current_text = self.gaussian_smiles_text.get('1.0', 'end-1c').strip()
            if not current_text:
                messagebox.showwarning('No SMILES', 'Please paste your SMILES strings first.')
                return
            
            lines = current_text.split('\n')
            smiles_list = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '.' in line and len(line) > 50:
                    parts = line.split('.')
                    for part in parts:
                        part = part.strip()
                        if part and any(c in part for c in '()[]=+-CNOS0123456789#%'):
                            if ':' in part:
                                _, smiles = part.split(':', 1)
                                smiles_list.append(smiles.strip())
                            else:
                                smiles_list.append(part)
                else:
                    if ':' in line:
                        _, smiles = line.split(':', 1)
                        smiles_list.append(smiles.strip())
                    else:
                        smiles_list.append(line)
            
            if len(smiles_list) != len(names):
                result = messagebox.askyesno('Count Mismatch',
                    f'Found {len(names)} names but {len(smiles_list)} SMILES. Continue anyway?')
                if not result:
                    return
            
            formatted_lines = []
            for idx, smiles in enumerate(smiles_list):
                name = names[idx] if idx < len(names) else None
                if name:
                    formatted_lines.append(f"{name}:{smiles}")
                else:
                    formatted_lines.append(smiles)
            
            self.gaussian_smiles_text.delete('1.0', 'end')
            self.gaussian_smiles_text.insert('1.0', '\n'.join(formatted_lines))
            
            messagebox.showinfo('Success', f'Matched {min(len(names), len(smiles_list))} name(s) with SMILES!')
        except Exception as e:
            messagebox.showerror('Error', f'Could not extract names from SVG: {str(e)}')
    
    def _orca_on_input_type_change(self):
        """Show/hide input fields based on input type selection"""
        if not hasattr(self, 'orca_input_row') or not hasattr(self, 'orca_smiles_row'):
            return
        try:
            input_type = self.vars.get('ORCA_INPUT_TYPE', tk.StringVar(value='xyz'))
            if isinstance(input_type, tk.StringVar):
                input_type = input_type.get()
            else:
                input_type = 'xyz'
        except:
            input_type = 'xyz'
        
        if input_type == 'smiles':
            self.orca_input_row.pack_forget()
            self.orca_smiles_row.pack(fill='both', expand=True, padx=10, pady=(0,8))
        else:
            self.orca_smiles_row.pack_forget()
            self.orca_input_row.pack(fill='x', padx=10, pady=(0,8))
            if hasattr(self, 'orca_input_label'):
                if input_type == 'log':
                    self.orca_input_label.config(text='Input (.log files or folder):')
                elif input_type == 'com':
                    self.orca_input_label.config(text='Input (.com files or folder):')
                else:  # xyz
                    self.orca_input_label.config(text='Input (.xyz files or folder):')
    
    def _orca_browse_inputs(self):
        """Browse for input files based on selected input type"""
        try:
            input_type = self.vars.get('ORCA_INPUT_TYPE', tk.StringVar(value='xyz'))
            if isinstance(input_type, tk.StringVar):
                input_type = input_type.get()
            else:
                input_type = 'xyz'
        except:
            input_type = 'xyz'
        
        if input_type == 'log':
            path = filedialog.askdirectory(title='Pick folder with .log files')
        elif input_type == 'com':
            path = filedialog.askdirectory(title='Pick folder with .com files')
        else:  # xyz
            path = filedialog.askdirectory(title='Pick folder with .xyz files')
        if path:
            self.vars['INPUTS'].set(str(Path(path)))
    
    def _orca_show_chemdraw_help(self):
        """Show help dialog for ChemDraw integration"""
        help_text = """How to Use SMILES with Names from ChemDraw:

STEP 1 - Copy SMILES:
1. In ChemDraw, select ALL structures (Ctrl+A or drag select)
2. Go to: Edit → Copy As → SMILES (or Alt+Ctrl+C)
3. Paste into the SMILES Input box above

STEP 2 - Get Names from SVG:
1. In ChemDraw: File → Save As → SVG format
2. Click "Load Names from SVG" button above
3. Select your SVG file
4. Names will be automatically matched with your SMILES!

ALTERNATIVE - Manual Format:
You can also type manually:
  • Just SMILES: CCO
  • With name: FLIMBD_1:CCO
  • Tab-separated: FLIMBD_1<TAB>CCO
  • Comments: # This is a comment"""
        
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
    
    def _orca_load_names_from_svg(self):
        """Load names from SVG file"""
        if not XML_AVAILABLE:
            messagebox.showerror('XML Parser Required', 'XML parser is required to extract names from SVG files.')
            return
        
        file_path = filedialog.askopenfilename(
            title='Select SVG File (exported from ChemDraw)',
            filetypes=[('SVG Files', '*.svg'), ('All Files', '*.*')]
        )
        
        if not file_path:
            return
        
        try:
            names = extract_names_from_svg(Path(file_path))
            if not names:
                messagebox.showwarning('No Names Found', 'No text labels found in the SVG file.')
                return
            
            current_text = self.orca_smiles_text.get('1.0', 'end-1c').strip()
            if not current_text:
                messagebox.showwarning('No SMILES', 'Please paste your SMILES strings first.')
                return
            
            lines = current_text.split('\n')
            smiles_list = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '.' in line and len(line) > 50:
                    parts = line.split('.')
                    for part in parts:
                        part = part.strip()
                        if part and any(c in part for c in '()[]=+-CNOS0123456789#%'):
                            if ':' in part:
                                _, smiles = part.split(':', 1)
                                smiles_list.append(smiles.strip())
                            else:
                                smiles_list.append(part)
                else:
                    if ':' in line:
                        _, smiles = line.split(':', 1)
                        smiles_list.append(smiles.strip())
                    else:
                        smiles_list.append(line)
            
            if len(smiles_list) != len(names):
                result = messagebox.askyesno('Count Mismatch',
                    f'Found {len(names)} names but {len(smiles_list)} SMILES. Continue anyway?')
                if not result:
                    return
            
            formatted_lines = []
            for idx, smiles in enumerate(smiles_list):
                name = names[idx] if idx < len(names) else None
                if name:
                    formatted_lines.append(f"{name}:{smiles}")
                else:
                    formatted_lines.append(smiles)
            
            self.orca_smiles_text.delete('1.0', 'end')
            self.orca_smiles_text.insert('1.0', '\n'.join(formatted_lines))
            
            messagebox.showinfo('Success', f'Matched {min(len(names), len(smiles_list))} name(s) with SMILES!')
        except Exception as e:
            messagebox.showerror('Error', f'Could not extract names from SVG: {str(e)}')
    
    def _gaussian_tab_advanced(self):
        """Build Gaussian Advanced tab"""
        f = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(f, text='🔧 Advanced')
        
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
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def on_enter(event):
            canvas.focus_set()
        # Bind to both canvas and scrollable frame for better coverage
        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Enter>", on_enter)
        scrollable_frame.bind("<MouseWheel>", on_mousewheel)
        scrollable_frame.bind("<Enter>", on_enter)
        # Also bind to the frame itself
        f.bind("<MouseWheel>", lambda e: on_mousewheel(e))
        f.bind("<Enter>", lambda e: on_enter(e))
        
        route_card = self._gaussian_create_card(scrollable_frame, 'Route Editors (for Selected Steps)')
        route_card.pack(fill='x', padx=15, pady=8)
        
        route_header = tk.Frame(route_card, bg=self.colors['card'])
        route_header.pack(fill='x', padx=10, pady=(0,8))
        
        explanation_text = """Route Editor: Manually customize the Gaussian route card for each computational step.
• Override auto-generated routes with custom keywords
• Add specialized options (e.g., opt=modredundant, calcfc, tight, etc.)
• Routes are displayed for steps you've selected in the Main Settings tab
• Click "Update All Routes" to regenerate routes from current settings"""
        
        explanation_label = tk.Label(route_card, text=explanation_text, 
                                    font=('Segoe UI', 8), bg=self.colors['card'], 
                                    fg=self.colors['text'], justify='left', wraplength=750)
        explanation_label.pack(anchor='w', padx=10, pady=5)
        
        tk.Button(route_header, text='Update All Routes', command=self._gaussian_update_all_routes,
                 font=('Segoe UI', 9), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=12, pady=4, cursor='hand2').pack(side='right')
        
        if not hasattr(self, 'step_route_frames'):
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
            
            geom_frame = tk.Frame(step_frame, bg=self.colors['card'])
            geom_frame.pack(fill='x', padx=8, pady=5)
            tk.Label(geom_frame, text='Geometry Source:', font=('Segoe UI', 9),
                    bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,5))
            geom_var = tk.StringVar(value='default')
            self.step_geom_source_vars[step] = geom_var
            
            for label, val in [('Default', 'default'), ('Coords S1', 'coords_1'), ('Coords S4', 'coords_4'),
                              ('oldchk S1', 'oldchk_1'), ('oldchk S4', 'oldchk_4')]:
                tk.Radiobutton(geom_frame, text=label, variable=geom_var, value=val,
                              font=('Segoe UI', 8), bg=self.colors['card'], fg=self.colors['text'],
                              selectcolor=self.colors['card'], activebackground=self.colors['card'],
                              activeforeground=self.colors['text']).pack(side='left', padx=3)
            
            route_text = scrolledtext.ScrolledText(step_frame, height=2, wrap='none', font=('Consolas', 9),
                                                  bg='white', fg='black')
            route_text.pack(fill='x', padx=8, pady=(0,5))
            self.step_route_texts[step] = route_text
            step_frame.pack_forget()
        
        redundant_card = self._gaussian_create_card(scrollable_frame, 'Redundant Coordinates (Step 4 Only)')
        redundant_card.pack(fill='both', expand=True, padx=15, pady=8)
        
        redundant_content = tk.Frame(redundant_card, bg=self.colors['card'])
        redundant_content.pack(fill='both', expand=True, padx=10, pady=(0,10))
        
        redundant_explanation = """Redundant Internal Coordinates for Step 4 (Excited State Optimization).
Format: One coordinate per line (e.g., "D 4 5 6 7 180.0 F" for a frozen dihedral)
Example: D 4 5 6 7 180.0 F    (Freeze dihedral angle at 180 degrees)"""
        
        tk.Label(redundant_content, text=redundant_explanation,
                font=('Segoe UI', 8), bg=self.colors['card'], fg=self.colors['text'],
                justify='left', wraplength=750).pack(anchor='w', pady=3)
        # Always create new widget to avoid parent frame issues
        if hasattr(self, 'redundant_text') and self.redundant_text is not None:
            try:
                self.redundant_text.destroy()
            except:
                pass
        self.redundant_text = scrolledtext.ScrolledText(redundant_content, height=8, wrap='word', font=('Consolas', 10),
                                                        bg='white', fg='black')
        self.redundant_text.pack(fill='both', expand=True, pady=5)
        self.redundant_text.insert('1.0', GAUSSIAN_DEFAULTS.get('REDUNDANT_COORDS', ''))
        
        watermark_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        watermark_frame.pack(fill='x', pady=20)
        watermark_label = tk.Label(watermark_frame, text='Designed by Abedi', 
                                   font=('Segoe UI', 8, 'italic'), 
                                   bg=self.colors['bg'], fg=self.colors['text_light'])
        watermark_label.pack(pady=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self._gaussian_update_visible_routes(immediate=True)
    
    def _gaussian_tab_tict(self):
        """Gaussian TICT Rotation Tab - reuses implementation from gaussian_steps_gui"""
        # Since we import from gaussian_steps_gui, we can create a minimal wrapper
        # or reuse the same structure. For now, let's create it directly here.
        f = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(f, text='🔄 TICT Rotation')
        
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
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Initialize TICT variables for Gaussian (only advanced TICT mode)
        if not hasattr(self, 'tict_vars_gaussian'):
            self.tict_vars_gaussian = {
                'INPUT_FILE': tk.StringVar(value=""),
                'OUTPUT_DIR': tk.StringVar(value=""),
                'AXIS': tk.StringVar(value="3,10"),
                'BRANCH_A': tk.StringVar(value="11,18,22"),
                'BRANCH_A_STEP': tk.StringVar(value="-8.81"),
                'BRANCH_B': tk.StringVar(value="12-13,19-21,23-26"),
                'BRANCH_B_STEP': tk.StringVar(value="-9.36"),
                'NUM_STEPS': tk.StringVar(value="9"),
            }
        
        # File Selection Section
        file_card = self._gaussian_create_card(scrollable_frame, 'Input/Output Files')
        file_card.pack(fill='x', padx=15, pady=8)
        
        file_content = tk.Frame(file_card, bg=self.colors['card'])
        file_content.pack(fill='x', padx=10, pady=10)
        
        input_row = tk.Frame(file_content, bg=self.colors['card'])
        input_row.pack(fill='x', pady=5)
        tk.Label(input_row, text='Input File (.com):', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        tk.Entry(input_row, textvariable=self.tict_vars_gaussian['INPUT_FILE'], width=60,
                font=('Segoe UI', 9)).pack(side='left', fill='x', expand=True, padx=(0,5))
        tk.Button(input_row, text='Browse...', command=lambda: self._tict_browse_input_gaussian(),
                 font=('Segoe UI', 9), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=15, pady=4, cursor='hand2').pack(side='left')
        
        output_row = tk.Frame(file_content, bg=self.colors['card'])
        output_row.pack(fill='x', pady=5)
        tk.Label(output_row, text='Output Directory:', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        tk.Entry(output_row, textvariable=self.tict_vars_gaussian['OUTPUT_DIR'], width=60,
                font=('Segoe UI', 9)).pack(side='left', fill='x', expand=True, padx=(0,5))
        tk.Button(output_row, text='Browse...', command=lambda: self._tict_browse_output_gaussian(),
                 font=('Segoe UI', 9), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=15, pady=4, cursor='hand2').pack(side='left')
        
        # TICT Rotation Parameters (old style - advanced mode only)
        params_card = self._gaussian_create_card(scrollable_frame, 'TICT Rotation Parameters (1-Based Atom Indices)')
        params_card.pack(fill='x', padx=15, pady=8)
        params_content = tk.Frame(params_card, bg=self.colors['card'])
        params_content.pack(fill='x', padx=10, pady=10)
        params_content.columnconfigure(1, weight=1)
        
        row = 0
        tk.Label(params_content, text='Rotation Axis (2 atoms, e.g., "3,10"):', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(params_content, textvariable=self.tict_vars_gaussian['AXIS'], font=('Segoe UI', 9)).grid(row=row, column=1, sticky='ew', padx=5, pady=5)
        row += 1
        
        tk.Label(params_content, text='Branch A Indices:', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(params_content, textvariable=self.tict_vars_gaussian['BRANCH_A'], font=('Segoe UI', 9)).grid(row=row, column=1, sticky='ew', padx=5, pady=5)
        row += 1
        
        tk.Label(params_content, text='Branch A Step (degrees):', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(params_content, textvariable=self.tict_vars_gaussian['BRANCH_A_STEP'], font=('Segoe UI', 9), width=20).grid(row=row, column=1, sticky='w', padx=5, pady=5)
        row += 1
        
        tk.Label(params_content, text='Branch B Indices:', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(params_content, textvariable=self.tict_vars_gaussian['BRANCH_B'], font=('Segoe UI', 9)).grid(row=row, column=1, sticky='ew', padx=5, pady=5)
        row += 1
        
        tk.Label(params_content, text='Branch B Step (degrees):', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(params_content, textvariable=self.tict_vars_gaussian['BRANCH_B_STEP'], font=('Segoe UI', 9), width=20).grid(row=row, column=1, sticky='w', padx=5, pady=5)
        row += 1
        
        tk.Label(params_content, text='Number of Steps (0 to N inclusive):', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(params_content, textvariable=self.tict_vars_gaussian['NUM_STEPS'], font=('Segoe UI', 9), width=20).grid(row=row, column=1, sticky='w', padx=5, pady=5)
        row += 1
        
        help_text = """Enter atom indices using 1-based indexing (as in GaussView).
You can use ranges (e.g., "12-13") and comma-separated lists (e.g., "11,18,22").
Example: "12-13,19-21,23-26" means atoms 12,13,19,20,21,23,24,25,26."""
        tk.Label(params_content, text=help_text, font=('Segoe UI', 8), bg=self.colors['card'],
                fg=self.colors['text_light'], justify='left', wraplength=600).grid(row=row, column=0, columnspan=2, sticky='w', padx=5, pady=(10,5))
        
        # Status and Log
        log_card = self._gaussian_create_card(scrollable_frame, 'Status & Log')
        log_card.pack(fill='both', expand=True, padx=15, pady=8)
        log_content = tk.Frame(log_card, bg=self.colors['card'])
        log_content.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.tict_log_gaussian = scrolledtext.ScrolledText(log_content, wrap='word', font=('Consolas', 10),
                                                  bg='white', fg='black', height=12)
        self.tict_log_gaussian.pack(fill='both', expand=True)
        self.tict_log_gaussian.insert('1.0', 'Ready. Select input file and set parameters, then click "Generate Rotated Geometries".\n')
        
        # Generate Button (moved to be last, after status log)
        button_card = self._gaussian_create_card(scrollable_frame, None)
        button_card.pack(fill='x', padx=15, pady=8)
        button_frame = tk.Frame(button_card, bg=self.colors['card'])
        button_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(button_frame, text='Generate Rotated Geometries (Gaussian .com)', command=lambda: self._tict_generate_gaussian(),
                 font=('Segoe UI', 11, 'bold'), bg=self.colors['accent'], fg='white',
                 relief='flat', padx=25, pady=10, cursor='hand2').pack(side='left', padx=5)
        
        if not TICT_AVAILABLE:
            warning_label = tk.Label(button_frame, text='⚠️ TICT module not available', 
                                    font=('Segoe UI', 9), bg=self.colors['card'], fg='red')
            warning_label.pack(side='left', padx=10)
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Watermark
        watermark_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        watermark_frame.pack(fill='x', pady=20)
        watermark_label = tk.Label(watermark_frame, text='Designed by Abedi', 
                                   font=('Segoe UI', 8, 'italic'), 
                                   bg=self.colors['bg'], fg=self.colors['text_light'])
        watermark_label.pack()
    
    def _gaussian_tab_ai_assistant(self):
        """Gaussian AI Assistant Tab with Gemini Pro"""
        f = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(f, text='🤖 AI Assistant')
        
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
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Settings Section
        settings_card = self._gaussian_create_card(scrollable_frame, 'AI Settings')
        settings_card.pack(fill='x', padx=15, pady=8)
        
        settings_content = tk.Frame(settings_card, bg=self.colors['card'])
        settings_content.pack(fill='x', padx=10, pady=10)
        
        # Check Ollama status
        try:
            from ai_assistant import OLLAMA_AVAILABLE, GEMINI_AVAILABLE
            ollama_status = "✓ Available" if OLLAMA_AVAILABLE else "✗ Not installed"
            gemini_status = "✓ Available" if GEMINI_AVAILABLE else "✗ Not installed"
        except:
            ollama_status = "Unknown"
            gemini_status = "Unknown"
        
        status_row = tk.Frame(settings_content, bg=self.colors['card'])
        status_row.pack(fill='x', pady=5)
        tk.Label(status_row, text='Ollama (Free, Local):', font=('Segoe UI', 10, 'bold'),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        tk.Label(status_row, text=ollama_status, font=('Segoe UI', 10),
                bg=self.colors['card'], fg='green' if '✓' in ollama_status else 'red').pack(side='left', padx=(0,20))
        
        if not OLLAMA_AVAILABLE:
            import platform
            if platform.system() == 'Darwin':  # macOS
                install_text = 'Install: Download from https://ollama.com/download or: brew install ollama'
            else:  # Linux
                install_text = 'Install: curl -fsSL https://ollama.com/install.sh | sh'
            install_label = tk.Label(status_row, text=install_text, 
                                    font=('Segoe UI', 8), bg=self.colors['card'], fg=self.colors['text_light'])
            install_label.pack(side='left', padx=10)
        
        # Gemini API Key (optional)
        gemini_row = tk.Frame(settings_content, bg=self.colors['card'])
        gemini_row.pack(fill='x', pady=5)
        tk.Label(gemini_row, text='Gemini API Key (Optional):', font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['text']).pack(side='left', padx=(0,10))
        
        if not hasattr(self, 'gemini_api_key_var'):
            self.gemini_api_key_var = tk.StringVar()
            try:
                from ai_assistant import load_gemini_api_key
                existing_key = load_gemini_api_key()
                if existing_key:
                    self.gemini_api_key_var.set(existing_key)
            except:
                pass
        
        api_entry = tk.Entry(gemini_row, textvariable=self.gemini_api_key_var, show='*', width=40, font=('Segoe UI', 9))
        api_entry.pack(side='left', fill='x', expand=True, padx=(0,5))
        tk.Button(gemini_row, text='Save', command=lambda: self._save_gemini_key(api_entry.get()),
                 font=('Segoe UI', 9), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=15, pady=4, cursor='hand2').pack(side='left')
        tk.Label(gemini_row, text='Get free key: https://makersuite.google.com/app/apikey', 
                font=('Segoe UI', 8), bg=self.colors['card'], fg=self.colors['text_light']).pack(side='left', padx=10)
        
        # Chat Interface
        chat_card = self._gaussian_create_card(scrollable_frame, 'Chat with AI Assistant')
        chat_card.pack(fill='both', expand=True, padx=15, pady=8)
        
        chat_content = tk.Frame(chat_card, bg=self.colors['card'])
        chat_content.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Chat display area
        chat_display = scrolledtext.ScrolledText(chat_content, wrap='word', font=('Segoe UI', 10),
                                                  bg='white', fg='black', height=20, state=tk.DISABLED)
        chat_display.pack(fill='both', expand=True, pady=(0,10))
        chat_display.tag_config('user', foreground='blue', font=('Segoe UI', 10, 'bold'))
        chat_display.tag_config('assistant', foreground='green', font=('Segoe UI', 10))
        chat_display.tag_config('error', foreground='red', font=('Segoe UI', 10))
        
        # Initial message
        chat_display.config(state=tk.NORMAL)
        chat_display.insert('1.0', '🤖 AI Assistant: Hello! I can help you set up Gaussian quantum chemistry calculations.\n\n'
                                   'Tell me what you want to calculate, and I\'ll guide you through the process.\n\n'
                                   'Examples:\n'
                                   '- "I want to optimize a benzene molecule in DMSO"\n'
                                   '- "Calculate excited states for a TICT molecule"\n'
                                   '- "Set up a full workflow for fluorescence calculations"\n\n', 'assistant')
        chat_display.config(state=tk.DISABLED)
        
        # Input area
        input_frame = tk.Frame(chat_content, bg=self.colors['card'])
        input_frame.pack(fill='x')
        
        input_entry = tk.Entry(input_frame, font=('Segoe UI', 10))
        input_entry.pack(side='left', fill='x', expand=True, padx=(0,5))
        input_entry.bind('<Return>', lambda e: self._send_ai_message(input_entry, chat_display, 'gaussian'))
        
        button_frame = tk.Frame(input_frame, bg=self.colors['card'])
        button_frame.pack(side='left')
        
        tk.Button(button_frame, text='Send', command=lambda: self._send_ai_message(input_entry, chat_display, 'gaussian'),
                 font=('Segoe UI', 9, 'bold'), bg=self.colors['accent'], fg='white',
                 relief='flat', padx=20, pady=5, cursor='hand2').pack(side='left', padx=2)
        tk.Button(button_frame, text='Generate Files', command=lambda: self._generate_files_from_ai_conversation(chat_display, 'gaussian'),
                 font=('Segoe UI', 9, 'bold'), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=15, pady=5, cursor='hand2').pack(side='left', padx=2)
        tk.Button(button_frame, text='Clear', command=lambda: self._clear_ai_chat(chat_display),
                 font=('Segoe UI', 9), bg=self.colors['secondary'], fg='white',
                 relief='flat', padx=15, pady=5, cursor='hand2').pack(side='left', padx=2)
        
        # Store references
        self.gaussian_chat_display = chat_display
        self.gaussian_chat_input = input_entry
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Watermark
        watermark_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        watermark_frame.pack(fill='x', pady=20)
        watermark_label = tk.Label(watermark_frame, text='Powered by Ollama (Free) / Gemini | Designed by Abedi', 
                                   font=('Segoe UI', 8, 'italic'), 
                                   bg=self.colors['bg'], fg=self.colors['text_light'])
        watermark_label.pack()
    
    def _gaussian_tab_generate(self):
        """Build Gaussian Generate tab"""
        f = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(f, text='🚀 Generate')
        
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
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def on_enter(event):
            canvas.focus_set()
        # Bind to both canvas and scrollable frame for better coverage
        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Enter>", on_enter)
        scrollable_frame.bind("<MouseWheel>", on_mousewheel)
        scrollable_frame.bind("<Enter>", on_enter)
        # Also bind to the frame itself
        f.bind("<MouseWheel>", lambda e: on_mousewheel(e))
        f.bind("<Enter>", lambda e: on_enter(e))
        
        button_card = self._gaussian_create_card(scrollable_frame, None)
        button_card.pack(fill='x', padx=15, pady=10)
        button_frame = tk.Frame(button_card, bg=self.colors['card'])
        button_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(button_frame, text='Preview', command=self._gaussian_preview,
                 font=('Segoe UI', 10, 'bold'), bg=self.colors['secondary'], fg='white',
                 relief='flat', padx=20, pady=8, cursor='hand2').pack(side='left', padx=5)
        tk.Button(button_frame, text='Generate', command=self._gaussian_generate,
                 font=('Segoe UI', 10, 'bold'), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=20, pady=8, cursor='hand2').pack(side='left', padx=5)
        tk.Button(button_frame, text='Save Prefs', command=self._gaussian_save_prefs,
                 font=('Segoe UI', 9), bg=self.colors['accent'], fg='white',
                 relief='flat', padx=15, pady=8, cursor='hand2').pack(side='left', padx=5)
        tk.Button(button_frame, text='Load Prefs', command=self._gaussian_load_prefs,
                 font=('Segoe UI', 9), bg=self.colors['accent'], fg='white',
                 relief='flat', padx=15, pady=8, cursor='hand2').pack(side='left', padx=5)
        tk.Button(button_frame, text='Reset to Defaults', command=self._gaussian_reset_to_defaults,
                 font=('Segoe UI', 9), bg=self.colors['primary'], fg='white',
                 relief='flat', padx=15, pady=8, cursor='hand2').pack(side='left', padx=5)
        tk.Button(button_frame, text='Generate PySOC Scripts', command=self._gaussian_run_pysoc,
                 font=('Segoe UI', 9), bg=self.colors['accent'], fg='white',
                 relief='flat', padx=15, pady=8, cursor='hand2').pack(side='left', padx=5)
        
        status_card = self._gaussian_create_card(scrollable_frame, None)
        status_card.pack(fill='x', padx=15, pady=(0,10))
        status_frame = tk.Frame(status_card, bg=self.colors['card'])
        status_frame.pack(fill='x', padx=10, pady=8)
        # Always create new status widget to avoid issues
        if hasattr(self, 'gaussian_status') and self.gaussian_status is not None:
            try:
                self.gaussian_status.destroy()
            except:
                pass
        self.gaussian_status = tk.Label(status_frame, text='Ready', font=('Segoe UI', 10, 'bold'),
                              bg=self.colors['card'], fg=self.colors['primary'])
        self.gaussian_status.pack(anchor='w')
        
        preview_card = self._gaussian_create_card(scrollable_frame, 'Preview & Output')
        preview_card.pack(fill='both', expand=True, padx=15, pady=(0,10))
        preview_content = tk.Frame(preview_card, bg=self.colors['card'])
        preview_content.pack(fill='both', expand=True, padx=10, pady=10)
        # Always create new preview widget to avoid issues
        if hasattr(self, 'gaussian_preview') and self.gaussian_preview is not None:
            try:
                self.gaussian_preview.destroy()
            except:
                pass
        self.gaussian_preview = scrolledtext.ScrolledText(preview_content, wrap='word', font=('Consolas', 10),
                                                bg='white', fg='black', height=20)
        self.gaussian_preview.pack(fill='both', expand=True)
        
        # Help Section - Generation Help
        help_card = self._gaussian_create_card(scrollable_frame, '📚 Generation Help')
        help_card.pack(fill='x', padx=15, pady=(0,10))
        help_content = tk.Frame(help_card, bg=self.colors['card'])
        help_content.pack(fill='x', padx=10, pady=(0,10))
        
        help_text = """Preview: Shows a preview of the input files that will be generated (first 3 inputs) | Generate: Creates all .com and .sh files in the output directory | Save Prefs: Saves all current settings to a preferences file | Load Prefs: Loads previously saved settings | Reset to Defaults: Resets all fields to default values | Generate PySOC Scripts: Creates scripts for Spin-Orbit Coupling calculations (after Gaussian jobs complete) | After generation, the output folder will automatically open."""
        
        help_label = tk.Label(help_content, text=help_text, 
                             font=('Segoe UI', 9), bg=self.colors['card'], 
                             fg=self.colors['text'], justify='left', wraplength=900)
        help_label.pack(anchor='w', padx=5, pady=5)
        
        watermark_frame = tk.Frame(scrollable_frame, bg=self.colors['bg'])
        watermark_frame.pack(fill='x', pady=10)
        watermark_label = tk.Label(watermark_frame, text='Designed by Abedi', 
                                   font=('Segoe UI', 8, 'italic'), 
                                   bg=self.colors['bg'], fg=self.colors['text_light'])
        watermark_label.pack()
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _gaussian_collect(self):
        """Collect Gaussian configuration from GUI"""
        def _to_int_or_none(s):
            if s is None:
                return None
            s = str(s).strip()
            if s == '' or s.lower() == 'none':
                return None
            try:
                return int(s)
            except (ValueError, TypeError):
                return None
        
        inline = [k for k,v in self.inline_vars.items() if v.get()]
        multi_steps = [k for k,v in self.multi_step_vars.items() if v.get()]
        
        manual_routes = {}
        if hasattr(self, 'step_route_texts'):
            for step, route_text in self.step_route_texts.items():
                route = route_text.get('1.0', 'end-1c').strip()
                if route:
                    manual_routes[step] = route
        
        geom_sources = {}
        if hasattr(self, 'step_geom_source_vars'):
            for step, var in self.step_geom_source_vars.items():
                source = var.get()
                if source != 'default':
                    geom_sources[step] = source
        
        redundant_coords = ""
        if hasattr(self, 'redundant_text'):
            redundant_coords = self.redundant_text.get('1.0', 'end-1c').strip()
        
        mode = self.vars['MODE'].get()
        
        # Get widget values - ensure we get actual values, not empty strings that fall back to defaults
        functional_val = self.cb_func.get().strip() if hasattr(self, 'cb_func') and self.cb_func else ''
        basis_val = self.cb_basis.get().strip() if hasattr(self, 'cb_basis') and self.cb_basis else ''
        solvent_model_val = self.cb_smodel.get().strip() if hasattr(self, 'cb_smodel') and self.cb_smodel else ''
        solvent_name_val = self.cb_sname.get().strip() if hasattr(self, 'cb_sname') and self.cb_sname else ''
        scheduler_val = self.cb_sched.get().strip() if hasattr(self, 'cb_sched') and self.cb_sched else ''
        queue_val = self.cb_queue.get().strip() if hasattr(self, 'cb_queue') and self.cb_queue else ''
        
        # Only use defaults if widget doesn't exist or value is truly empty
        cfg = dict(
            MODE=mode,
            STEP=int(self.vars['STEP'].get()),
            INPUT_TYPE=self.vars['INPUT_TYPE'].get(),
            INPUTS=self.vars['INPUTS'].get().strip(),
            SMILES_INPUT=self.gaussian_smiles_text.get('1.0', 'end-1c').strip() if hasattr(self, 'gaussian_smiles_text') else '',
            OUT_DIR=self.vars['OUT_DIR'].get().strip(),
            FUNCTIONAL=functional_val if functional_val else GAUSSIAN_DEFAULTS['FUNCTIONAL'],
            BASIS=basis_val if basis_val else GAUSSIAN_DEFAULTS['BASIS'],
            SOLVENT_MODEL=solvent_model_val if solvent_model_val else GAUSSIAN_DEFAULTS['SOLVENT_MODEL'],
            SOLVENT_NAME=solvent_name_val if solvent_name_val else GAUSSIAN_DEFAULTS['SOLVENT_NAME'],
            TD_NSTATES=int(self.vars['TD_NSTATES'].get()),
            TD_ROOT=int(self.vars['TD_ROOT'].get()),
            STATE_TYPE=self.vars['STATE_TYPE'].get() if 'STATE_TYPE' in self.vars else GAUSSIAN_DEFAULTS['STATE_TYPE'],
            POP_FULL=bool(self.vars['POP_FULL'].get()),
            DISPERSION=bool(self.vars['DISPERSION'].get()),
            NPROC=int(self.vars['NPROC'].get()),
            MEM=self.vars['MEM'].get().strip(),
            SCHEDULER=scheduler_val if scheduler_val else GAUSSIAN_DEFAULTS['SCHEDULER'],
            QUEUE=queue_val if queue_val else GAUSSIAN_DEFAULTS['QUEUE'],
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
            SOC_ENABLE=bool(self.vars['SOC_ENABLE'].get()) if 'SOC_ENABLE' in self.vars else GAUSSIAN_DEFAULTS['SOC_ENABLE'],
        )
        return cfg
    
    def _gaussian_preview(self):
        """Preview Gaussian input files"""
        # Ensure preview and status widgets exist
        if not hasattr(self, 'gaussian_preview') or self.gaussian_preview is None:
            messagebox.showerror('Error', 'Preview widget not initialized. Please restart the application.')
            return
        if not hasattr(self, 'gaussian_status') or self.gaussian_status is None:
            messagebox.showerror('Error', 'Status widget not initialized. Please restart the application.')
            return
            
        cfg = self._gaussian_collect()
        
        if cfg['INPUT_TYPE'] == 'smiles':
            smiles_text = cfg['SMILES_INPUT'].strip()
            if not smiles_text:
                self.gaussian_preview.delete('1.0','end')
                self.gaussian_preview.insert('1.0','Please enter at least one SMILES string.')
                self.gaussian_status.config(text='No SMILES input')
                return
            
            if not RDKIT_AVAILABLE:
                error_msg = "RDKit is required for SMILES input."
                if RDKIT_ERROR:
                    error_msg += f"\n\nError: {RDKIT_ERROR}"
                self.gaussian_preview.delete('1.0','end')
                self.gaussian_preview.insert('1.0', error_msg)
                self.gaussian_status.config(text='RDKit not available')
                return
            
            lines = smiles_text.split('\n')
            smiles_data = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '.' in line and len(line) > 50:
                    parts = line.split('.')
                    for part in parts:
                        part = part.strip()
                        if part and any(c in part for c in '()[]=+-CNOS0123456789#%'):
                            name, smiles = parse_smiles_line(part)
                            if smiles:
                                smiles_data.append((name, smiles))
                else:
                    name, smiles = parse_smiles_line(line)
                    if smiles:
                        smiles_data.append((name, smiles))
            
            if not smiles_data:
                self.gaussian_preview.delete('1.0','end')
                self.gaussian_preview.insert('1.0','Please enter at least one SMILES string.')
                self.gaussian_status.config(text='No SMILES input')
                return
            
            files_data = []
            for idx, (name, smiles) in enumerate(smiles_data[:10], 1):
                try:
                    cm, coords, base = smiles_to_coords(smiles, cfg['CHARGE'], cfg['MULT'], name)
                    cm = cm_override(cm, cfg['CHARGE'], cfg['MULT'])
                    if not name:
                        base = f"{base}_{idx}"
                    if cfg['REMOVE_PREFIX']:
                        base = cfg['REMOVE_PREFIX'] + base
                    if cfg['REMOVE_SUFFIX']:
                        base = base + cfg['REMOVE_SUFFIX']
                    files_data.append((base, cm, coords))
                except Exception as e:
                    pass
        elif cfg['INPUT_TYPE'] == 'log':
            files = find_geoms(cfg['INPUTS'], input_type='log')
            if not files:
                self.gaussian_preview.delete('1.0','end')
                self.gaussian_preview.insert('1.0','No .log files found.')
                self.gaussian_status.config(text='No inputs')
                return
            files_data = []
            for p in files[:3]:
                try:
                    cm, coords, base = parse_gaussian_log(p)
                    cm = cm_override(cm, cfg['CHARGE'], cfg['MULT'])
                    if cfg['REMOVE_PREFIX'] or cfg['REMOVE_SUFFIX']:
                        cleaned = remove_prefix_suffix(p.name, cfg['REMOVE_PREFIX'], cfg['REMOVE_SUFFIX'])
                        if '.' in cleaned:
                            base = cleaned.rsplit('.', 1)[0]
                        else:
                            base = cleaned
                    files_data.append((base, cm, coords))
                except Exception as e:
                    pass
        else:
            files = find_geoms(cfg['INPUTS'], input_type='com')
            if not files:
                self.gaussian_preview.delete('1.0','end')
                self.gaussian_preview.insert('1.0','No .com files found.')
                self.gaussian_status.config(text='No inputs')
                return
            files_data = []
            for p in files[:3]:
                cm, coords = extract_cm_coords(read_lines(p))
                cm = cm_override(cm, cfg['CHARGE'], cfg['MULT'])
                base = p.stem
                if cfg['REMOVE_PREFIX'] or cfg['REMOVE_SUFFIX']:
                    base = remove_prefix_suffix(p.name, cfg['REMOVE_PREFIX'], cfg['REMOVE_SUFFIX']).replace('.com', '')
                files_data.append((base, cm, coords))
        
        if cfg.get('SOC_ENABLE', False):
            chunks = []
            for base, cm, coords in files_data:
                solv_tag_val = solv_tag(cfg['SOLVENT_MODEL'], cfg['SOLVENT_NAME'])
                job = f"{base}_{cfg['FUNCTIONAL']}_{cfg['BASIS']}_{solv_tag_val}_SOC"
                title = f"PySOC {cfg['FUNCTIONAL']}/{cfg['BASIS']}"
                route = route_step2(cfg)
                save_rwf = True
                lines = make_com_inline(job, cfg['NPROC'], cfg['MEM'], route, title, cm, coords, save_rwf=save_rwf)
                chunks.append(f"### {job}.com\n"+"\n".join(lines))
        else:
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
                    if k == 7:
                        route = step_route(k, cfg) + " geom=check guess=read"
                        lines = make_com_linked(job, cfg['NPROC'], cfg['MEM'], f"{jobname(6, base, cfg)}.chk", route, f"{title} {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm, save_rwf=save_rwf)
                    elif source and source.startswith('oldchk_'):
                        oldchk_name = f"{jobname(1 if source == 'oldchk_1' else 4, base, cfg)}.chk"
                        route = step_route(k, cfg) + " geom=check guess=read"
                        lines = make_com_linked(job, cfg['NPROC'], cfg['MEM'], oldchk_name, route, f"{title} {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm, save_rwf=save_rwf)
                    elif cfg['MODE'] == 'full' and k!=1 and (k not in (cfg['INLINE_STEPS'] or [])) and not source:
                        route = step_route(k, cfg) + " geom=check guess=read"
                        lines = make_com_linked(job, cfg['NPROC'], cfg['MEM'], f"{jobname(1 if k<5 else 4, base, cfg)}.chk", route, f"{title} {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm, save_rwf=save_rwf)
                    else:
                        coords_use = add_redundant_coords(coords.copy(), cfg.get('REDUNDANT_COORDS', ""), k)
                        route = step_route(k, cfg)
                        lines = make_com_inline(job, cfg['NPROC'], cfg['MEM'], route, f"{title} {cfg['FUNCTIONAL']}/{cfg['BASIS']}", cm, coords_use, save_rwf=save_rwf)
                    chunks.append(f"### {job}.com\n"+"\n".join(lines))
        
        self.gaussian_preview.delete('1.0','end')
        self.gaussian_preview.insert('1.0','\n\n'.join(chunks))
        self.gaussian_status.config(text=f"Preview OK — {len(files_data)} input(s)")
    
    def _gaussian_generate(self):
        """Generate Gaussian input files"""
        cfg = self._gaussian_collect()
        
        if cfg['INPUT_TYPE'] == 'smiles':
            smiles_text = cfg['SMILES_INPUT'].strip()
            if not smiles_text:
                messagebox.showerror('No input', 'Please enter at least one SMILES string.')
                return
            
            if not RDKIT_AVAILABLE:
                error_msg = "RDKit is required for SMILES input."
                if RDKIT_ERROR:
                    error_msg += f"\n\nError: {RDKIT_ERROR}"
                messagebox.showerror('RDKit required', error_msg)
                return
            
            lines = smiles_text.split('\n')
            smiles_data = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '.' in line and len(line) > 50:
                    parts = line.split('.')
                    for part in parts:
                        part = part.strip()
                        if part and any(c in part for c in '()[]=+-CNOS0123456789#%'):
                            name, smiles = parse_smiles_line(part)
                            if smiles:
                                smiles_data.append((name, smiles))
                else:
                    name, smiles = parse_smiles_line(line)
                    if smiles:
                        smiles_data.append((name, smiles))
            
            if not smiles_data:
                messagebox.showerror('No input', 'Please enter at least one SMILES string.')
                return
            
            files_data = []
            errors = []
            for idx, (name, smiles) in enumerate(smiles_data, 1):
                try:
                    cm, coords, base = smiles_to_coords(smiles, cfg['CHARGE'], cfg['MULT'], name)
                    cm = cm_override(cm, cfg['CHARGE'], cfg['MULT'])
                    if not name:
                        base = f"{base}_{idx}"
                    if cfg['REMOVE_PREFIX']:
                        base = cfg['REMOVE_PREFIX'] + base
                    if cfg['REMOVE_SUFFIX']:
                        base = base + cfg['REMOVE_SUFFIX']
                    files_data.append((base, cm, coords))
                except Exception as e:
                    errors.append(f"Entry {idx}: {str(e)}")
            
            if errors and not files_data:
                messagebox.showerror('SMILES Processing Errors', '\n'.join(errors[:5]))
                return
        elif cfg['INPUT_TYPE'] == 'log':
            files = find_geoms(cfg['INPUTS'], input_type='log')
            if not files:
                messagebox.showerror('No inputs', 'No .log files found.')
                return
            files_data = []
            errors = []
            for p in files:
                try:
                    cm, coords, base = parse_gaussian_log(p)
                    cm = cm_override(cm, cfg['CHARGE'], cfg['MULT'])
                    if cfg['REMOVE_PREFIX'] or cfg['REMOVE_SUFFIX']:
                        cleaned = remove_prefix_suffix(p.name, cfg['REMOVE_PREFIX'], cfg['REMOVE_SUFFIX'])
                        if '.' in cleaned:
                            base = cleaned.rsplit('.', 1)[0]
                        else:
                            base = cleaned
                    files_data.append((base, cm, coords))
                except Exception as e:
                    errors.append(f"{p.name}: {str(e)}")
            
            if errors and not files_data:
                messagebox.showerror('Log Processing Errors', '\n'.join(errors[:5]))
                return
        else:
            files = find_geoms(cfg['INPUTS'], input_type='com')
            if not files:
                messagebox.showerror('No inputs', 'No .com files found.')
                return
            files_data = []
            for p in files:
                base = p.stem
                if cfg['REMOVE_PREFIX'] or cfg['REMOVE_SUFFIX']:
                    base = remove_prefix_suffix(p.name, cfg['REMOVE_PREFIX'], cfg['REMOVE_SUFFIX']).replace('.com', '')
                cm, coords = extract_cm_coords(read_lines(p))
                cm = cm_override(cm, cfg['CHARGE'], cfg['MULT'])
                files_data.append((base, cm, coords))
        
        out = Path(cfg['OUT_DIR'])
        out.mkdir(exist_ok=True)
        submit_all: List[str] = []
        submit_by_step: dict[int, List[str]] = {k: [] for k in range(1,8)}
        formchk_by_step: dict[int, List[str]] = {k: [] for k in range(1,8)}
        
        if cfg.get('SOC_ENABLE', False):
            for base, cm, coords in files_data:
                solv_tag_val = solv_tag(cfg['SOLVENT_MODEL'], cfg['SOLVENT_NAME'])
                job = f"{base}_{cfg['FUNCTIONAL']}_{cfg['BASIS']}_{solv_tag_val}_SOC"
                title = f"PySOC {cfg['FUNCTIONAL']}/{cfg['BASIS']}"
                route = route_step2(cfg)
                save_rwf = True
                com = make_com_inline(job, cfg['NPROC'], cfg['MEM'], route, title, cm, coords, save_rwf=save_rwf)
                write_lines(out / f"{job}.com", com)
                write_lines(out / f"{job}.sh", write_sh(job, cfg))
                sched_cmd = ("qsub " if cfg['SCHEDULER']=="pbs" else ("sbatch " if cfg['SCHEDULER']=="slurm" else "bash ")) + f"{job}.sh"
                submit_all.append(sched_cmd)
        else:
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
            
            for k in range(1,8):
                if submit_by_step[k]: write_exec(out / f'{k:02d}sub.sh', submit_by_step[k])
                if formchk_by_step[k]: write_exec(out / f'{k:02d}formchk.sh', formchk_by_step[k])
        
        write_exec(out / 'submit_all.sh', submit_all)
        msg = f"Generated jobs for {len(files_data)} input(s) → {out.resolve()}"
        self.gaussian_status.config(text=msg)
        self.gaussian_preview.delete('1.0','end')
        self.gaussian_preview.insert('1.0', msg + "\n\n" + "\n".join(submit_all[:200]))
        
        try:
            if os.name == 'nt':
                os.startfile(str(out.resolve()))
            elif sys.platform == 'darwin':
                subprocess.run(['open', str(out.resolve())])
            else:
                subprocess.run(['xdg-open', str(out.resolve())])
        except Exception:
            pass
        
        messagebox.showinfo('Done', msg)
    
    def _gaussian_run_pysoc(self):
        """Generate PySOC submission scripts"""
        log_dir = filedialog.askdirectory(title='Select directory containing Gaussian .log files')
        if not log_dir:
            return
        
        log_dir = Path(log_dir)
        log_files = sorted(log_dir.glob("*_SOC.log"))
        
        if not log_files:
            messagebox.showwarning('No log files', f'No *_SOC.log files found in {log_dir}')
            return
        
        output_dir = filedialog.askdirectory(title='Select directory to save PySOC submission scripts')
        if not output_dir:
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        master_script = output_dir / "run_pysoc.sh"
        script_lines = [
            "#!/bin/bash",
            "# PySOC calculation script for all _SOC.log files",
            "# Generated by Quantum Steps Generator",
            "",
            "for logfile in *_SOC.log; do",
            "    base_name=\"${logfile%.log}\"",
            "    rwf_file=\"${base_name}.rwf\"",
            "    if [ -f \"$rwf_file\" ]; then",
            "        echo \"Processing $logfile ...\"",
            "        pysoc \"$logfile\" --rwf_file \"$rwf_file\" > \"${base_name}_RESULTS.txt\" 2>&1",
            "        pysoc \"$logfile\" --rwf_file \"$rwf_file\" -c > \"${base_name}_RESULTS.csv\" 2>&1",
            "    else",
            "        echo \"WARNING: Missing .rwf file for $logfile. Skipping...\"",
            "    fi",
            "done",
            "",
            "echo \"All calculations complete.\"",
        ]
        
        write_exec(master_script, script_lines)
        
        analysis_script = output_dir / "combine_pysoc_results.py"
        analysis_script_lines = [
            "#!/usr/bin/env python3",
            '"""Combine PySOC Results into Excel"""',
            "",
            "import pandas as pd",
            "import glob",
            "import sys",
            "from pathlib import Path",
            "",
            "def combine_pysoc_results():",
            '    """Combine all PySOC CSV results into a single Excel file"""',
            "    ",
            '    csv_files = sorted(glob.glob("*_RESULTS.csv"))',
            "    ",
            "    if not csv_files:",
            '        print("No *_RESULTS.csv files found in current directory.")',
            '        print("Make sure you\'ve run run_pysoc.sh first.")',
            "        sys.exit(1)",
            "    ",
            '    print(f"Found {len(csv_files)} result file(s)...")',
            "    ",
            '    output_file = "PySOC_Combined_Results.xlsx"',
            "    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:",
            "        ",
            "        for csv_file in csv_files:",
            "            try:",
            '                molecule_name = Path(csv_file).stem.replace("_RESULTS", "")',
            "                df = pd.read_csv(csv_file)",
            "                df.columns = df.columns.str.strip()",
            "                sheet_name = molecule_name[:31] if len(molecule_name) > 31 else molecule_name",
            "                base_sheet_name = sheet_name",
            "                counter = 1",
            "                while sheet_name in [ws.title for ws in writer.book.worksheets]:",
            '                    sheet_name = f"{base_sheet_name[:28]}_{counter}"',
            "                    counter += 1",
            "                df.to_excel(writer, sheet_name=sheet_name, index=False)",
            '                print(f"  Added {molecule_name} ({len(df)} rows)")',
            "            except Exception as e:",
            '                print(f"  Error processing {csv_file}: {e}")',
            "                continue",
            "        ",
            "        summary_data = []",
            "        for csv_file in csv_files:",
            "            try:",
            '                molecule_name = Path(csv_file).stem.replace("_RESULTS", "")',
            "                df = pd.read_csv(csv_file)",
            "                if 'RSS (cm-1)' in df.columns:",
            "                    max_soc = df['RSS (cm-1)'].max()",
            "                    mean_soc = df['RSS (cm-1)'].mean()",
            "                    n_transitions = len(df)",
            "                else:",
            '                    max_soc = "N/A"',
            '                    mean_soc = "N/A"',
            "                    n_transitions = len(df)",
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
        
        try:
            with open(analysis_script, 'w', encoding='utf-8', newline='\n') as f:
                f.write('\n'.join(analysis_script_lines))
            if os.name != 'nt':
                os.chmod(analysis_script, 0o755)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to write analysis script: {e}')
            return
        
        analysis_bash = output_dir / "combine_results.sh"
        analysis_bash_lines = [
            "#!/bin/bash",
            "# Combine PySOC results into Excel",
            "",
            "if ! command -v python3 &> /dev/null; then",
            "    echo 'Error: python3 not found. Please install Python 3.'",
            "    exit 1",
            "fi",
            "",
            "python3 -c 'import pandas, openpyxl' 2>/dev/null",
            "if [ $? -ne 0 ]; then",
            "    echo 'Installing required packages...'",
            "    pip3 install pandas openpyxl",
            "fi",
            "",
            "python3 combine_pysoc_results.py",
        ]
        
        write_exec(analysis_bash, analysis_bash_lines)
        
        preview_text = f"Generated PySOC scripts:\n\n"
        preview_text += f"1. run_pysoc.sh - Run PySOC calculations\n"
        preview_text += f"2. combine_pysoc_results.py - Combine results into Excel\n"
        preview_text += f"3. combine_results.sh - Run the combination script\n\n"
        preview_text += f"Found {len(log_files)} *_SOC.log file(s)\n"
        preview_text += f"Script location: {output_dir}\n"
        
        self.gaussian_preview.delete('1.0', 'end')
        self.gaussian_preview.insert('1.0', preview_text)
        self.gaussian_status.config(text=f'Generated PySOC script for {len(log_files)} log files')
        
        try:
            if os.name == 'nt':
                os.startfile(str(output_dir.resolve()))
        except:
            pass
        
        messagebox.showinfo('PySOC Scripts Generated', 
                          f'Generated 3 scripts in:\n{output_dir}\n\n'
                          f'Found {len(log_files)} *_SOC.log file(s)')
    
    def _gaussian_save_prefs(self):
        """Save Gaussian preferences"""
        # IMPORTANT: Collect config fresh to ensure we save current widget values
        cfg = self._gaussian_collect()
        cfg['lists'] = dict(
            func=self.cb_func.values if hasattr(self, 'cb_func') and self.cb_func and hasattr(self.cb_func, 'values') else [],
            basis=self.cb_basis.values if hasattr(self, 'cb_basis') and self.cb_basis and hasattr(self.cb_basis, 'values') else [],
            smodel=self.cb_smodel.values if hasattr(self, 'cb_smodel') and self.cb_smodel and hasattr(self.cb_smodel, 'values') else [],
            sname=self.cb_sname.values if hasattr(self, 'cb_sname') and self.cb_sname and hasattr(self.cb_sname, 'values') else [],
            sched=self.cb_sched.values if hasattr(self, 'cb_sched') and self.cb_sched and hasattr(self.cb_sched, 'values') else [],
            queue=self.cb_queue.values if hasattr(self, 'cb_queue') and self.cb_queue and hasattr(self.cb_queue, 'values') else []
        )
        try:
            with open(PREFS_FILE, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2)
            self.gaussian_status.config(text=f'Saved preferences → {PREFS_FILE}')
        except Exception as e:
            messagebox.showwarning('Save prefs', str(e))
    
    def _gaussian_load_prefs(self):
        """Load Gaussian preferences"""
        for k,v in GAUSSIAN_DEFAULTS.items():
            if k not in self.vars and not isinstance(v, list):
                if isinstance(v, bool): self.vars[k] = tk.BooleanVar(value=v)
                elif isinstance(v, int): self.vars[k] = tk.IntVar(value=v)
                else: self.vars[k] = tk.StringVar(value=str(v) if v is not None else '')
        
        if not PREFS_FILE.exists():
            return
        
        try:
            with open(PREFS_FILE, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            
            def _restore(cb, key: str):
                if cb and key in cfg.get('lists', {}):
                    vals = cfg['lists'][key]
                    if hasattr(cb, 'values'):
                        cb.values = vals
                        cb.combo['values'] = vals
            
            _restore(self.cb_func, 'func')
            _restore(self.cb_basis, 'basis')
            _restore(self.cb_smodel, 'smodel')
            _restore(self.cb_sname, 'sname')
            _restore(self.cb_sched, 'sched')
            _restore(self.cb_queue, 'queue')
            
            fields_to_always_default = {'INPUTS', 'OUT_DIR', 'NPROC', 'MEM', 'REMOVE_PREFIX', 'REMOVE_SUFFIX'}
            for key, var in self.vars.items():
                if key in cfg and not isinstance(cfg[key], list):
                    if key in fields_to_always_default:
                        continue
                    try:
                        var.set(cfg[key])
                    except Exception:
                        pass
            
            inline = set(cfg.get('INLINE_STEPS', []))
            for k,v in self.inline_vars.items():
                v.set(k in inline)
            
            if hasattr(self, 'multi_step_vars'):
                multi = set(cfg.get('MULTI_STEPS', []))
                if 'MULTI_STEPS' in cfg:
                    for k,v in self.multi_step_vars.items():
                        v.set(k in multi)
                else:
                    for v in self.multi_step_vars.values():
                        v.set(False)
            
            if hasattr(self, 'step_route_texts'):
                manual_routes = cfg.get('MANUAL_ROUTES', {})
                for step, route_text in self.step_route_texts.items():
                    if step in manual_routes:
                        route_text.delete('1.0', 'end')
                        route_text.insert('1.0', manual_routes[step])
            
            if hasattr(self, 'step_geom_source_vars'):
                geom_sources = cfg.get('GEOM_SOURCE', {})
                for step, var in self.step_geom_source_vars.items():
                    if step in geom_sources:
                        var.set(geom_sources[step])
            
            # Restore combo box values (CRITICAL - must restore before using)
            if 'FUNCTIONAL' in cfg and hasattr(self, 'cb_func') and self.cb_func:
                self.cb_func.set(cfg['FUNCTIONAL'])
            if 'BASIS' in cfg and hasattr(self, 'cb_basis') and self.cb_basis:
                self.cb_basis.set(cfg['BASIS'])
            if 'SOLVENT_MODEL' in cfg and hasattr(self, 'cb_smodel') and self.cb_smodel:
                self.cb_smodel.set(cfg['SOLVENT_MODEL'])
            if 'SOLVENT_NAME' in cfg and hasattr(self, 'cb_sname') and self.cb_sname:
                self.cb_sname.set(cfg['SOLVENT_NAME'])
            if 'SCHEDULER' in cfg and hasattr(self, 'cb_sched') and self.cb_sched:
                self.cb_sched.set(cfg['SCHEDULER'])
            if 'QUEUE' in cfg and hasattr(self, 'cb_queue') and self.cb_queue:
                self.cb_queue.set(cfg['QUEUE'])
            
            if hasattr(self, 'redundant_text'):
                redundant = cfg.get('REDUNDANT_COORDS', '')
                self.redundant_text.delete('1.0', 'end')
                self.redundant_text.insert('1.0', redundant)
            
            if 'INPUT_TYPE' in cfg:
                if 'INPUT_TYPE' in self.vars:
                    self.vars['INPUT_TYPE'].set(cfg['INPUT_TYPE'])
                    self._gaussian_on_input_type_change()
            if 'SMILES_INPUT' in cfg:
                if hasattr(self, 'gaussian_smiles_text'):
                    self.gaussian_smiles_text.delete('1.0', 'end')
                    self.gaussian_smiles_text.insert('1.0', cfg['SMILES_INPUT'])
            
            self._gaussian_update_visible_routes(immediate=True)
            self.gaussian_status.config(text=f'Loaded preferences from {PREFS_FILE}')
        except Exception as e:
            messagebox.showwarning('Load prefs', f'Could not load preferences: {e}')
    
    def _gaussian_reset_to_defaults(self):
        """Reset all Gaussian fields to default values"""
        if not messagebox.askyesno('Reset to Defaults', 'This will reset all fields to default values. Continue?'):
            return
        
        for key, var in self.vars.items():
            if key in GAUSSIAN_DEFAULTS:
                default_val = GAUSSIAN_DEFAULTS[key]
                try:
                    if isinstance(var, tk.BooleanVar):
                        var.set(bool(default_val))
                    elif isinstance(var, tk.IntVar):
                        var.set(int(default_val) if default_val is not None else 0)
                    else:
                        var.set(str(default_val) if default_val is not None else '')
                except Exception:
                    pass
        
        if hasattr(self, 'multi_step_vars'):
            for v in self.multi_step_vars.values():
                v.set(False)
        
        if hasattr(self, 'inline_vars'):
            for v in self.inline_vars.values():
                v.set(False)
        
        if hasattr(self, 'cb_func'):
            self.cb_func.set(GAUSSIAN_DEFAULTS['FUNCTIONAL'])
        if hasattr(self, 'cb_basis'):
            self.cb_basis.set(GAUSSIAN_DEFAULTS['BASIS'])
        if hasattr(self, 'cb_smodel'):
            self.cb_smodel.set(GAUSSIAN_DEFAULTS['SOLVENT_MODEL'])
        if hasattr(self, 'cb_sname'):
            self.cb_sname.set(GAUSSIAN_DEFAULTS['SOLVENT_NAME'])
        if hasattr(self, 'cb_sched'):
            self.cb_sched.set(GAUSSIAN_DEFAULTS['SCHEDULER'])
        if hasattr(self, 'cb_queue'):
            self.cb_queue.set(GAUSSIAN_DEFAULTS['QUEUE'])
        
        self._gaussian_on_mode_change()
        self._gaussian_update_mode_buttons()
        
        self.gaussian_status.config(text='Reset to defaults')
        messagebox.showinfo('Reset Complete', 'All fields have been reset to default values.')
    
    # ========== TICT Helper Methods ==========
    def _tict_browse_input_gaussian(self):
        """Browse for input .com file for TICT rotation (Gaussian)"""
        path = filedialog.askopenfilename(title='Select input .com file', filetypes=[('Gaussian COM files', '*.com'), ('All files', '*.*')])
        if path:
            self.tict_vars_gaussian['INPUT_FILE'].set(path)
            if not self.tict_vars_gaussian['OUTPUT_DIR'].get():
                self.tict_vars_gaussian['OUTPUT_DIR'].set(str(Path(path).parent))
    
    def _tict_browse_output_gaussian(self):
        """Browse for output directory for TICT rotation (Gaussian)"""
        path = filedialog.askdirectory(title='Select output directory for rotated geometries')
        if path:
            self.tict_vars_gaussian['OUTPUT_DIR'].set(path)
    
    def _tict_generate_gaussian(self):
        """Generate TICT rotated geometries for Gaussian"""
        if not TICT_AVAILABLE:
            messagebox.showerror('TICT Module Error', 'TICT rotation module is not available.')
            return
        
        self.tict_log_gaussian.delete('1.0', tk.END)
        self.tict_log_gaussian.insert('1.0', 'Starting TICT rotation generation...\n\n')
        
        input_file = self.tict_vars_gaussian['INPUT_FILE'].get().strip()
        output_dir = self.tict_vars_gaussian['OUTPUT_DIR'].get().strip()
        
        if not input_file or not os.path.exists(input_file):
            self.tict_log_gaussian.insert(tk.END, f'ERROR: Please select a valid input file.\n')
            return
        
        if not output_dir:
            self.tict_log_gaussian.insert(tk.END, 'ERROR: Please select an output directory.\n')
            return
        
        try:
            # TICT rotation mode (old style)
            from tict_rotation import generate_tict_rotations
            
            axis_str = self.tict_vars_gaussian['AXIS'].get().strip()
            branch_a_str = self.tict_vars_gaussian['BRANCH_A'].get().strip()
            branch_a_step = float(self.tict_vars_gaussian['BRANCH_A_STEP'].get().strip())
            branch_b_str = self.tict_vars_gaussian['BRANCH_B'].get().strip()
            branch_b_step = float(self.tict_vars_gaussian['BRANCH_B_STEP'].get().strip())
            num_steps = int(self.tict_vars_gaussian['NUM_STEPS'].get().strip())
            
            base_name = Path(input_file).stem
            tict_output_dir = os.path.join(output_dir, f"{base_name}_tict_rotations")
            
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
        except ValueError as e:
            self.tict_log_gaussian.insert(tk.END, f'ERROR: Invalid parameter value: {e}\n')
            return
        except Exception as e:
            self.tict_log_gaussian.insert(tk.END, f'ERROR: {str(e)}\n')
            messagebox.showerror('TICT Rotation Error', str(e))
            return
        
        if success:
            self.tict_log_gaussian.insert(tk.END, f'\n{message}\n')
            self.tict_log_gaussian.insert(tk.END, f'\n✅ Success! Rotated geometries saved to:\n  {tict_output_dir}\n')
            messagebox.showinfo('TICT Rotation Complete', f'Successfully generated {len(files_created)} rotated geometry files!')
        else:
            self.tict_log_gaussian.insert(tk.END, f'\n❌ ERROR: {message}\n')
            messagebox.showerror('TICT Rotation Error', message)
        
        self.tict_log_gaussian.see(tk.END)
    
    def _tict_browse_input_orca(self):
        """Browse for input .xyz or .inp file for TICT rotation (ORCA)"""
        path = filedialog.askopenfilename(title='Select input .xyz or .inp file', 
                                         filetypes=[('XYZ files', '*.xyz'), ('ORCA input files', '*.inp'), ('All files', '*.*')])
        if path:
            self.tict_vars_orca['INPUT_FILE'].set(path)
            if not self.tict_vars_orca['OUTPUT_DIR'].get():
                self.tict_vars_orca['OUTPUT_DIR'].set(str(Path(path).parent))
    
    def _tict_browse_output_orca(self):
        """Browse for output directory for TICT rotation (ORCA)"""
        path = filedialog.askdirectory(title='Select output directory for rotated geometries')
        if path:
            self.tict_vars_orca['OUTPUT_DIR'].set(path)
    
    def _tict_generate_orca(self):
        """Generate TICT rotated geometries for ORCA"""
        if not TICT_AVAILABLE:
            messagebox.showerror('TICT Module Error', 'TICT rotation module is not available.')
            return
        
        self.tict_log_orca.delete('1.0', tk.END)
        self.tict_log_orca.insert('1.0', 'Starting TICT rotation generation for ORCA...\n\n')
        
        input_file = self.tict_vars_orca['INPUT_FILE'].get().strip()
        output_dir = self.tict_vars_orca['OUTPUT_DIR'].get().strip()
        
        if not input_file or not os.path.exists(input_file):
            self.tict_log_orca.insert(tk.END, f'ERROR: Please select a valid input file.\n')
            return
        
        if not output_dir:
            self.tict_log_orca.insert(tk.END, 'ERROR: Please select an output directory.\n')
            return
        
        try:
            axis_str = self.tict_vars_orca['AXIS'].get().strip()
            branch_a_str = self.tict_vars_orca['BRANCH_A'].get().strip()
            branch_a_step = float(self.tict_vars_orca['BRANCH_A_STEP'].get().strip())
            branch_b_str = self.tict_vars_orca['BRANCH_B'].get().strip()
            branch_b_step = float(self.tict_vars_orca['BRANCH_B_STEP'].get().strip())
            num_steps = int(self.tict_vars_orca['NUM_STEPS'].get().strip())
        except ValueError as e:
            self.tict_log_orca.insert(tk.END, f'ERROR: Invalid parameter value: {e}\n')
            return
        
        base_name = Path(input_file).stem
        tict_output_dir = os.path.join(output_dir, f"{base_name}_tict_rotations")
        
        success, message, files_created = generate_tict_rotations(
            input_file=input_file,
            output_dir=tict_output_dir,
            axis_str=axis_str,
            branch_a_str=branch_a_str,
            branch_a_step_deg=branch_a_step,
            branch_b_str=branch_b_str,
            branch_b_step_deg=branch_b_step,
            num_steps=num_steps,
            file_format="orca"
        )
        
        if success:
            self.tict_log_orca.insert(tk.END, f'\n{message}\n')
            self.tict_log_orca.insert(tk.END, f'\n✅ Success! Rotated .xyz geometries saved to:\n  {tict_output_dir}\n')
            messagebox.showinfo('TICT Rotation Complete', f'Successfully generated {len(files_created)} rotated .xyz files for ORCA!')
        else:
            self.tict_log_orca.insert(tk.END, f'\n❌ ERROR: {message}\n')
            messagebox.showerror('TICT Rotation Error', message)
        
        self.tict_log_orca.see(tk.END)
    
    # ========== AI Assistant Helper Methods ==========
    def _save_gemini_key(self, api_key: str):
        """Save Gemini API key"""
        try:
            from ai_assistant import save_gemini_api_key
            success, error_msg = save_gemini_api_key(api_key)
            if success:
                messagebox.showinfo('Success', 'API key saved successfully!')
                # Update the StringVar to reflect the saved key (in case it was modified)
                if hasattr(self, 'gemini_api_key_var'):
                    self.gemini_api_key_var.set(api_key.strip())
                if hasattr(self, 'gemini_api_key_var_orca'):
                    self.gemini_api_key_var_orca.set(api_key.strip())
            else:
                error_text = error_msg if error_msg else 'Failed to save API key.'
                messagebox.showerror('Error', error_text)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to save API key: {str(e)}')
    
    def _send_ai_message(self, input_entry, chat_display, software: str):
        """Send message to AI assistant"""
        user_message = input_entry.get().strip()
        if not user_message:
            return
        
        # Clear input
        input_entry.delete(0, tk.END)
        
        # Add user message to chat
        chat_display.config(state=tk.NORMAL)
        chat_display.insert(tk.END, f'\n👤 You: {user_message}\n\n', 'user')
        chat_display.see(tk.END)
        chat_display.config(state=tk.DISABLED)
        chat_display.update()
        
        # Initialize AI assistant if needed
        if not hasattr(self, f'ai_assistant_{software}'):
            try:
                from ai_assistant import QuantumChemistryAssistant, load_gemini_api_key, OLLAMA_AVAILABLE
                # Prefer Ollama (free, no API key needed, no quota limits)
                api_key = load_gemini_api_key()  # Optional, only needed if Ollama not available
                
                try:
                    # Try Ollama first (free, no API key, unlimited)
                    setattr(self, f'ai_assistant_{software}', QuantumChemistryAssistant(api_key=api_key, software=software, use_ollama=True))
                    # Show a message that we're using Ollama
                    chat_display.config(state=tk.NORMAL)
                    if not chat_display.get('1.0', tk.END).strip():
                        chat_display.insert(tk.END, '✅ Using Ollama (free, local, unlimited)\n\n', 'assistant')
                        chat_display.config(state=tk.DISABLED)
                        chat_display.update()
                        chat_display.config(state=tk.NORMAL)
                except Exception as e:
                    if "Ollama" in str(e) or "ollama" in str(e).lower():
                        # Ollama not available, try Gemini
                        if not api_key:
                            chat_display.config(state=tk.NORMAL)
                            chat_display.insert(tk.END, '⚠️ Ollama not available. Install Ollama (free, recommended):\n'
                                                       'Windows: Download from https://ollama.com/download\n'
                                                       'Mac/Linux: curl -fsSL https://ollama.com/install.sh | sh\n'
                                                       'Then run: ollama pull llama3.1:8b\n\n'
                                                       'Or get Gemini key: https://makersuite.google.com/app/apikey\n\n', 'error')
                            chat_display.config(state=tk.DISABLED)
                            return
                        setattr(self, f'ai_assistant_{software}', QuantumChemistryAssistant(api_key=api_key, software=software, use_ollama=False))
                    else:
                        raise
            except Exception as e:
                chat_display.config(state=tk.NORMAL)
                chat_display.insert(tk.END, f'❌ Error initializing AI: {str(e)}\n\n', 'error')
                chat_display.config(state=tk.DISABLED)
                return
        
        # Get AI assistant
        assistant = getattr(self, f'ai_assistant_{software}')
        
        # Check if user wants to generate files
        if user_message.lower().strip() in ['generate files', 'generate', 'create files', 'go']:
            self._generate_files_from_ai_conversation(chat_display, software)
            return
        
        # Send message and get response
        try:
            response, success = assistant.send_message(user_message)
            
            chat_display.config(state=tk.NORMAL)
            if success:
                # Check if response contains generation marker
                if "GENERATE_FILES_START" in response and "GENERATE_FILES_END" in response:
                    # Extract config and generate files
                    config = assistant.extract_generation_config(response)
                    if config:
                        chat_display.insert(tk.END, f'🤖 AI Assistant: I have all the information. Generating files now...\n\n', 'assistant')
                        chat_display.config(state=tk.DISABLED)
                        chat_display.update()
                        self._generate_files_from_ai_config(config, chat_display, software)
                        return
                    else:
                        chat_display.insert(tk.END, f'🤖 AI Assistant: {response}\n\n', 'assistant')
                else:
                    chat_display.insert(tk.END, f'🤖 AI Assistant: {response}\n\n', 'assistant')
            else:
                # Check if it's a quota error suggesting Ollama
                if "quota exceeded" in response.lower() or "install ollama" in response.lower():
                    chat_display.insert(tk.END, f'⚠️ {response}\n\n', 'error')
                    # Try to reinitialize with Ollama if available
                    try:
                        from ai_assistant import OLLAMA_AVAILABLE
                        if OLLAMA_AVAILABLE:
                            chat_display.insert(tk.END, '🔄 Attempting to switch to Ollama (free, local)...\n\n', 'assistant')
                            chat_display.config(state=tk.DISABLED)
                            chat_display.update()
                            # Delete current assistant and recreate with Ollama
                            if hasattr(self, f'ai_assistant_{software}'):
                                delattr(self, f'ai_assistant_{software}')
                            from ai_assistant import QuantumChemistryAssistant
                            setattr(self, f'ai_assistant_{software}', QuantumChemistryAssistant(api_key=None, software=software, use_ollama=True))
                            # Retry the message
                            assistant = getattr(self, f'ai_assistant_{software}')
                            response, success = assistant.send_message(user_message)
                            chat_display.config(state=tk.NORMAL)
                            if success:
                                chat_display.insert(tk.END, f'✅ Switched to Ollama successfully!\n\n🤖 AI Assistant: {response}\n\n', 'assistant')
                            else:
                                chat_display.insert(tk.END, f'⚠️ {response}\n\n', 'error')
                    except Exception:
                        chat_display.insert(tk.END, f'❌ {response}\n\n', 'error')
                else:
                    chat_display.insert(tk.END, f'❌ {response}\n\n', 'error')
            chat_display.config(state=tk.DISABLED)
            chat_display.see(tk.END)
        except Exception as e:
            chat_display.config(state=tk.NORMAL)
            chat_display.insert(tk.END, f'❌ Error: {str(e)}\n\n', 'error')
            chat_display.config(state=tk.DISABLED)
    
    def _clear_ai_chat(self, chat_display):
        """Clear AI chat"""
        chat_display.config(state=tk.NORMAL)
        chat_display.delete('1.0', tk.END)
        software = 'gaussian' if hasattr(self, 'gaussian_chat_display') and chat_display == self.gaussian_chat_display else 'orca'
        chat_display.insert('1.0', f'🤖 AI Assistant: Chat cleared. How can I help you set up {software.upper()} calculations?\n\n', 'assistant')
        chat_display.config(state=tk.DISABLED)
        
        # Reset conversation if assistant exists
        if hasattr(self, f'ai_assistant_{software}'):
            getattr(self, f'ai_assistant_{software}').reset_conversation()
    
    def _generate_files_from_ai_conversation(self, chat_display, software: str):
        """Extract configuration from AI conversation and generate files"""
        if not hasattr(self, f'ai_assistant_{software}'):
            chat_display.config(state=tk.NORMAL)
            chat_display.insert(tk.END, '❌ Error: No active conversation. Please chat with the AI first.\n\n', 'error')
            chat_display.config(state=tk.DISABLED)
            return
        
        assistant = getattr(self, f'ai_assistant_{software}')
        
        # Ask AI to provide configuration in structured format
        chat_display.config(state=tk.NORMAL)
        chat_display.insert(tk.END, '\n👤 You: Please provide the configuration as JSON so I can generate the files.\n\n', 'user')
        chat_display.config(state=tk.DISABLED)
        chat_display.update()
        
        try:
            response, success = assistant.send_message("Please output the complete configuration as JSON wrapped in GENERATE_FILES_START and GENERATE_FILES_END markers. CRITICAL: You MUST include INPUT_FILE and OUTPUT_DIR fields - extract these from our conversation. Include all parameters: INPUT_FILE (REQUIRED), OUTPUT_DIR (REQUIRED), MODE, STEP, METHOD/BASIS (or FUNCTIONAL/BASIS for Gaussian), SOLVENT_MODEL, SOLVENT_NAME, CHARGE, MULT, NPROC/NPROCS, MEM/MAXCORE_MB, SCHEDULER, and any other relevant parameters. For TICT scans, also include: CALCULATION_TYPE, DIHEDRAL_ATOMS, SCAN_RANGE, NUM_STEPS.")
            
            if success:
                config = assistant.extract_generation_config(response)
                if config:
                    chat_display.config(state=tk.NORMAL)
                    chat_display.insert(tk.END, f'🤖 AI Assistant: Configuration extracted. Generating files...\n\n', 'assistant')
                    chat_display.config(state=tk.DISABLED)
                    chat_display.update()
                    self._generate_files_from_ai_config(config, chat_display, software)
                else:
                    chat_display.config(state=tk.NORMAL)
                    chat_display.insert(tk.END, f'🤖 AI Assistant: {response}\n\n', 'assistant')
                    chat_display.insert(tk.END, '⚠️ Could not extract configuration from response. Please ensure the AI provided a valid JSON configuration.\n\n', 'error')
                    chat_display.config(state=tk.DISABLED)
            else:
                chat_display.config(state=tk.NORMAL)
                chat_display.insert(tk.END, f'❌ Error: {response}\n\n', 'error')
                chat_display.config(state=tk.DISABLED)
        except Exception as e:
            chat_display.config(state=tk.NORMAL)
            chat_display.insert(tk.END, f'❌ Error: {str(e)}\n\n', 'error')
            chat_display.config(state=tk.DISABLED)
    
    def _generate_files_from_ai_config(self, config: Dict, chat_display, software: str):
        """Generate files from AI-extracted configuration"""
        try:
            from ai_file_generator import generate_files_from_ai_config
            
            success, message, generated_files = generate_files_from_ai_config(config, software)
            
            chat_display.config(state=tk.NORMAL)
            if success:
                chat_display.insert(tk.END, f'✅ {message}\n\n', 'assistant')
                chat_display.insert(tk.END, f'Generated {len(generated_files)} file(s).\n\n', 'assistant')
                
                # Open output directory
                output_dir = config.get('OUTPUT_DIR', '')
                if output_dir and os.path.exists(output_dir):
                    try:
                        if sys.platform == 'darwin':
                            subprocess.run(['open', output_dir])
                        elif sys.platform == 'win32':
                            os.startfile(output_dir)
                        else:
                            subprocess.run(['xdg-open', output_dir])
                    except:
                        pass
                
                messagebox.showinfo('Files Generated', message)
            else:
                chat_display.insert(tk.END, f'❌ Error: {message}\n\n', 'error')
                messagebox.showerror('Generation Error', message)
            chat_display.config(state=tk.DISABLED)
            chat_display.see(tk.END)
        except Exception as e:
            chat_display.config(state=tk.NORMAL)
            chat_display.insert(tk.END, f'❌ Error generating files: {str(e)}\n\n', 'error')
            chat_display.config(state=tk.DISABLED)
            messagebox.showerror('Error', f'Failed to generate files: {str(e)}')

if __name__ == "__main__":
    root = tk.Tk()
    app = QuantumStepsApp(root)
    root.mainloop()

