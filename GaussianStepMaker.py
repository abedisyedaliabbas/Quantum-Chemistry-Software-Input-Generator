#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Syed Ali Abbas Abedi

One-step OR Full-steps Gaussian generator 

Modes
-----
MODE = "single" -> generate one step (STEP = 1..7)
MODE = "full"   -> generate all steps 1..7 in one folder

File names
----------
<STEP><BASENAME>_<FUNCTIONAL>_<BASIS>_<solvtag>.com/.sh

"""

from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Tuple
import re, glob

# ===================== CONFIG =====================
MODE        = "full"            # "single" or "full"
STEP        = 4                 # used when MODE="single", 1..7
INPUTS      = "./*.com"         # glob or a folder path
OUT_DIR     = "Jobs"            # output folder

FUNCTIONAL  = "m062x"
BASIS       = "def2tzvp"

SOLVENT_MODEL = "SMD"           # "none","SMD","PCM","IEFPCM","CPCM"
SOLVENT_NAME  = "DMSO"          # ignored if SOLVENT_MODEL="none"

TD_NSTATES  = 3
TD_ROOT     = 1

POP_FULL    = False             # pop=(full,orbitals=2,threshorbitals=1)
DISPERSION  = False             # EmpiricalDispersion=GD3BJ

# Resource / scheduler
NPROC       = 32
MEM         = "64GB"
SCHEDULER   = "pbs"             # "pbs","slurm","local"
QUEUE       = "normal"
WALLTIME    = "24:00:00"
PROJECT     = "15002108"        # PBS -P
ACCOUNT     = ""                # SLURM -A

# FULL mode geometry control
# Steps you want to embed inline coords for (others will be linked):
INLINE_STEPS = []               # user selection (e.g., [4]); ignored for steps not allowed
# Only these steps may ever be inlined in FULL mode:
INLINE_ALLOWED = {1, 4}         # Step 1 is always inline; Step 4 optionally inline
# For steps 5–7, when you inline, choose the source step's geometry (1 or 4):
INLINE_SOURCE_5TO7 = 4          # 1 = input .com; 4 = “after step 4” (uses input coords as placeholder)

# Charge/multiplicity override (set both to ints to force; leave as None to parse from .com)
CHARGE      = None              # e.g., 0
MULT        = None              # e.g., 1
# ==================================================

# ---------------- utilities ----------------
def natural_key(s: str):
    parts = re.split(r'(\d+)', s.lower())
    return tuple(int(p) if p.isdigit() else p for p in parts)

def read_lines(p: Path) -> List[str]:
    return p.read_text(errors="ignore").splitlines()

def write_lines(p: Path, lines: Sequence[str]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="\n") as f:
        for L in lines:
            f.write(str(L).rstrip("\r\n") + "\n")

def find_geoms(pattern: str) -> List[Path]:
    path = Path(pattern)
    if path.exists() and path.is_dir():
        files = sorted(path.glob("*.com"), key=lambda x: natural_key(x.name))
    else:
        files = [Path(m) for m in glob.glob(pattern)]
        files = sorted([p for p in files if p.is_file() and p.suffix.lower()==".com"], key=lambda x: natural_key(x.name))
    return files

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

def build_scrf(model: str, solvent: str, tail: str = "") -> str:
    if model.lower() in ("", "none", "vac", "vacuum"): return ""
    parts = [model]
    if solvent: parts.append(f"solvent={solvent}")
    if tail:    parts.append(tail)
    return ", ".join(parts)

def route_line(method: str, basis: str, td: str = "", scrf: str = "", extras: str = "") -> str:
    td_part   = f" TD=({td})" if td else ""
    scrf_part = f" SCRF=({scrf})" if scrf else ""
    base = f"# {method}/{basis}{td_part}{scrf_part}"
    if extras: base += f" {extras}"
    return re.sub(r"\s+", " ", base).strip()

def cm_override(parsed_cm: str) -> str:
    if CHARGE is None or MULT is None:
        return parsed_cm
    return f"{int(CHARGE)} {int(MULT)}"

def make_com_inline(job: str, nproc: int, mem: str, route: str, title: str, cm: str, coords: List[str]) -> List[str]:
    return [f"%nprocshared={nproc}", f"%mem={mem}", f"%chk={job}.chk",
            route, "", title, "", cm, *coords, "", ""]

def make_com_linked(job: str, nproc: int, mem: str, oldchk: str, route: str, title: str, cm: str) -> List[str]:
    # linked uses oldchk + geom=check; only include charge/mult line
    return [f"%nprocshared={nproc}", f"%mem={mem}", f"%oldchk={oldchk}", f"%chk={job}.chk",
            route, "", title, "", cm, "", ""]

def pbs_script(job: str) -> List[str]:
    return [
        "#!/bin/bash",
        f"#PBS -q {QUEUE}",
        f"#PBS -N {job}",
        f"#PBS -l select=1:ncpus={NPROC}:mpiprocs={NPROC}:mem={MEM}",
        (f"#PBS -l walltime={WALLTIME}" if WALLTIME else "").strip(),
        (f"#PBS -P {PROJECT}" if PROJECT else "").strip(),
        f"#PBS -o {job}.o", f"#PBS -e {job}.e",
        "cd $PBS_O_WORKDIR",
        f"g16 < {job}.com > {job}.log",
    ]

def slurm_script(job: str) -> List[str]:
    return [
        "#!/bin/bash",
        f"#SBATCH -J {job}",
        f"#SBATCH -p {QUEUE}",
        "#SBATCH -N 1",
        f"#SBATCH --ntasks={NPROC}",
        f"#SBATCH --mem={MEM}",
        (f"#SBATCH -t {WALLTIME}" if WALLTIME else "").strip(),
        (f"#SBATCH -A {ACCOUNT}" if ACCOUNT else "").strip(),
        f"#SBATCH -o {job}.out", f"#SBATCH -e {job}.err",
        f"g16 < {job}.com > {job}.log",
    ]

def local_script(job: str) -> List[str]:
    return ["#!/bin/bash", f"g16 < {job}.com > {job}.log &"]

def write_sh(job: str) -> List[str]:
    if SCHEDULER == "pbs":   return pbs_script(job)
    if SCHEDULER == "slurm": return slurm_script(job)
    return local_script(job)

# ---------------- routes per step ----------------
TD_BLOCK = f"NStates={TD_NSTATES}, Root={TD_ROOT}"
POP_KW   = " pop=(full,orbitals=2,threshorbitals=1)" if POP_FULL else ""
DISP_KW  = " EmpiricalDispersion=GD3BJ" if DISPERSION else ""
SCRF     = build_scrf(SOLVENT_MODEL, SOLVENT_NAME)
SCRF_CLR = build_scrf(SOLVENT_MODEL, SOLVENT_NAME, "CorrectedLR")

def route_step1(): return route_line(FUNCTIONAL, BASIS, scrf=SCRF, extras=f"Opt Freq{POP_KW}{DISP_KW}")
def route_step2(): return route_line(FUNCTIONAL, BASIS, td=TD_BLOCK, scrf=SCRF, extras=f"{POP_KW}{DISP_KW}")
def route_step3(): return route_line(FUNCTIONAL, BASIS, td=TD_BLOCK, scrf=SCRF_CLR, extras=f"{POP_KW}{DISP_KW}")
def route_step4(): return route_line(FUNCTIONAL, BASIS, td=TD_BLOCK, scrf=SCRF, extras=f"Opt=CalcFC Freq{DISP_KW}")
def route_step5(): return route_line(FUNCTIONAL, BASIS, scrf=SCRF, extras=f"density{POP_KW}{DISP_KW}")
def route_step6(): return route_line(FUNCTIONAL, BASIS, td=TD_BLOCK, scrf=SCRF_CLR, extras=f"{DISP_KW}")
def route_step7(): return route_line(FUNCTIONAL, BASIS, td=TD_BLOCK, scrf=SCRF, extras=f"{POP_KW}{DISP_KW}")

# ---------------- generation ----------------
def generate_single(base: str, cm: str, coords: List[str], out_path: Path, solv_tag: str):
    prefix = f"{STEP:02d}"
    job = f"{prefix}{base}_{FUNCTIONAL}_{BASIS}_{solv_tag}"
    title_map = {
        1: f"Step1 GS Opt {FUNCTIONAL}/{BASIS}",
        2: f"Step2 Abs {FUNCTIONAL}/{BASIS}",
        3: f"Step3 Abs cLR {FUNCTIONAL}/{BASIS}",
        4: f"Step4 ES Opt {FUNCTIONAL}/{BASIS}",
        5: f"Step5 Density {FUNCTIONAL}/{BASIS}",
        6: f"Step6 ES cLR {FUNCTIONAL}/{BASIS}",
        7: f"Step7 De-excitation {FUNCTIONAL}/{BASIS}",
    }
    route = {1: route_step1, 2: route_step2, 3: route_step3, 4: route_step4,
             5: route_step5, 6: route_step6, 7: route_step7}[STEP]()
    cm_use = cm_override(cm)
    com = make_com_inline(job, NPROC, MEM, route, title_map[STEP], cm_use, coords)
    write_lines(out_path / f"{job}.com", com)
    write_lines(out_path / f"{job}.sh", write_sh(job))
    return [job]

def generate_full(base: str, cm_in: str, coords_in: List[str], out_path: Path, solv_tag: str):
    jobs = []
    # names
    J = lambda k: f"{k:02d}{base}_{FUNCTIONAL}_{BASIS}_{solv_tag}"
    j01,j02,j03,j04,j05,j06,j07 = (J(i) for i in range(1,8))

    # Step 1: always inline from input
    cm_use = cm_override(cm_in)
    com01 = make_com_inline(j01, NPROC, MEM, route_step1(), f"Step1 GS Opt {FUNCTIONAL}/{BASIS}", cm_use, coords_in)
    write_lines(out_path / f"{j01}.com", com01); write_lines(out_path / f"{j01}.sh", write_sh(j01)); jobs.append(j01)

    # Helpers to decide inline/linked and which coords to use when inlining
    def inline_for(step: int) -> bool:
        # Step 1 is always inline; Step 4 optionally inline via INLINE_STEPS.
        if step == 1:
            return True
        return (step in INLINE_ALLOWED) and (step in INLINE_STEPS)

    def coords_for(step: int) -> Tuple[str, List[str]]:
        # 2–4 get Step1 geometry if inlined
        if step in (2,3,4):
            return cm_use, coords_in
        # 5–7 get geometry from INLINE_SOURCE_5TO7 (we inline input coords as starter)
        if step in (5,6,7):
            return cm_use, coords_in
        return cm_use, coords_in

    # Step 2
    if inline_for(2):
        cm2, c2 = coords_for(2)
        com02 = make_com_inline(j02, NPROC, MEM, route_step2(), f"Step2 Abs {FUNCTIONAL}/{BASIS}", cm2, c2)
    else:
        r2 = route_step2() + " geom=check guess=read"
        com02 = make_com_linked(j02, NPROC, MEM, f"{j01}.chk", r2, f"Step2 Abs {FUNCTIONAL}/{BASIS}", cm_use)
    write_lines(out_path / f"{j02}.com", com02); write_lines(out_path / f"{j02}.sh", write_sh(j02)); jobs.append(j02)

    # Step 3
    if inline_for(3):
        cm3, c3 = coords_for(3)
        com03 = make_com_inline(j03, NPROC, MEM, route_step3(), f"Step3 Abs cLR {FUNCTIONAL}/{BASIS}", cm3, c3)
    else:
        r3 = route_step3() + " geom=check guess=read"
        com03 = make_com_linked(j03, NPROC, MEM, f"{j01}.chk", r3, f"Step3 Abs cLR {FUNCTIONAL}/{BASIS}", cm_use)
    write_lines(out_path / f"{j03}.com", com03); write_lines(out_path / f"{j03}.sh", write_sh(j03)); jobs.append(j03)

    # Step 4
    if inline_for(4):
        cm4, c4 = coords_for(4)
        com04 = make_com_inline(j04, NPROC, MEM, route_step4(), f"Step4 ES Opt {FUNCTIONAL}/{BASIS}", cm4, c4)
    else:
        r4 = route_step4() + " geom=check guess=read"
        com04 = make_com_linked(j04, NPROC, MEM, f"{j01}.chk", r4, f"Step4 ES Opt {FUNCTIONAL}/{BASIS}", cm_use)
    write_lines(out_path / f"{j04}.com", com04); write_lines(out_path / f"{j04}.sh", write_sh(j04)); jobs.append(j04)

    # Step 5
    if inline_for(5):
        cm5, c5 = coords_for(5)  # from INLINE_SOURCE_5TO7 (we inline input coords as starter)
        com05 = make_com_inline(j05, NPROC, MEM, route_step5(), f"Step5 Density {FUNCTIONAL}/{BASIS}", cm5, c5)
    else:
        r5 = route_step5() + " geom=check guess=read"
        com05 = make_com_linked(j05, NPROC, MEM, f"{j04}.chk", r5, f"Step5 Density {FUNCTIONAL}/{BASIS}", cm_use)
    write_lines(out_path / f"{j05}.com", com05); write_lines(out_path / f"{j05}.sh", write_sh(j05)); jobs.append(j05)

    # Step 6
    if inline_for(6):
        cm6, c6 = coords_for(6)
        com06 = make_com_inline(j06, NPROC, MEM, route_step6(), f"Step6 ES cLR {FUNCTIONAL}/{BASIS}", cm6, c6)
    else:
        r6 = route_step6() + " geom=check guess=read"
        com06 = make_com_linked(j06, NPROC, MEM, f"{j04}.chk", r6, f"Step6 ES cLR {FUNCTIONAL}/{BASIS}", cm_use)
    write_lines(out_path / f"{j06}.com", com06); write_lines(out_path / f"{j06}.sh", write_sh(j06)); jobs.append(j06)

    # Step 7
    if inline_for(7):
        cm7, c7 = coords_for(7)
        com07 = make_com_inline(j07, NPROC, MEM, route_step7(), f"Step7 De-excitation {FUNCTIONAL}/{BASIS}", cm7, c7)
    else:
        r7 = route_step7() + " geom=check guess=read"
        com07 = make_com_linked(j07, NPROC, MEM, f"{j04}.chk", r7, f"Step7 De-excitation {FUNCTIONAL}/{BASIS}", cm_use)
    write_lines(out_path / f"{j07}.com", com07); write_lines(out_path / f"{j07}.sh", write_sh(j07)); jobs.append(j07)

    return jobs

# ---------------- main ----------------
def main():
    files = find_geoms(INPUTS)
    if not files: raise SystemExit("No .com files found.")
    out_path = Path(OUT_DIR); out_path.mkdir(exist_ok=True)

    solv_tag = "vac" if SOLVENT_MODEL.lower() in ("none","vac","vacuum") else SOLVENT_NAME.lower()
    submit_lines = []

    for p in files:
        base = p.stem
        parsed_cm, coords = extract_cm_coords(read_lines(p))
        parsed_cm = cm_override(parsed_cm)  # apply CM override if set
        if MODE == "single":
            jobs = generate_single(base, parsed_cm, coords, out_path, solv_tag)
        else:
            jobs = generate_full(base, parsed_cm, coords, out_path, solv_tag)
        for j in jobs:
            submit_lines.append(("qsub " if SCHEDULER=="pbs" else "sbatch " if SCHEDULER=="slurm" else "bash ") + f"{j}.sh")

    write_lines(out_path / "submit_all.sh", submit_lines)
    print(f"Done. Mode={MODE} ({'STEP '+str(STEP) if MODE=='single' else 'FULL 1–7'}), {len(files)} molecule(s) -> {out_path.resolve()}")

if __name__ == "__main__":
    main()
