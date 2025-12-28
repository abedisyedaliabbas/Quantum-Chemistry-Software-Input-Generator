"""
AI File Generator Module
Generates quantum chemistry input files based on AI-extracted configuration
"""
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import os
import numpy as np
import glob

# Try to import TICT rotation module
try:
    from tict_rotation import generate_tict_rotations, load_gaussian_geometry, load_orca_geometry
    TICT_AVAILABLE = True
except ImportError:
    TICT_AVAILABLE = False


def calculate_dihedral_angle(coords: np.ndarray, atom1: int, atom2: int, atom3: int, atom4: int) -> float:
    """
    Calculate the dihedral angle between four atoms (in degrees).
    
    Args:
        coords: Numpy array of shape (n_atoms, 3) with atomic coordinates
        atom1, atom2, atom3, atom4: 0-based atom indices
    
    Returns:
        Dihedral angle in degrees
    """
    # Convert to 0-based indexing if needed (assuming input is 1-based)
    if atom1 > 0:
        atom1, atom2, atom3, atom4 = atom1 - 1, atom2 - 1, atom3 - 1, atom4 - 1
    
    # Get coordinates
    p1 = coords[atom1]
    p2 = coords[atom2]
    p3 = coords[atom3]
    p4 = coords[atom4]
    
    # Calculate vectors
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3
    
    # Normalize v2
    v2_norm = np.linalg.norm(v2)
    if v2_norm < 1e-10:
        return 0.0
    v2_unit = v2 / v2_norm
    
    # Project v1 and v3 onto plane perpendicular to v2
    v1_proj = v1 - np.dot(v1, v2_unit) * v2_unit
    v3_proj = v3 - np.dot(v3, v2_unit) * v2_unit
    
    # Calculate angle between projections
    v1_norm = np.linalg.norm(v1_proj)
    v3_norm = np.linalg.norm(v3_proj)
    
    if v1_norm < 1e-10 or v3_norm < 1e-10:
        return 0.0
    
    # Calculate cross product to determine sign
    cross = np.cross(v1_proj, v3_proj)
    sign = np.sign(np.dot(cross, v2_unit))
    
    # Calculate angle using atan2 for better sign handling
    # This gives us the angle in the range -pi to pi
    sin_angle = np.linalg.norm(cross) * sign / (v1_norm * v3_norm)
    cos_angle = np.dot(v1_proj, v3_proj) / (v1_norm * v3_norm)
    angle_rad = np.arctan2(sin_angle, cos_angle)
    
    # Convert to degrees (range: -180 to 180)
    angle_deg = np.degrees(angle_rad)
    
    # Normalize to 0-360 range
    # The issue: angles near 180 (like 179) might be calculated as -1, which becomes 359
    # Solution: if the normalized angle is > 270, it might actually be closer to 180
    if angle_deg < 0:
        angle_deg += 360
    
    # Fix the case where small negative angles become large positive angles
    # If angle is > 270, check if subtracting 180 gives a more reasonable value
    # Example: 359.22 should become 179.22 (359.22 - 180 = 179.22)
    if angle_deg > 270:
        # This is likely a small negative angle that was normalized incorrectly
        # Subtract 180 to get the equivalent angle in the 0-180 range
        angle_deg = angle_deg - 180
    
    return angle_deg

# Try to import GUI generation functions
try:
    from gaussian_steps_gui import (
        GAUSSIAN_DEFAULTS,
        generate_single, generate_full,
        find_geoms, remove_prefix_suffix, extract_cm_coords,
        parse_gaussian_log, parse_smiles_line, smiles_to_coords,
        read_lines, write_lines, write_exec, add_redundant_coords,
        cm_override
    )
    GAUSSIAN_AVAILABLE = True
except ImportError:
    GAUSSIAN_AVAILABLE = False

# ORCA functions are in quantum_steps_gui but we avoid circular import by importing only when needed
ORCA_AVAILABLE = False
ORCA_MODULE = None
try:
    import importlib
    ORCA_MODULE = importlib.import_module('quantum_steps_gui')
    ORCA_AVAILABLE = True
except ImportError:
    ORCA_AVAILABLE = False


def generate_files_from_ai_config(config: Dict, software: str = "gaussian") -> Tuple[bool, str, List[str]]:
    """
    Generate quantum chemistry input files based on AI-extracted configuration
    
    Args:
        config: Dictionary containing all necessary configuration
        software: "gaussian" or "orca"
    
    Returns:
        Tuple of (success: bool, message: str, generated_files: List[str])
    """
    # Check if this is a TICT/torsional scan request
    calc_type = config.get('CALCULATION_TYPE', '').lower()
    is_tict = (
        'tict' in calc_type or 
        'torsional' in calc_type or 
        'dihedral' in calc_type or
        'scan' in calc_type or
        config.get('DIHEDRAL_ATOMS') is not None or
        config.get('SCAN_RANGE') is not None
    )
    
    if is_tict and TICT_AVAILABLE:
        return _generate_tict_files(config, software)
    
    if software.lower() == "gaussian":
        if not GAUSSIAN_AVAILABLE:
            return False, "Gaussian modules not available", []
        return _generate_gaussian_files(config)
    elif software.lower() == "orca":
        if not ORCA_AVAILABLE:
            return False, "ORCA modules not available", []
        return _generate_orca_files(config)
    else:
        return False, f"Unknown software: {software}", []


def _generate_gaussian_files(config: Dict) -> Tuple[bool, str, List[str]]:
    """Generate Gaussian input files"""
    try:
        # Extract required parameters
        input_path = config.get('INPUT_FILE', '').strip()
        output_dir = config.get('OUTPUT_DIR', '').strip()
        input_type = config.get('INPUT_TYPE', 'com')  # com, log, smiles
        
        if not input_path:
            return False, "Input file path is required", []
        if not output_dir:
            return False, "Output directory is required", []
        
        # Set defaults
        mode = config.get('MODE', 'single')
        step = int(config.get('STEP', 1))
        functional = config.get('FUNCTIONAL', GAUSSIAN_DEFAULTS.get('FUNCTIONAL', 'm062x'))
        basis = config.get('BASIS', GAUSSIAN_DEFAULTS.get('BASIS', 'def2TZVP'))
        solvent_model = config.get('SOLVENT_MODEL', GAUSSIAN_DEFAULTS.get('SOLVENT_MODEL', 'none'))
        solvent_name = config.get('SOLVENT_NAME', GAUSSIAN_DEFAULTS.get('SOLVENT_NAME', 'water'))
        charge = config.get('CHARGE')
        mult = config.get('MULT')
        nproc = int(config.get('NPROC', 8))
        mem = config.get('MEM', '8GB')
        scheduler = config.get('SCHEDULER', 'local')
        queue = config.get('QUEUE', 'normal')
        walltime = config.get('WALLTIME', '24:00:00')
        project = config.get('PROJECT', '')
        account = config.get('ACCOUNT', '')
        td_nstates = int(config.get('TD_NSTATES', 3))
        td_root = int(config.get('TD_ROOT', 1))
        state_type = config.get('STATE_TYPE', 'singlet')
        
        # Build config dict for generation functions
        cfg = {
            'MODE': mode,
            'STEP': step,
            'INPUT_TYPE': input_type,
            'INPUTS': input_path,
            'SMILES_INPUT': config.get('SMILES_INPUT', ''),
            'OUT_DIR': output_dir,
            'FUNCTIONAL': functional,
            'BASIS': basis,
            'SOLVENT_MODEL': solvent_model,
            'SOLVENT_NAME': solvent_name,
            'CHARGE': charge,
            'MULT': mult,
            'NPROC': nproc,
            'MEM': mem,
            'SCHEDULER': scheduler,
            'QUEUE': queue,
            'WALLTIME': walltime,
            'PROJECT': project,
            'ACCOUNT': account,
            'TD_NSTATES': td_nstates,
            'TD_ROOT': td_root,
            'STATE_TYPE': state_type,
            'POP_FULL': config.get('POP_FULL', False),
            'DISPERSION': config.get('DISPERSION', False),
            'INLINE_STEPS': config.get('INLINE_STEPS', []),
            'MULTI_STEPS': config.get('MULTI_STEPS', []),
            'REMOVE_PREFIX': config.get('REMOVE_PREFIX', ''),
            'REMOVE_SUFFIX': config.get('REMOVE_SUFFIX', ''),
            'REDUNDANT_COORDS': config.get('REDUNDANT_COORDS', ''),
            'GEOM_SOURCE': config.get('GEOM_SOURCE', {}),
            'MANUAL_ROUTES': config.get('MANUAL_ROUTES', {}),
            'SOC_ENABLE': config.get('SOC_ENABLE', False),
        }
        
        # Process input files
        files_data = []
        if input_type == 'smiles':
            smiles_text = cfg['SMILES_INPUT']
            if not smiles_text:
                return False, "SMILES input is required when INPUT_TYPE is 'smiles'", []
            
            lines = smiles_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                name, smiles = parse_smiles_line(line)
                if smiles:
                    try:
                        cm, coords, base = smiles_to_coords(smiles, charge or 0, mult or 1, name)
                        cm = cm_override(cm, charge, mult)
                        files_data.append((base, cm, coords))
                    except Exception as e:
                        return False, f"Error processing SMILES '{smiles}': {str(e)}", []
        elif input_type == 'log':
            files = find_geoms(input_path, input_type='log')
            if not files:
                return False, f"No .log files found: {input_path}", []
            
            for p in files:
                try:
                    cm, coords, base = parse_gaussian_log(p)
                    cm = cm_override(cm, charge, mult)
                    if cfg['REMOVE_PREFIX'] or cfg['REMOVE_SUFFIX']:
                        cleaned = remove_prefix_suffix(p.name, cfg['REMOVE_PREFIX'], cfg['REMOVE_SUFFIX'])
                        base = cleaned.rsplit('.', 1)[0] if '.' in cleaned else cleaned
                    files_data.append((base, cm, coords))
                except Exception as e:
                    return False, f"Error parsing log file {p.name}: {str(e)}", []
        else:  # com
            files = find_geoms(input_path, input_type='com')
            if not files:
                return False, f"No .com files found: {input_path}", []
            
            for p in files:
                base = p.stem
                if cfg['REMOVE_PREFIX'] or cfg['REMOVE_SUFFIX']:
                    base = remove_prefix_suffix(p.name, cfg['REMOVE_PREFIX'], cfg['REMOVE_SUFFIX']).replace('.com', '')
                cm, coords = extract_cm_coords(read_lines(p))
                cm = cm_override(cm, charge, mult)
                files_data.append((base, cm, coords))
        
        if not files_data:
            return False, "No valid input files to process", []
        
        # Create output directory
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Generate files
        submit_all = []
        submit_by_step = {i: [] for i in range(1, 8)}
        formchk_by_step = {i: [] for i in range(1, 8)}
        
        for base, cm, coords in files_data:
            if mode == 'full':
                jobs = generate_full(base, cm, coords, out_path, cfg)
            else:
                jobs = generate_single(base, cm, coords, out_path, cfg)
            
            for j in jobs:
                try:
                    step_num = int(Path(j).name[:2])
                except:
                    step_num = None
                
                sched_cmd = ("qsub " if scheduler == "pbs" else ("sbatch " if scheduler == "slurm" else "bash ")) + f"{j}.sh"
                submit_all.append(sched_cmd)
                if step_num and 1 <= step_num <= 7:
                    submit_by_step[step_num].append(sched_cmd)
                    formchk_by_step[step_num].append(f"formchk {j}.chk")
        
        # Write helper scripts
        for k in range(1, 8):
            if submit_by_step[k]:
                write_exec(out_path / f'{k:02d}sub.sh', submit_by_step[k])
            if formchk_by_step[k]:
                write_exec(out_path / f'{k:02d}formchk.sh', formchk_by_step[k])
        
        write_exec(out_path / 'submit_all.sh', submit_all)
        
        generated_files = submit_all
        msg = f"Successfully generated {len(generated_files)} jobs for {len(files_data)} input(s) → {out_path.resolve()}"
        
        return True, msg, generated_files
        
    except Exception as e:
        return False, f"Error generating files: {str(e)}", []


def _find_input_files(pattern: str, file_format: str = "gaussian") -> List[Path]:
    """
    Find input files based on pattern (supports single files, folders, and glob patterns).
    
    Args:
        pattern: File path, folder path, or glob pattern
        file_format: "gaussian" or "orca"
    
    Returns:
        List of file paths
    """
    extension = ".com" if file_format == "gaussian" else (".xyz", ".inp")
    path = Path(pattern)
    
    files = []
    
    # Check if it's a directory
    if path.exists() and path.is_dir():
        if isinstance(extension, tuple):
            for ext in extension:
                files.extend(path.glob(f"*{ext}"))
        else:
            files.extend(path.glob(f"*{extension}"))
        files = sorted(files)
    # Check if it's a single file
    elif path.exists() and path.is_file():
        files = [path]
    else:
        # Try glob pattern
        if isinstance(extension, tuple):
            for ext in extension:
                files.extend([Path(p) for p in glob.glob(pattern + ext)])
                files.extend([Path(p) for p in glob.glob(pattern + f"*{ext}")])
        else:
            files.extend([Path(p) for p in glob.glob(pattern + extension)])
            files.extend([Path(p) for p in glob.glob(pattern + f"*{extension}")])
        files = sorted(set(files))  # Remove duplicates
    
    return [f for f in files if f.is_file()]


def _calculate_current_dihedral(input_file: str, dihedral_atoms: List[int], file_format: str) -> Optional[float]:
    """Calculate the current dihedral angle from an input file"""
    try:
        # Load geometry
        if file_format == "gaussian":
            header, title, cm, symbols, coords = load_gaussian_geometry(input_file)
            if coords is None:
                return None
        else:  # orca
            cm, symbols, coords = load_orca_geometry(input_file)
            if coords is None:
                return None
        
        # Calculate dihedral angle (atoms are 1-based, function expects 0-based)
        a1, a2, a3, a4 = dihedral_atoms
        angle = calculate_dihedral_angle(coords, a1, a2, a3, a4)
        return angle
    except Exception:
        return None


def _generate_tict_files(config: Dict, software: str) -> Tuple[bool, str, List[str]]:
    """Generate TICT/torsional scan files using the TICT rotation module"""
    try:
        input_path = config.get('INPUT_FILE', '').strip()
        output_dir = config.get('OUTPUT_DIR', '').strip()
        
        if not input_path:
            # Provide helpful error message with what's in config
            missing_fields = []
            if not input_path:
                missing_fields.append("INPUT_FILE")
            if not output_dir:
                missing_fields.append("OUTPUT_DIR")
            config_summary = ", ".join([f"{k}: {v}" for k, v in config.items() if k in ['CALCULATION_TYPE', 'DIHEDRAL_ATOMS', 'SCAN_RANGE', 'NUM_STEPS']])
            return False, f"Missing required fields for TICT scan: {', '.join(missing_fields)}. Please ensure the AI assistant includes INPUT_FILE and OUTPUT_DIR in the configuration. Current config: {config_summary}", []
        if not output_dir:
            return False, "OUTPUT_DIR is required for TICT scan. Please provide the output directory path.", []
        
        # Determine file format
        file_format = "gaussian" if software.lower() == "gaussian" else "orca"
        
        # Find input files (supports single file, folder, or glob pattern)
        input_files = _find_input_files(input_path, file_format)
        if not input_files:
            ext = ".com" if file_format == "gaussian" else ".xyz/.inp"
            return False, f"No {ext} files found matching: {input_path}", []
        
        # For TICT rotation, we need complete parameters (AXIS, BRANCH_A, BRANCH_B)
        # The old implementation requires manual specification of branches
        # Check if we have the required TICT parameters
        axis_str = config.get('AXIS', '').strip()
        branch_a_str = config.get('BRANCH_A', '').strip()
        branch_a_step = config.get('BRANCH_A_STEP', 0.0)
        branch_b_str = config.get('BRANCH_B', '').strip()
        branch_b_step = config.get('BRANCH_B_STEP', 0.0)
        num_steps_tict = int(config.get('NUM_STEPS', 10))
        
        # If we only have dihedral atoms but not full TICT params, we can't proceed
        if not axis_str or not branch_b_str:
            return False, "TICT rotation requires manual specification of rotation axis and branch atoms. Please use the GUI TICT Rotation tab with advanced TICT mode, or ask the AI assistant to provide complete TICT parameters: AXIS (e.g., '3,10'), BRANCH_A (e.g., '11,18,22'), BRANCH_A_STEP, BRANCH_B (e.g., '12-13,19-21'), BRANCH_B_STEP, and NUM_STEPS.", []
        
        # Import TICT rotation function
        from tict_rotation import generate_tict_rotations
        
        # Create output directory
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        all_generated_files = []
        processed_count = 0
        
        # Process each input file
        for input_file in input_files:
            try:
                # Create subdirectory for this input file
                base_name = Path(input_file).stem
                tict_output_dir = out_path / f"{base_name}_tict_rotations"
                tict_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Convert step values to float
                try:
                    branch_a_step_deg = float(branch_a_step)
                except (ValueError, TypeError):
                    branch_a_step_deg = 0.0
                
                try:
                    branch_b_step_deg = float(branch_b_step)
                except (ValueError, TypeError):
                    branch_b_step_deg = 0.0
                
                num_steps_int = num_steps_tict
                
                # Use advanced TICT rotation method (old working method)
                success, message, files_created = generate_tict_rotations(
                    input_file=str(input_file),
                    output_dir=str(tict_output_dir),
                    axis_str=axis_str,
                    branch_a_str=branch_a_str,
                    branch_a_step_deg=branch_a_step_deg,
                    branch_b_str=branch_b_str,
                    branch_b_step_deg=branch_b_step_deg,
                    num_steps=num_steps_int,
                    file_format=file_format
                )
                
                if success:
                    all_generated_files.extend([str(f) for f in files_created])
                    processed_count += 1
                else:
                    # Continue with other files even if one fails
                    continue
                    
            except Exception as e:
                # Continue with other files even if one fails
                continue
        
        if processed_count == 0:
            return False, f"Failed to process any files from: {input_path}", []
        
        msg = f"Successfully generated TICT rotation files for {processed_count} input(s) → {out_path.resolve()}"
        
        return True, msg, all_generated_files
            
    except Exception as e:
        return False, f"Error generating TICT files: {str(e)}", []


def _generate_orca_files(config: Dict) -> Tuple[bool, str, List[str]]:
    """Generate ORCA input files"""
    if not ORCA_AVAILABLE or ORCA_MODULE is None:
        return False, "ORCA modules not available", []
    
    try:
        # Import ORCA functions from the module
        ORCA_DEFAULTS = ORCA_MODULE.ORCA_DEFAULTS
        orca_build_step = ORCA_MODULE.orca_build_step
        orca_jobname = ORCA_MODULE.orca_jobname
        orca_write_sh = ORCA_MODULE.orca_write_sh
        orca_find_inputs = ORCA_MODULE.orca_find_inputs
        orca_extract_geom = ORCA_MODULE.orca_extract_geom
        # Use write_lines and write_exec from gaussian_steps_gui (they're the same)
        
        # Extract required parameters
        input_path = config.get('INPUT_FILE', '').strip()
        output_dir = config.get('OUTPUT_DIR', '').strip()
        input_type = config.get('INPUT_TYPE', 'xyz')  # xyz, com, log
        
        if not input_path:
            return False, "Input file path is required", []
        if not output_dir:
            return False, "Output directory is required", []
        
        # Set defaults
        mode = config.get('MODE', 'single')
        step = int(config.get('STEP', 4))
        method = config.get('METHOD', ORCA_DEFAULTS.get('METHOD', 'm06-2x'))
        basis = config.get('BASIS', ORCA_DEFAULTS.get('BASIS', 'def2-TZVP'))
        solvent_model = config.get('SOLVENT_MODEL', ORCA_DEFAULTS.get('SOLVENT_MODEL', 'none'))
        solvent_name = config.get('SOLVENT_NAME', ORCA_DEFAULTS.get('SOLVENT_NAME', 'DMSO'))
        charge = config.get('CHARGE')
        mult = config.get('MULT')
        nprocs = int(config.get('NPROCS', 8))
        maxcore_mb = int(config.get('MAXCORE_MB', 4000))
        orca_path = config.get('ORCA_PATH', '/path/to/orca')
        scheduler = config.get('SCHEDULER', 'pbs')
        queue = config.get('QUEUE', 'normal')
        walltime = config.get('WALLTIME', '24:00:00')
        project = config.get('PROJECT', '')
        account = config.get('ACCOUNT', '')
        td_nroots = int(config.get('TD_NROOTS', 6))
        td_iroot = int(config.get('TD_IROOT', 1))
        td_tda = config.get('TD_TDA', True)
        follow_iroot = config.get('FOLLOW_IROOT', True)
        epsilon = config.get('EPSILON')
        custom_keywords = config.get('CUSTOM_KEYWORDS', '')
        custom_block = config.get('CUSTOM_BLOCK', '')
        
        # Build config dict
        cfg = {
            'MODE': mode,
            'STEP': step,
            'INPUT_TYPE': input_type,
            'INPUTS': input_path,
            'OUT_DIR': output_dir,
            'METHOD': method,
            'BASIS': basis,
            'SOLVENT_MODEL': solvent_model,
            'SOLVENT_NAME': solvent_name,
            'EPSILON': epsilon,
            'CHARGE': charge,
            'MULT': mult,
            'NPROCS': nprocs,
            'MAXCORE_MB': maxcore_mb,
            'ORCA_PATH': orca_path,
            'SCHEDULER': scheduler,
            'QUEUE': queue,
            'WALLTIME': walltime,
            'PROJECT': project,
            'ACCOUNT': account,
            'TD_NROOTS': td_nroots,
            'TD_IROOT': td_iroot,
            'TD_TDA': td_tda,
            'FOLLOW_IROOT': follow_iroot,
            'CUSTOM_KEYWORDS': custom_keywords,
            'CUSTOM_BLOCK': custom_block,
            'REMOVE_PREFIX': config.get('REMOVE_PREFIX', ''),
            'REMOVE_SUFFIX': config.get('REMOVE_SUFFIX', ''),
        }
        
        # Process input files
        files = orca_find_inputs(input_path, input_type)
        if not files:
            return False, f"No input files found: {input_path}", []
        
        files_data = []
        for p in files:
            try:
                cm, coords = orca_extract_geom(p, input_type)
                base = p.stem
                if cfg['REMOVE_PREFIX'] or cfg['REMOVE_SUFFIX']:
                    remove_prefix_suffix = ORCA_MODULE.remove_prefix_suffix if hasattr(ORCA_MODULE, 'remove_prefix_suffix') else lambda n, p, s: n
                    cleaned = remove_prefix_suffix(p.name, cfg['REMOVE_PREFIX'], cfg['REMOVE_SUFFIX'])
                    base = cleaned.rsplit('.', 1)[0] if '.' in cleaned else cleaned
                files_data.append((base, cm, coords))
            except Exception as e:
                return False, f"Error extracting geometry from {p.name}: {str(e)}", []
        
        if not files_data:
            return False, "No valid input files to process", []
        
        # Create output directory
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Determine which steps to generate
        if mode == 'full':
            chosen_steps = [1, 2, 4, 7, 9]
        elif mode == 'multiple':
            chosen_steps = config.get('MULTI_STEPS', [step])
        else:
            chosen_steps = [step]
        
        # Generate files
        submit_lines = []
        generated = []
        for base_name, cm, coords in files_data:
            for k in chosen_steps:
                job = orca_jobname(k, base_name, cfg['METHOD'], cfg['BASIS'], 
                                 cfg['SOLVENT_MODEL'], cfg['SOLVENT_NAME'])
                inp = orca_build_step(k, cfg, cm, coords)
                # Replace JOB placeholder
                inp = [line.replace('JOB', job) if 'JOB' in line else line for line in inp]
                write_lines(out_path / f"{job}.inp", inp)
                write_exec(out_path / f"{job}.sh", orca_write_sh(job, cfg))
                
                if cfg['SCHEDULER'] == "pbs":
                    submit_lines.append(f"qsub {job}.sh")
                elif cfg['SCHEDULER'] == "slurm":
                    submit_lines.append(f"sbatch {job}.sh")
                else:
                    submit_lines.append(f"bash {job}.sh")
                generated.append(job)
        
        write_exec(out_path / "submit_all.sh", submit_lines)
        
        msg = f"Successfully generated {len(generated)} jobs for {len(files_data)} input(s) → {out_path.resolve()}"
        return True, msg, generated
        
    except Exception as e:
        return False, f"Error generating files: {str(e)}", []
