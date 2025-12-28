"""
TICT Rotation Module
--------------------
Core functions for performing TICT (Twisted Intramolecular Charge Transfer) rotations.
Based on the old working TICT implementation.
Supports both Gaussian (.com) and ORCA (.xyz/.inp) file formats.
"""
import numpy as np
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict


def parse_atom_string(atom_string: str) -> List[int]:
    """
    Parses a GaussView-style atom string (e.g., "3-4,7,10-11")
    into a list of 0-based atom indices.
    
    Args:
        atom_string: String like "3-4,7,10-11" (1-based indexing)
        
    Returns:
        Sorted list of 0-based atom indices
    """
    if not atom_string:
        return []
    
    indices = set()
    parts = atom_string.split(',')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        if '-' in part:
            try:
                start, end = part.split('-')
                start_val = int(start)
                end_val = int(end)
                
                # Ensure start is less than end
                if start_val > end_val:
                    start_val, end_val = end_val, start_val
                
                # Add all numbers in the range (inclusive), convert to 0-based
                for i in range(start_val, end_val + 1):
                    indices.add(i - 1)
            except ValueError:
                pass
        else:
            try:
                indices.add(int(part) - 1)  # Convert to 0-based
            except ValueError:
                pass
    
    return sorted(list(indices))


def load_gaussian_geometry(filepath: str) -> Tuple[List[str], str, str, List[str], np.ndarray]:
    """
    Loads geometry from a Gaussian .com file.
    
    Returns:
        Tuple of (header_lines, title_line, cm_line, symbols, coords_array)
        Returns None for first element if error, with error message in title_line
    """
    try:
        with open(filepath, 'r') as f:
            lines = [line.rstrip() for line in f.readlines()]
        
        header_lines = []
        title_line = "Title (from original file)"
        cm_line = "0 1"
        coord_lines_raw = []
        
        cm_re = re.compile(r'^\s*(-?\d+)\s+(\d+)\s*$')
        coord_re = re.compile(r'^\s*(\S+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*')
        
        # Find the end of the header (first blank line)
        first_blank_line = -1
        for i, line in enumerate(lines):
            if not line.strip():
                first_blank_line = i
                break
        if first_blank_line == -1:
            return None, "Error: Could not find a blank line after route section.", None, None, None
        
        header_lines = lines[:first_blank_line]
        
        # Find the title (line after first blank)
        if first_blank_line + 1 < len(lines):
            title_line = lines[first_blank_line + 1].strip()
        
        # Find the C/M line (after title and another blank line)
        second_blank_line = -1
        for i in range(first_blank_line + 2, len(lines)):
            if not lines[i].strip():
                second_blank_line = i
                break
        if second_blank_line == -1:
            return None, "Error: Could not find a blank line after title.", None, None, None
        
        cm_line_index = second_blank_line + 1
        if cm_line_index < len(lines) and cm_re.match(lines[cm_line_index].strip()):
            cm_line = lines[cm_line_index].strip()
        else:
            return None, "Error: Could not find charge/multiplicity line.", None, None, None
        
        # Find coordinates (after C/M line)
        for i in range(cm_line_index + 1, len(lines)):
            line = lines[i].strip()
            if not line:
                break  # End of coordinates
            if coord_re.match(line):
                coord_lines_raw.append(line)
        
        if not coord_lines_raw:
            return None, "Error: Could not find coordinates.", None, None, None
        
        symbols, coords_list = [], []
        for line in coord_lines_raw:
            match = coord_re.match(line)
            if match:
                symbols.append(match.group(1))
                coords_list.append([float(match.group(2)), float(match.group(3)), float(match.group(4))])
        
        return header_lines, title_line, cm_line, symbols, np.array(coords_list)
    
    except Exception as e:
        return None, f"Error: {e}", None, None, None


def load_orca_geometry(filepath: str) -> Tuple[str, List[str], np.ndarray]:
    """
    Loads geometry from an ORCA .xyz or .inp file.
    
    Returns:
        Tuple of (cm_line, symbols, coords_array)
        Returns None for symbols if error, with error message in cm_line
    """
    try:
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r') as f:
            lines = [line.rstrip() for line in f.readlines()]
        
        # Try .xyz format first (starts with number of atoms)
        if lines and lines[0].strip().isdigit():
            n_atoms = int(lines[0].strip())
            title_line = lines[1].strip() if len(lines) > 1 else ""
            
            # Try to extract charge/multiplicity from title line
            cm_line = "0 1"  # Default
            cm_match = re.search(r'(\d+)\s+(\d+)', title_line)
            if cm_match:
                cm_line = f"{cm_match.group(1)} {cm_match.group(2)}"
            
            symbols = []
            coords_list = []
            coord_re = re.compile(r'^\s*(\S+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*')
            
            for i in range(2, min(2 + n_atoms, len(lines))):
                line = lines[i].strip()
                if not line:
                    break
                match = coord_re.match(line)
                if match:
                    symbols.append(match.group(1))
                    coords_list.append([float(match.group(2)), float(match.group(3)), float(match.group(4))])
            
            if len(symbols) != n_atoms:
                return f"Error: Expected {n_atoms} atoms, found {len(symbols)}.", None, None
            
            return cm_line, symbols, np.array(coords_list)
        
        # Try .inp format (look for *xyz section)
        else:
            in_xyz = False
            symbols = []
            coords_list = []
            coord_re = re.compile(r'^\s*(\S+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*')
            cm_line = "0 1"
            
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.lower().startswith('*xyz'):
                    in_xyz = True
                    # Try to get charge/multiplicity from *xyz line
                    cm_match = re.search(r'(\d+)\s+(\d+)', line_stripped)
                    if cm_match:
                        cm_line = f"{cm_match.group(1)} {cm_match.group(2)}"
                    continue
                if line_stripped.lower() == '*':
                    if in_xyz:
                        break  # End of xyz block
                if in_xyz and coord_re.match(line_stripped):
                    match = coord_re.match(line_stripped)
                    symbols.append(match.group(1))
                    coords_list.append([float(match.group(2)), float(match.group(3)), float(match.group(4))])
            
            if not symbols:
                return "Error: Could not find coordinates in ORCA file.", None, None
            
            return cm_line, symbols, np.array(coords_list)
    
    except Exception as e:
        return f"Error: {e}", None, None


def get_rotation_matrix(axis: np.ndarray, theta_rad: float) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Returns the 3D rotation matrix (Rodrigues' rotation formula).
    
    Args:
        axis: 3D rotation axis vector
        theta_rad: Rotation angle in radians
        
    Returns:
        Tuple of (rotation_matrix, error_message)
    """
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        return None, "Error: Rotation axis vector is zero."
    
    axis = axis / axis_norm  # Normalize
    a = np.cos(theta_rad / 2.0)
    b, c, d = -axis * np.sin(theta_rad / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    
    matrix = np.array([[aa+bb-cc-dd, 2*(bc-ad), 2*(bd+ac)],
                       [2*(bc+ad), aa+cc-bb-dd, 2*(cd-ab)],
                       [2*(bd-ac), 2*(cd+ab), aa+dd-bb-cc]])
    return matrix, None


def apply_rotation(coords: np.ndarray, atom_indices: List[int], matrix: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    Applies a rotation to a subset of atoms.
    
    Args:
        coords: Nx3 array of coordinates
        atom_indices: List of 0-based atom indices to rotate
        matrix: 3x3 rotation matrix
        center: 3D point around which to rotate
        
    Returns:
        Modified coordinates array
    """
    if not atom_indices:
        return coords
    
    moving_coords = coords[atom_indices].copy()
    moving_coords_translated = moving_coords - center
    moving_coords_rotated = np.dot(matrix, moving_coords_translated.T).T
    coords[atom_indices] = moving_coords_rotated + center
    return coords


def get_coordinates_as_com_lines(symbols: List[str], coords: np.ndarray) -> List[str]:
    """Format coordinates as Gaussian .com file lines."""
    return [f" {s:<2}         {c[0]: 12.8f} {c[1]: 12.8f} {c[2]: 12.8f}" for s, c in zip(symbols, coords)]


def get_coordinates_as_xyz_lines(symbols: List[str], coords: np.ndarray) -> List[str]:
    """Format coordinates as .xyz file lines."""
    return [f"  {s:<2}   {c[0]: 15.10f}   {c[1]: 15.10f}   {c[2]: 15.10f}" for s, c in zip(symbols, coords)]


def generate_tict_rotations(
    input_file: str,
    output_dir: str,
    axis_str: str,
    branch_a_str: str,
    branch_a_step_deg: float,
    branch_b_str: str,
    branch_b_step_deg: float,
    num_steps: int,
    file_format: str = "gaussian"  # "gaussian" or "orca"
) -> Tuple[bool, str, List[str]]:
    """
    Generates TICT rotated geometry files using the old working logic.
    
    Args:
        input_file: Path to input geometry file (.com for Gaussian, .xyz/.inp for ORCA)
        output_dir: Directory to save rotated geometries
        axis_str: Rotation axis atoms (e.g., "3,10" for 1-based indices)
        branch_a_str: Branch A atom indices (1-based)
        branch_a_step_deg: Rotation step for Branch A in degrees
        branch_b_str: Branch B atom indices (1-based)
        branch_b_step_deg: Rotation step for Branch B in degrees
        num_steps: Number of rotation steps (0 to num_steps inclusive)
        file_format: Output format ("gaussian" or "orca")
        
    Returns:
        Tuple of (success: bool, message: str, files_created: List[str])
    """
    try:
        # Parse atom strings (1-based -> 0-based)
        axis_indices = parse_atom_string(axis_str)
        branch_a_indices = parse_atom_string(branch_a_str)
        branch_b_indices = parse_atom_string(branch_b_str)
        
        if len(axis_indices) != 2:
            return False, f"Error: Rotation Axis must have exactly 2 atoms. You provided: {axis_str}", []
        
        # Load geometry based on format
        if file_format.lower() == "orca":
            cm_line, symbols, original_coords = load_orca_geometry(input_file)
            if symbols is None:
                return False, f"Error: Could not load ORCA geometry file: {cm_line}", []
            title_line = ""
            header_lines = []
        else:  # gaussian
            header_lines, title_line, cm_line, symbols, original_coords = load_gaussian_geometry(input_file)
            if symbols is None:
                return False, f"Error: Could not load Gaussian geometry file: {title_line}", []
        
        # Define rotation axis (OLD WORKING LOGIC)
        atom_b, atom_c = axis_indices[0], axis_indices[1]
        rotation_center = original_coords[atom_b].copy()  # Use first atom of axis as center
        rotation_axis_vector = original_coords[atom_c] - original_coords[atom_b]
        
        if np.linalg.norm(rotation_axis_vector) == 0:
            return False, f"Error: Axis atoms {atom_b+1} and {atom_c+1} are at the same position.", []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base name
        base_name = Path(input_file).stem
        
        files_created = []
        
        # Generate rotated geometries (OLD WORKING LOGIC - always start from original)
        for i in range(num_steps + 1):
            current_coords = original_coords.copy()  # Always start from original
            
            # Rotate Branch A
            angle_a_deg = i * branch_a_step_deg
            angle_a_rad = np.radians(angle_a_deg)
            if angle_a_rad != 0:
                matrix_a, err = get_rotation_matrix(rotation_axis_vector, angle_a_rad)
                if err:
                    return False, err, []
                current_coords = apply_rotation(current_coords, branch_a_indices, matrix_a, rotation_center)
            
            # Rotate Branch B
            angle_b_deg = i * branch_b_step_deg
            angle_b_rad = np.radians(angle_b_deg)
            if angle_b_rad != 0:
                matrix_b, err = get_rotation_matrix(rotation_axis_vector, angle_b_rad)
                if err:
                    return False, err, []
                current_coords = apply_rotation(current_coords, branch_b_indices, matrix_b, rotation_center)
            
            # Write output file
            if file_format.lower() == "orca":
                output_filename = f"{base_name}_{i:03d}.xyz"
                output_filepath = os.path.join(output_dir, output_filename)
                
                coord_lines = get_coordinates_as_xyz_lines(symbols, current_coords)
                with open(output_filepath, 'w') as f:
                    f.write(f"{len(symbols)}\n")
                    f.write(f"TICT Rotation Step {i}\n")
                    f.write("\n".join(coord_lines) + "\n")
            else:  # gaussian
                output_filename = f"{base_name}_{i:03d}.com"
                output_filepath = os.path.join(output_dir, output_filename)
                
                coord_lines = get_coordinates_as_com_lines(symbols, current_coords)
                with open(output_filepath, 'w') as f:
                    f.write("\n".join(header_lines) + "\n")
                    f.write("\n")
                    f.write(f"{title_line} (TICT Step {i})\n")
                    f.write("\n")
                    f.write(f"{cm_line}\n")
                    f.write("\n".join(coord_lines) + "\n")
                    f.write("\n")
            
            files_created.append(output_filepath)
        
        return True, f"Successfully created {len(files_created)} rotated geometry files.", files_created
        
    except Exception as e:
        return False, f"Error: {str(e)}", []

