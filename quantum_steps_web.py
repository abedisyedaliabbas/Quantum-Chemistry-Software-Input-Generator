#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Chemical Calculations Step Maker - Web Version
-------------------------------------------------------
Flask-based web application for generating input files for quantum chemistry software.

Run:
  python quantum_steps_web.py
  Then open http://localhost:5000 in your browser
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from pathlib import Path
import json
import zipfile
import io
import os
import tempfile
import shutil
from typing import Dict, Any, List

# Import all the backend logic from the desktop version
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
    )
    GAUSSIAN_AVAILABLE = True
except ImportError as e:
    GAUSSIAN_AVAILABLE = False
    GAUSSIAN_ERROR = str(e)
    print(f"Warning: Could not import Gaussian modules: {e}")

# Import ORCA functions from quantum_steps_gui
try:
    from quantum_steps_gui import (
        ORCA_DEFAULTS,
        orca_find_inputs, orca_extract_geom, orca_parse_log, orca_smiles_to_coords,
        orca_make_inp, orca_write_sh, orca_build_step, orca_jobname,
    )
    ORCA_AVAILABLE = True
except ImportError as e:
    ORCA_AVAILABLE = False
    ORCA_DEFAULTS = {}
    # Define placeholder functions
    def orca_build_step(*args, **kwargs): return []
    def orca_jobname(*args, **kwargs): return "job"
    def orca_write_sh(*args, **kwargs): return []
    print(f"Warning: Could not import ORCA modules: {e}")

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configuration
UPLOAD_FOLDER = Path(tempfile.gettempdir()) / 'quantum_steps_uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER = Path(tempfile.gettempdir()) / 'quantum_steps_output'
OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/defaults/<software>')
def get_defaults(software):
    """Get default settings for software"""
    if software == 'gaussian':
        return jsonify(GAUSSIAN_DEFAULTS if GAUSSIAN_AVAILABLE else {})
    elif software == 'orca':
        return jsonify(ORCA_DEFAULTS)
    return jsonify({})


@app.route('/api/route-preview', methods=['POST'])
def route_preview():
    """Preview route for a specific step"""
    data = request.json
    software = data.get('software')
    step = data.get('step')
    config = data.get('config', {})
    
    if software == 'gaussian' and GAUSSIAN_AVAILABLE:
        try:
            route_funcs = {
                1: route_step1, 2: route_step2, 3: route_step3, 4: route_step4,
                5: route_step5, 6: route_step6, 7: route_step7
            }
            if step in route_funcs:
                route = route_funcs[step](config)
                return jsonify({'route': route, 'error': None})
        except Exception as e:
            return jsonify({'route': '', 'error': str(e)})
    
    return jsonify({'route': '', 'error': 'Not supported'})


@app.route('/api/smiles-to-coords', methods=['POST'])
def smiles_to_coords_endpoint():
    """Convert SMILES to coordinates"""
    data = request.json
    smiles_input = data.get('smiles', '')
    charge = data.get('charge', 0)
    mult = data.get('mult', 1)
    
    if not RDKIT_AVAILABLE:
        return jsonify({'error': 'RDKit not available', 'coords': []})
    
    try:
        lines = smiles_input.strip().split('\n')
        results = []
        for line in lines:
            if not line.strip() or line.strip().startswith('#'):
                continue
            parsed = parse_smiles_line(line)
            if parsed:
                name, smiles_str, ch, mu = parsed
                coords = smiles_to_coords(smiles_str, ch or charge, mu or mult)
                if coords:
                    results.append({
                        'name': name or f'molecule_{len(results)+1}',
                        'smiles': smiles_str,
                        'coords': coords,
                        'charge': ch or charge,
                        'mult': mu or mult
                    })
        return jsonify({'coords': results, 'error': None})
    except Exception as e:
        return jsonify({'coords': [], 'error': str(e)})


@app.route('/api/extract-names-from-svg', methods=['POST'])
def extract_names_from_svg_endpoint():
    """Extract names from SVG file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded', 'names': []})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected', 'names': []})
    
    if not XML_AVAILABLE:
        return jsonify({'error': 'XML parser not available', 'names': []})
    
    try:
        # Save uploaded file temporarily
        svg_path = UPLOAD_FOLDER / file.filename
        file.save(str(svg_path))
        
        # Extract names
        names = extract_names_from_svg(svg_path)
        
        # Clean up
        svg_path.unlink()
        
        return jsonify({'names': names, 'error': None})
    except Exception as e:
        return jsonify({'names': [], 'error': str(e)})


@app.route('/api/parse-log', methods=['POST'])
def parse_log_endpoint():
    """Parse Gaussian log file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Save uploaded file temporarily
        log_path = UPLOAD_FOLDER / file.filename
        file.save(str(log_path))
        
        # Parse log file
        cm, coords = parse_gaussian_log(log_path)
        
        # Clean up
        log_path.unlink()
        
        return jsonify({
            'charge': cm[0] if cm else 0,
            'mult': cm[1] if cm else 1,
            'coords': coords,
            'error': None
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/generate', methods=['POST'])
def generate_files():
    """Generate input files"""
    data = request.json
    software = data.get('software')
    config = data.get('config', {})
    files_data = data.get('files', [])  # List of file contents or file info
    
    try:
        # Create temporary output directory
        output_dir = OUTPUT_FOLDER / f"job_{os.urandom(8).hex()}"
        output_dir.mkdir(exist_ok=True)
        
        if software == 'gaussian' and GAUSSIAN_AVAILABLE:
            # Process files and generate Gaussian inputs
            generated_files = []
            for file_info in files_data:
                base_name = file_info.get('name', 'molecule')
                coords = file_info.get('coords', [])
                charge = file_info.get('charge', 0)
                mult = file_info.get('mult', 1)
                
                # Generate files based on mode
                mode = config.get('MODE', 'single')
                if mode == 'full':
                    jobs = generate_full(base_name, f"{charge} {mult}", coords, output_dir, config)
                else:
                    jobs = generate_single(base_name, f"{charge} {mult}", coords, output_dir, config)
                
                generated_files.extend(jobs)
        
        elif software == 'orca' and ORCA_AVAILABLE:
            # Process ORCA generation
            generated_files = []
            for file_info in files_data:
                base_name = file_info.get('name', 'molecule')
                coords = file_info.get('coords', [])
                charge = file_info.get('charge', 0)
                mult = file_info.get('mult', 1)
                
                mode = config.get('MODE', 'single')
                step = config.get('STEP', 4)
                cm = f"{charge} {mult}"
                
                if mode == 'full':
                    steps = [1, 2, 4, 7, 9]
                else:
                    steps = [step]
                
                for s in steps:
                    try:
                        # Build step content
                        step_content = orca_build_step(s, config, cm, coords)
                        
                        # Generate job name
                        method = config.get('METHOD', 'm06-2x')
                        basis = config.get('BASIS', 'def2-TZVP')
                        solv_model = config.get('SOLVENT_MODEL', 'none')
                        solv_name = config.get('SOLVENT_NAME', 'DMSO')
                        job = orca_jobname(s, base_name, method, basis, solv_model, solv_name)
                        
                        # Replace "JOB" placeholder in step content with actual job name
                        step_content = [line.replace('JOB', job) if 'JOB' in line else line for line in step_content]
                        
                        # Write .inp file
                        inp_path = output_dir / f"{job}.inp"
                        write_lines(inp_path, step_content)
                        generated_files.append(str(inp_path))
                        
                        # Write .sh file
                        sh_content = orca_write_sh(job, config)
                        sh_path = output_dir / f"{job}.sh"
                        write_exec(sh_path, sh_content)
                        generated_files.append(str(sh_path))
                    except Exception as e:
                        print(f"Error generating ORCA step {s}: {e}")
                        continue
        
        # Create ZIP file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in generated_files:
                if Path(file_path).exists():
                    zip_file.write(file_path, Path(file_path).name)
        
        zip_buffer.seek(0)
        
        # Clean up output directory
        shutil.rmtree(output_dir, ignore_errors=True)
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'{software}_jobs.zip'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preview', methods=['POST'])
def preview_files():
    """Preview generated files without creating them"""
    data = request.json
    software = data.get('software')
    config = data.get('config', {})
    file_info = data.get('file', {})
    
    try:
        base_name = file_info.get('name', 'molecule')
        coords = file_info.get('coords', [])
        charge = file_info.get('charge', 0)
        mult = file_info.get('mult', 1)
        
        previews = []
        
        if software == 'gaussian' and GAUSSIAN_AVAILABLE:
            mode = config.get('MODE', 'single')
            step = config.get('STEP', 1)
            
            if mode == 'full':
                steps = [1, 2, 3, 4, 5, 6, 7]
            elif mode == 'multiple':
                steps = config.get('MULTI_STEPS', [step])
            else:
                steps = [step]
            
            for s in steps:
                route = step_route(s, config)
                job = jobname(s, base_name, config)
                title = f"Step{s} {config.get('FUNCTIONAL', '')}/{config.get('BASIS', '')}"
                com_content = make_com_inline(
                    job, config.get('NPROC', 64), config.get('MEM', '128GB'),
                    route, title, f"{charge} {mult}", coords,
                    save_rwf=config.get('SOC_ENABLE', False)
                )
                previews.append({
                    'filename': f"{job}.com",
                    'content': '\n'.join(com_content)
                })
        
        elif software == 'orca' and ORCA_AVAILABLE:
            mode = config.get('MODE', 'single')
            step = config.get('STEP', 4)
            cm = f"{charge} {mult}"
            
            if mode == 'full':
                steps = [1, 2, 4, 7, 9]
            else:
                steps = [step]
            
            for s in steps:
                try:
                    step_content = orca_build_step(s, config, cm, coords)
                    method = config.get('METHOD', 'm06-2x')
                    basis = config.get('BASIS', 'def2-TZVP')
                    solv_model = config.get('SOLVENT_MODEL', 'none')
                    solv_name = config.get('SOLVENT_NAME', 'DMSO')
                    job = orca_jobname(s, base_name, method, basis, solv_model, solv_name)
                    
                    # Replace "JOB" placeholder in step content with actual job name
                    step_content = [line.replace('JOB', job) if 'JOB' in line else line for line in step_content]
                    
                    previews.append({
                        'filename': f"{job}.inp",
                        'content': '\n'.join(step_content)
                    })
                except Exception as e:
                    previews.append({
                        'filename': f"error_step{s}.inp",
                        'content': f"# Error generating step {s}: {str(e)}"
                    })
        
        return jsonify({'previews': previews, 'error': None})
    
    except Exception as e:
        return jsonify({'previews': [], 'error': str(e)})


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'gaussian_available': GAUSSIAN_AVAILABLE,
        'rdkit_available': RDKIT_AVAILABLE if GAUSSIAN_AVAILABLE else False,
        'xml_available': XML_AVAILABLE if GAUSSIAN_AVAILABLE else False,
    })


if __name__ == '__main__':
    import socket
    
    # Get local IP address
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
    
    local_ip = get_local_ip()
    
    print("=" * 60)
    print("Quantum Chemical Calculations Step Maker - Web Version")
    print("=" * 60)
    print(f"Starting server...")
    print(f"\n📍 Access URLs:")
    print(f"   Local:    http://localhost:5000")
    print(f"   Network:  http://{local_ip}:5000")
    print(f"\n💡 Share this URL with friends on the same WiFi:")
    print(f"   http://{local_ip}:5000")
    print(f"\n📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📁 Output folder: {OUTPUT_FOLDER}")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

