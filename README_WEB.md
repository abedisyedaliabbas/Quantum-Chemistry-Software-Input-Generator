# Quantum Chemical Calculations Step Maker - Web Version

A web-based version of the Quantum Chemical Calculations Step Maker that works on iOS Safari and any modern web browser.

## Features

- ✅ Works on iOS Safari, Android Chrome, and desktop browsers
- ✅ Full Gaussian and ORCA support
- ✅ SMILES input with ChemDraw SVG name extraction
- ✅ File upload (.com, .log files)
- ✅ Preview generated files before downloading
- ✅ Download all generated files as a ZIP archive
- ✅ Modern, responsive UI

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements_web.txt
```

2. Make sure you have the desktop version files:
   - `gaussian_steps_gui.py`
   - `quantum_steps_gui.py`
   - `ORCA_Step_Maker.py` (if using ORCA)

## Running the Web App

1. Start the Flask server:
```bash
python quantum_steps_web.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. For iOS devices on the same network:
```
http://YOUR_COMPUTER_IP:5000
```

To find your computer's IP address:
- Windows: `ipconfig` (look for IPv4 Address)
- Mac/Linux: `ifconfig` or `ip addr`

## Usage

1. **Select Software**: Choose Gaussian or ORCA at the top
2. **Configure Settings**: Use the tabs to configure:
   - Mode & Steps
   - Method & Basis Set
   - Solvent
   - TD-DFT settings
   - Resources
   - Scheduler
   - Advanced options
3. **Input Files**: 
   - Upload .com or .log files, OR
   - Enter SMILES strings
4. **Preview**: Click "Preview" to see generated files
5. **Generate**: Click "Generate Files" to download a ZIP archive

## File Structure

```
.
├── quantum_steps_web.py      # Flask backend
├── templates/
│   └── index.html            # Main web UI
├── static/
│   ├── css/
│   │   └── style.css         # Styles
│   └── js/
│       └── app.js            # Frontend logic
└── requirements_web.txt      # Python dependencies
```

## API Endpoints

- `GET /` - Main page
- `GET /api/defaults/<software>` - Get default settings
- `POST /api/route-preview` - Preview route for a step
- `POST /api/smiles-to-coords` - Convert SMILES to coordinates
- `POST /api/extract-names-from-svg` - Extract names from SVG
- `POST /api/parse-log` - Parse Gaussian log file
- `POST /api/preview` - Preview generated files
- `POST /api/generate` - Generate and download files
- `GET /health` - Health check

## Notes

- Files are temporarily stored in system temp directory
- Generated ZIP files are automatically cleaned up
- The web version uses the same backend logic as the desktop version
- RDKit is required for SMILES processing (install separately if needed)

## Troubleshooting

**Can't access from iOS device:**
- Make sure your computer and iOS device are on the same WiFi network
- Check firewall settings (port 5000 should be open)
- Try using `0.0.0.0` as host (already set in code)

**RDKit not working:**
- Install RDKit separately: `pip install rdkit-pypi` or use conda
- Some features may be limited without RDKit

**Import errors:**
- Make sure `gaussian_steps_gui.py` and `quantum_steps_gui.py` are in the same directory
- Check that all required Python packages are installed


