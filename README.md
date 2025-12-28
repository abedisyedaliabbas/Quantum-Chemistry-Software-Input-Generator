# 🧬 Quantum Chemical Calculations Step Maker

A comprehensive Python application for automatically generating input files for quantum chemistry software packages (Gaussian and ORCA). Available as both a **desktop GUI application** and a **web-based interface** that works on iOS, Android, and desktop browsers.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## ✨ Features

### 🖥️ Desktop GUI Version
- **Beautiful modern interface** with tabbed navigation
- **Dual software support**: Gaussian (G16/G09) and ORCA
- **Multiple input types**: 
  - `.com` files (Gaussian input format)
  - `.log` files (extract geometry from completed calculations)
  - `.xyz` files (XYZ coordinate format)
  - **SMILES strings** with ChemDraw SVG name extraction
- **Full workflow generation** (steps 1-7 for Gaussian, steps 1,2,4,7,9 for ORCA)
- **PySOC integration** for Spin-Orbit Coupling calculations
- **Scheduler support**: PBS, SLURM, and Local
- **Advanced features**: 
  - TD-DFT with Singlet/Triplet/Mixed states
  - Solvent models (PCM, SMD)
  - Custom routes and keywords
  - Redundant coordinates
  - Geometry chaining

### 🌐 Web Version
- **Cross-platform**: Works on iOS Safari, Android Chrome, and desktop browsers
- **No installation required**: Just open in browser
- **File upload/download**: Easy file handling via web interface
- **Same powerful features** as desktop version
- **Responsive design**: Optimized for mobile and desktop

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Features in Detail](#features-in-detail)
- [Web Version](#web-version)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- RDKit (for SMILES processing) - optional but recommended
  ```bash
  # Using conda (recommended)
  conda install -c conda-forge rdkit
  
  # Or using pip (may have limitations)
  pip install rdkit-pypi
  ```

### Desktop GUI Version

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abedisyedaliabbas/Quantum-Chemistry-Software-Input-Generator.git
   cd Quantum-Chemistry-Software-Input-Generator
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python quantum_steps_gui.py
   ```

### Web Version

1. **Install web dependencies:**
   ```bash
   pip install -r requirements_web.txt
   ```

2. **Start the web server:**
   ```bash
   python quantum_steps_web.py
   ```

3. **Open in browser:**
   - Local: `http://localhost:5000`
   - Network: `http://YOUR_IP:5000` (for sharing with friends on same WiFi)

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed deployment instructions.

## 🎯 Quick Start

### Desktop GUI

1. Launch `quantum_steps_gui.py`
2. Select software (Gaussian or ORCA)
3. Choose input type (`.com`, `.log`, `.xyz`, or SMILES)
4. Configure settings in tabs:
   - **Mode & IO**: Select mode (Full/Single/Multiple) and input files
   - **Method**: Choose functional and basis set
   - **Solvent**: Configure solvent model
   - **TD-DFT**: Set excited state parameters
   - **Resources**: Set computational resources
   - **Scheduler**: Configure job submission
   - **Advanced**: Additional options
5. Click **Generate** to create all input files

### Web Version

1. Start the server: `python quantum_steps_web.py`
2. Open `http://localhost:5000` in your browser
3. Follow the same workflow as desktop version
4. Download generated files as ZIP archive

## 📖 Usage

### Gaussian Workflow Steps

1. **Step 1**: Ground state geometry optimization
2. **Step 2**: Vertical excitation (Franck-Condon state)
3. **Step 3**: cLR correction of vertical excitation energy
4. **Step 4**: Excited state geometry optimization
5. **Step 5**: Density calculation at optimized excited state geometry
6. **Step 6**: cLR correction of excited state energy
7. **Step 7**: Ground state energy at excited state geometry

### ORCA Workflow Steps

1. **Step 1**: Ground state optimization
2. **Step 2**: Vertical excitation (Franck-Condon state)
3. **Step 4**: Excited state optimization
4. **Step 7**: Ground state at excited state geometry
5. **Step 9**: Custom step (user-defined)

### Input Types

#### `.com` Files
- Standard Gaussian input files
- Automatically extracts geometry, charge, and multiplicity

#### `.log` Files
- Completed Gaussian calculations
- Extracts final optimized geometry
- Useful for continuing calculations

#### `.xyz` Files
- Standard XYZ coordinate format
- Simple coordinate input

#### SMILES Strings
- Chemical structure input using SMILES notation
- Supports ChemDraw SVG name extraction
- Automatic 3D coordinate generation using RDKit
- Format: `name:SMILES` or just `SMILES`

Example:
```
formaldehyde:CO
benzene:c1ccccc1
water:O
```

### SMILES with ChemDraw Names

1. Copy SMILES from ChemDraw: `Edit → Copy As → SMILES`
2. Export structure names: `File → Save As → SVG`
3. Click "Load Names from SVG" in the GUI
4. Names are automatically matched with SMILES!

## 🔧 Features in Detail

### Software Selection
- **Gaussian**: Full support for G16/G09 workflows
- **ORCA**: Complete ORCA input file generation

### Modes
- **Full Mode**: Generate all steps (1-7 for Gaussian, 1,2,4,7,9 for ORCA)
- **Single Mode**: Generate one selected step
- **Multiple Mode**: Select specific steps to generate

### TD-DFT Options
- **Singlet**: Standard closed-shell singlet states
- **Triplet**: Explicit triplet states (`TD(Triplets, NStates=n)`)
- **Mixed (50-50)**: Mixed singlet-triplet states for SOC calculations

### Solvent Models
- **PCM**: Polarizable Continuum Model
- **SMD**: Solvation Model based on Density
- **None**: Gas phase calculations

### PySOC Integration
- Automatic preparation for Spin-Orbit Coupling calculations
- Generates bash scripts for running PySOC
- Combines results into Excel files
- Requires: `TD(50-50)`, `%rwf`, `6D 10F GFInput`

### Schedulers
- **PBS**: Portable Batch System
- **SLURM**: Simple Linux Utility for Resource Management
- **Local**: Local execution scripts

## 🌐 Web Version

The web version provides the same functionality as the desktop GUI but accessible through any web browser.

### Sharing with Friends

**Local Network (Same WiFi):**
1. Start server: `python quantum_steps_web.py`
2. Share the URL shown (e.g., `http://192.168.1.100:5000`)
3. Friends can access from their devices

**Cloud Deployment:**
- Deploy to Railway, Render, or PythonAnywhere for public access
- See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for details

### Web API Endpoints

- `GET /` - Main web interface
- `GET /api/defaults/<software>` - Get default settings
- `POST /api/route-preview` - Preview route for a step
- `POST /api/smiles-to-coords` - Convert SMILES to coordinates
- `POST /api/extract-names-from-svg` - Extract names from SVG
- `POST /api/parse-log` - Parse Gaussian log file
- `POST /api/preview` - Preview generated files
- `POST /api/generate` - Generate and download files
- `GET /health` - Health check

## 📁 Project Structure

```
Quantum-Chemistry-Software-Input-Generator/
├── quantum_steps_gui.py          # Main desktop GUI application
├── quantum_steps_web.py         # Flask web server
├── gaussian_steps_gui.py        # Gaussian backend logic
├── ORCA_Step_Maker.py           # ORCA backend logic (if separate)
├── templates/                    # Web UI templates
│   └── index.html
├── static/                       # Web static files
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── requirements.txt             # Desktop dependencies
├── requirements_web.txt         # Web dependencies
├── README.md                     # This file
├── DEPLOYMENT_GUIDE.md          # Web deployment guide
├── LICENSE                       # MIT License
└── .gitignore                   # Git ignore rules
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Gaussian**: For the powerful quantum chemistry software
- **ORCA**: For the open-source quantum chemistry package
- **RDKit**: For SMILES processing and 3D coordinate generation
- **PySOC**: For Spin-Orbit Coupling calculations
- **ChemDraw**: For structure drawing and SMILES export

## 📧 Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**Made with ❤️ for the quantum chemistry community**


