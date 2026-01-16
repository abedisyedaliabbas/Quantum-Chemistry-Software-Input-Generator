# 🧬 Quantum Chemical Calculations Step Maker

A comprehensive Python application for automatically generating input files for quantum chemistry software packages (Gaussian and ORCA). Available as both a **desktop GUI application** and a **web-based interface** that works on iOS, Android, and desktop browsers.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## ✨ Features

### 🖥️ Desktop GUI Version
- **Beautiful modern interface** with tabbed navigation and scrollable frames
- **Dual software support**: Gaussian (G16/G09) and ORCA with unified interface
- **Multiple input types**: 
  - `.com` files (Gaussian input format)
  - `.log` files (extract geometry from completed calculations)
  - `.xyz` files (XYZ coordinate format)
  - **SMILES strings** with ChemDraw SVG name extraction
- **Full workflow generation** (steps 1-7 for Gaussian, steps 1,2,4,7,9 for ORCA)
- **TICT Rotation**: Advanced torsional/dihedral scan generation
- **AI Assistant**: Conversational AI for generating input files (Ollama & Gemini support)
- **PySOC integration** for Spin-Orbit Coupling calculations
- **Scheduler support**: PBS, SLURM, and Local with customizable resources
- **Advanced features**: 
  - TD-DFT with Singlet/Triplet/Mixed states
  - Solvent models (PCM, SMD, IEFPCM, CPCM)
  - Custom routes and keywords per step
  - Redundant coordinates
  - Geometry chaining
  - Bulk file processing (folders and glob patterns)

### 🌐 Web Version
- **Cross-platform**: Works on iOS Safari, Android Chrome, and desktop browsers
- **No installation required**: Just open in browser
- **File upload/download**: Easy file handling via web interface
- **Same powerful features** as desktop version
- **Responsive design**: Optimized for mobile and desktop

## 🚀 Quick Start

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

## 📖 Usage

### Workflow Steps

**Gaussian:**
1. Ground state geometry optimization
2. Vertical excitation (Franck-Condon state)
3. cLR correction of vertical excitation energy
4. Excited state geometry optimization
5. Density calculation at optimized excited state geometry
6. cLR correction of excited state energy
7. Ground state energy at excited state geometry

**ORCA:**
1. Ground state optimization
2. Vertical excitation (Franck-Condon state)
4. Excited state optimization
7. Ground state at excited state geometry
9. Custom step (user-defined)

### Input Types

- **`.com` files**: Standard Gaussian input files - automatically extracts geometry, charge, and multiplicity
- **`.log` files**: Completed Gaussian calculations - extracts final optimized geometry
- **`.xyz` files**: Standard XYZ coordinate format
- **SMILES strings**: Chemical structure input with automatic 3D coordinate generation using RDKit
  - Format: `name:SMILES` or just `SMILES`
  - Supports ChemDraw SVG name extraction

### Key Features

- **Modes**: Full (all steps), Single (one step), Multiple (selected steps)
- **TD-DFT Options**: Singlet, Triplet, Mixed (50-50) states
- **Solvent Models**: PCM, SMD, IEFPCM, CPCM, or gas phase
- **TICT Rotation**: Generate torsional/dihedral scans with automatic branch detection
- **AI Assistant**: Natural language interface for generating input files
- **PySOC Integration**: Automatic preparation for Spin-Orbit Coupling calculations
- **Schedulers**: PBS, SLURM, and Local execution scripts

## 🔧 Advanced Features

### TICT Rotation
Generate rotated geometries for torsional/dihedral scans. Supports:
- Automatic branch detection
- Multiple rotation axes
- Custom step sizes
- Bulk file processing

### AI Assistant
Conversational interface powered by Ollama (local, free) or Google Gemini:
- Natural language input file generation
- Automatic parameter detection
- Bulk file processing support
- TICT scan generation from descriptions

**Setup AI Assistant:**
- **Ollama** (Recommended): Install from [ollama.com](https://ollama.com), then run `ollama pull llama3.1:8b`
- **Google Gemini**: Get API key from [makersuite.google.com](https://makersuite.google.com/app/apikey) and enter in AI Assistant tab

### PySOC Integration
Automatic preparation for Spin-Orbit Coupling calculations:
- Generates `.com` files with `%rwf` and `6D 10F GFInput`
- Creates bash scripts for running PySOC
- Combines results into Excel files

## 📁 Project Structure

```
Quantum-Chemistry-Software-Input-Generator/
├── quantum_steps_gui.py          # Main desktop GUI application
├── quantum_steps_web.py          # Flask web server
├── gaussian_steps_gui.py         # Gaussian backend logic
├── ai_assistant.py               # AI assistant integration
├── ai_file_generator.py          # AI-driven file generation
├── tict_rotation.py              # TICT rotation module
├── templates/                    # Web UI templates
│   └── index.html
├── static/                       # Web static files
│   ├── css/style.css
│   └── js/app.js
├── requirements.txt              # Desktop dependencies
├── requirements_web.txt          # Web dependencies
└── README.md                     # This file
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
- **Ollama**: For local AI capabilities
- **Google Gemini**: For cloud AI capabilities

## 📧 Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**Made with ❤️ for the quantum chemistry community**
