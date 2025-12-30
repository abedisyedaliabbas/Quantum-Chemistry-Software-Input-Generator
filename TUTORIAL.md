# Quantum Chemistry Input Generator: Comprehensive Tutorial and Documentation

**Version 2.0** | **Last Updated: January 2025**

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Getting Started](#getting-started)
4. [Core Features](#core-features)
5. [Gaussian Workflow](#gaussian-workflow)
6. [ORCA Workflow](#orca-workflow)
7. [Input Types and Formats](#input-types-and-formats)
8. [Advanced Features](#advanced-features)
9. [AI Assistant](#ai-assistant)
10. [TICT Rotation and Dihedral Scans](#tict-rotation-and-dihedral-scans)
11. [PySOC Integration](#pysoc-integration)
12. [Best Practices](#best-practices)
13. [Troubleshooting](#troubleshooting)
14. [Examples and Case Studies](#examples-and-case-studies)

---

## 1. Introduction

### 1.1 Overview

The Quantum Chemistry Input Generator is a comprehensive software tool designed to automate the creation of input files for quantum chemistry calculations using Gaussian and ORCA software packages. This application streamlines the workflow for computational chemists by providing an intuitive graphical interface that eliminates manual file preparation and reduces the potential for errors.

### 1.2 Key Benefits

- **Time Efficiency**: Reduces input file preparation time from hours to minutes
- **Error Reduction**: Automated generation minimizes manual transcription errors
- **Workflow Automation**: Supports complete multi-step computational workflows
- **Flexibility**: Handles multiple input formats and calculation types
- **Accessibility**: Available as both desktop GUI and web-based interface

### 1.3 Target Audience

This software is designed for:
- Computational chemists performing quantum chemistry calculations
- Researchers working with excited states and photochemistry
- Students learning quantum chemistry workflows
- Groups requiring batch processing of multiple molecules
- Users needing automated workflow generation

---

## 2. Installation and Setup

### 2.1 System Requirements

**Minimum Requirements:**
- Python 3.8 or higher
- 4 GB RAM
- 500 MB free disk space
- Windows 10/11, macOS 10.14+, or Linux

**Recommended:**
- Python 3.10 or higher
- 8 GB RAM
- 1 GB free disk space

### 2.2 Installation Steps

#### Step 1: Clone the Repository

```bash
git clone https://github.com/abedisyedaliabbas/Quantum-Chemistry-Software-Input-Generator.git
cd Quantum-Chemistry-Software-Input-Generator
```

#### Step 2: Install Dependencies

**For Desktop GUI:**
```bash
pip install -r requirements.txt
```

**For Web Version:**
```bash
pip install -r requirements_web.txt
```

#### Step 3: Install RDKit (Optional but Recommended)

RDKit is required for SMILES string processing and 3D coordinate generation.

**Using Conda (Recommended):**
```bash
conda install -c conda-forge rdkit
```

**Using pip:**
```bash
pip install rdkit-pypi
```

**Note**: RDKit installation via pip may have limitations on Windows. Conda installation is strongly recommended.

#### Step 4: Verify Installation

Launch the application:
```bash
python quantum_steps_gui.py
```

If the GUI opens successfully, installation is complete.

### 2.3 Optional: AI Assistant Setup

#### Ollama (Free, Local AI)

1. Download Ollama from https://ollama.com
2. Install and run Ollama
3. Pull a model:
   ```bash
   ollama pull llama3.1:8b
   ```
4. The application will automatically detect Ollama

#### Google Gemini (Cloud AI)

1. Obtain API key from https://makersuite.google.com/app/apikey
2. Enter the key in the AI Assistant tab
3. The key is saved locally for future use

---

## 3. Getting Started

### 3.1 First Launch

Upon launching the application, you'll see:

1. **Software Selector**: Choose between Gaussian and ORCA
2. **Tabbed Interface**: Navigate through different configuration sections
3. **Status Bar**: Shows current operation status

### 3.2 Basic Workflow

1. **Select Software**: Choose Gaussian or ORCA from the main interface
2. **Choose Input Type**: Select `.com`, `.log`, `.xyz`, or SMILES
3. **Configure Settings**: Navigate through tabs to set parameters
4. **Preview**: Review generated files before creation
5. **Generate**: Create all input files and scripts

### 3.3 Interface Overview

The interface is organized into logical sections:

- **Mode & IO**: Input/output configuration
- **Method & Basis**: Computational method selection
- **Solvent**: Solvation model settings
- **TD-DFT**: Excited state parameters
- **Resources**: Computational resources
- **Scheduler**: Job submission settings
- **Advanced**: Custom routes and options
- **TICT Rotation**: Dihedral scan generation
- **AI Assistant**: Natural language interface
- **Generate**: File creation and preview

---

## 4. Core Features

### 4.1 Software Support

#### Gaussian (G16/G09)

Full support for Gaussian workflows including:
- All standard calculation types
- Geometry optimization
- Frequency calculations
- TD-DFT calculations
- Solvent models (PCM, SMD)
- Custom route cards

#### ORCA

Complete ORCA input file generation:
- Method and basis set selection
- TD-DFT with TDA option
- Solvent models (SMD, CPCM)
- Custom step definitions
- Resource management

### 4.2 Input Modes

#### Full Mode
Generates all workflow steps automatically:
- **Gaussian**: Steps 1-7 (complete workflow)
- **ORCA**: Steps 1, 2, 4, 7, 9

#### Single Mode
Generate one specific step:
- Select the step number
- Useful for re-running specific calculations
- Supports geometry chaining

#### Multiple Mode
Select specific steps to generate:
- Check boxes for desired steps
- Flexible workflow customization
- Efficient for partial workflows

### 4.3 Input File Types

#### .com Files (Gaussian Input)
- Standard Gaussian input format
- Automatically extracts:
  - Geometry coordinates
  - Charge and multiplicity
  - Route card information

#### .log Files (Gaussian Output)
- Extracts final optimized geometry
- Reads charge and multiplicity
- Useful for continuing calculations
- Supports multiple geometries in one file

#### .xyz Files
- Standard XYZ coordinate format
- Simple and universal format
- Easy to generate from other software

#### SMILES Strings
- Chemical structure notation
- Automatic 3D coordinate generation
- Supports ChemDraw integration
- Format: `name:SMILES` or `SMILES`

---

## 5. Gaussian Workflow

### 5.1 Complete Workflow Steps

The standard Gaussian workflow consists of seven steps:

#### Step 1: Ground State Geometry Optimization
```
# method/basis Opt Freq SCRF=(solvent)
```
- Optimizes ground state geometry
- Calculates vibrational frequencies
- Includes solvent effects if specified
- Generates checkpoint file for subsequent steps

#### Step 2: Vertical Excitation (Franck-Condon)
```
# method/basis TD(NStates=n) SCRF=(solvent)
```
- Calculates vertical excitation energy
- Uses optimized ground state geometry
- TD-DFT calculation
- Determines absorption maximum

#### Step 3: cLR Correction of Vertical Excitation
```
# method/basis TD(NStates=n) SCRF=(CorrectedLR,solvent)
```
- Corrected Linear Response (cLR) method
- More accurate excitation energies
- Accounts for state-specific solvation

#### Step 4: Excited State Geometry Optimization
```
# method/basis TD(NStates=n) Opt=CalcFC Freq SCRF=(solvent)
```
- Optimizes excited state geometry
- Uses calculated force constants
- Generates excited state frequencies
- Creates checkpoint for emission calculations

#### Step 5: Density Calculation
```
# method/basis density SCRF=(solvent)
```
- Calculates electron density
- At optimized excited state geometry
- Required for emission calculations

#### Step 6: cLR Correction of Excited State
```
# method/basis TD(NStates=n) SCRF=(CorrectedLR,solvent)
```
- cLR correction at excited state geometry
- More accurate emission energies
- State-specific solvation effects

#### Step 7: Ground State at Excited State Geometry
```
# method/basis SCRF=(solvent) geom=check guess=read
```
- Ground state energy at excited state geometry
- Uses geometry from Step 4
- Calculates reorganization energy
- Completes energy cycle

### 5.2 Configuration Example

**Scenario**: Calculate absorption and emission of a molecule in water

1. **Input**: `.com` file with initial geometry
2. **Method**: B3LYP/6-31G*
3. **Solvent**: SMD, Water
4. **TD-DFT**: 5 states, Root 1
5. **Mode**: Full (Steps 1-7)

**Generated Files**:
- `01molecule_b3lyp_6-31G*_water.com` (Step 1)
- `02molecule_b3lyp_6-31G*_water.com` (Step 2)
- ... (Steps 3-7)
- Submission scripts (`.sh` files)
- Helper scripts for batch submission

### 5.3 Geometry Chaining

The application supports automatic geometry chaining:

- **Default**: Steps use checkpoint files from previous steps
- **Inline**: Specific steps can use inline coordinates
- **Custom**: Manual specification of geometry sources

**Example**: Step 7 always uses geometry from Step 6 checkpoint file.

---

## 6. ORCA Workflow

### 6.1 ORCA Workflow Steps

ORCA workflow consists of five main steps:

#### Step 1: Ground State Optimization
```
! method basis Opt
```
- Optimizes ground state geometry
- Uses selected method and basis set
- Includes solvent if specified

#### Step 2: Vertical Excitation
```
! method basis TDDFT NROOTS n IROOT i
```
- Calculates vertical excitation
- TD-DFT with TDA option available
- Root following option

#### Step 4: Excited State Optimization
```
! method basis TDDFT NROOTS n IROOT i Opt
```
- Optimizes excited state geometry
- Follows specific root if enabled
- Generates optimized excited state

#### Step 7: Ground State at Excited State Geometry
```
! method basis
```
- Ground state calculation
- Uses optimized excited state geometry
- Calculates reorganization energy

#### Step 9: Custom Step
```
! [user-defined keywords]
```
- Fully customizable step
- User-defined route card
- Flexible for special calculations

### 6.2 ORCA-Specific Features

#### Method Selection
- Wide range of functionals (M06-2X, B3LYP, wB97X-D, etc.)
- Basis sets (def2-TZVP, def2-SVP, cc-pVDZ, etc.)
- Custom method entry supported

#### TD-DFT Options
- **NROOTS**: Number of roots to calculate
- **IROOT**: Root of interest
- **TDA**: Tamm-Dancoff Approximation
- **FOLLOW_IROOT**: Root following in optimization

#### Solvent Models
- **SMD**: Solvation Model based on Density
- **CPCM**: Conductor-like Polarizable Continuum Model
- **EPSILON**: Custom dielectric constant

---

## 7. Input Types and Formats

### 7.1 .com Files (Gaussian Input)

**Format**:
```
%nprocshared=64
%mem=128GB
%chk=filename.chk

# method/basis [keywords]

Title

0 1
C    0.000000    0.000000    0.000000
H    1.089000    0.000000    0.000000
...
```

**Extraction**:
- Route card: First line starting with `#`
- Charge/Multiplicity: Line before coordinates
- Coordinates: Lines after charge/multiplicity

### 7.2 .log Files (Gaussian Output)

**Extraction Process**:
1. Finds "Standard orientation:" sections
2. Uses last occurrence (final geometry)
3. Extracts atomic numbers and coordinates
4. Converts to element symbols and Gaussian format

**Supported Formats**:
- Standard orientation
- Input orientation
- Final optimized geometry

### 7.3 .xyz Files

**Format**:
```
number_of_atoms
comment_line
element x y z
element x y z
...
```

**Processing**:
- Reads atom count
- Extracts coordinates
- Assigns default charge/multiplicity (0 1)

### 7.4 SMILES Strings

**Format Options**:
```
name:SMILES
SMILES
name\tSMILES
```

**Examples**:
```
formaldehyde:CO
benzene:c1ccccc1
water:O
acetone:CC(=O)C
```

**Processing**:
1. Parses SMILES string
2. Generates 3D coordinates using RDKit
3. Adds hydrogens
4. Optimizes geometry (MMFF)
5. Extracts charge and multiplicity

**ChemDraw Integration**:
1. Export structures as SVG from ChemDraw
2. Copy SMILES strings
3. Use "Load Names from SVG" button
4. Names automatically matched with SMILES

---

## 8. Advanced Features

### 8.1 Custom Route Cards

**Purpose**: Override auto-generated routes for specific steps

**Usage**:
1. Navigate to Advanced tab
2. Select step number
3. Enter custom route card
4. Route completely replaces auto-generated route

**Example**:
```
# B3LYP/6-31G* Opt=ModRedundant Freq SCRF=(SMD,Water)
```

**Best Practices**:
- Ensure route syntax is correct
- Include all required keywords
- Test routes before batch processing

### 8.2 Redundant Coordinates

**Purpose**: Add redundant internal coordinates for Step 4 optimization

**Format**:
```
B 1 2 F
A 1 2 3 F
D 1 2 3 4 F
```

**Where**:
- B = Bond length
- A = Bond angle
- D = Dihedral angle
- Numbers = Atom indices
- F = Freeze flag

**Application**: Only added to Step 4 (excited state optimization)

### 8.3 Geometry Source Selection

**Options**:
- **Default**: Automatic chaining via checkpoint files
- **coords_1**: Use coordinates from Step 1
- **coords_4**: Use coordinates from Step 4
- **oldchk_1**: Use checkpoint from Step 1
- **oldchk_4**: Use checkpoint from Step 4

**Use Cases**:
- Restarting failed calculations
- Testing different geometries
- Custom workflow requirements

### 8.4 Inline Steps

**Purpose**: Force specific steps to use inline coordinates

**Configuration**:
- Check boxes for steps to inline
- Coordinates copied from input file
- Useful for debugging or special cases

### 8.5 Bulk Processing

**Supported**:
- Folder paths: Process all matching files in directory
- Glob patterns: `*.com`, `molecule_*.log`, etc.
- Multiple files: Automatic batch processing

**Example**:
```
Input: C:\Calculations\Molecules\*.com
Output: All files processed automatically
```

---

## 9. AI Assistant

### 9.1 Overview

The AI Assistant provides a natural language interface for generating input files, eliminating the need to navigate through multiple GUI tabs.

### 9.2 Supported AI Models

#### Ollama (Recommended)
- **Free**: No API costs
- **Local**: Runs on your computer
- **Private**: Data stays local
- **Model**: llama3.1:8b (or similar)

#### Google Gemini
- **Cloud-based**: Requires internet
- **Free tier**: Limited requests
- **Models**: gemini-2.5-flash, gemini-1.5-pro, etc.

### 9.3 Usage Examples

#### Example 1: Simple TD-DFT Calculation
```
User: "I want to run a TD-DFT calculation on benzene using B3LYP/6-31G* in water"

AI: [Asks for input file or generates from SMILES]
    [Creates configuration]
    [Generates files]
```

#### Example 2: TICT Scan
```
User: "Generate a TICT scan for molecule.com, rotate dihedral 12-11-10-3 from 0 to 90 degrees in 10 steps"

AI: [Detects TICT scan request]
    [Calculates current dihedral angle]
    [Generates rotated geometries]
    [Creates input files for each step]
```

#### Example 3: Bulk Processing
```
User: "Process all .com files in C:\Molecules folder with WB97XD/def2TZVP in DMSO"

AI: [Finds all files]
    [Applies settings to each]
    [Generates batch of input files]
```

### 9.4 AI Capabilities

- **Parameter Detection**: Automatically determines calculation type
- **Angle Calculation**: Calculates current dihedral angles from input files
- **Bulk Processing**: Handles folders and glob patterns
- **Error Prevention**: Validates configurations before generation
- **Context Awareness**: Remembers previous conversation context

### 9.5 Best Practices

1. **Be Specific**: Provide clear instructions
2. **Include Details**: Method, basis set, solvent, etc.
3. **Specify Input**: File path, folder, or SMILES
4. **Review Output**: Always check generated files
5. **Iterate**: Refine requests based on results

---

## 10. TICT Rotation and Dihedral Scans

### 10.1 Overview

TICT (Twisted Intramolecular Charge Transfer) rotation generates geometries for torsional/dihedral angle scans, essential for studying molecular conformations and potential energy surfaces.

### 10.2 Advanced TICT Mode

**Required Parameters**:
- **Axis**: Two atoms defining rotation axis (e.g., "11 12")
- **Branch A**: Atoms in first rotating branch (e.g., "1 2 3 4 5")
- **Branch A Step**: Rotation increment for Branch A (degrees)
- **Branch B**: Atoms in second rotating branch (e.g., "13 14 15")
- **Branch B Step**: Rotation increment for Branch B (degrees)
- **Number of Steps**: Total number of geometries to generate

**Example**:
```
Axis: 11 12
Branch A: 1 2 3 4 5 6 7 8 9 10
Branch A Step: 5.0
Branch B: 13 14 15 16 17
Branch B Step: 0.0
Number of Steps: 18
```

This generates 18 geometries rotating Branch A by 5° increments (0° to 85°).

### 10.3 Rotation Mechanism

**Process**:
1. Identifies rotation axis (bond between two atoms)
2. Determines atoms in each branch
3. Calculates rotation matrix (Rodrigues' rotation formula)
4. Applies rotation to branch atoms
5. Generates new coordinate set
6. Creates input file for each geometry

**Key Features**:
- Preserves bond lengths
- Maintains molecular structure
- Prevents bond breaking
- Accurate dihedral angles

### 10.4 Use Cases

- **Conformational Analysis**: Study different conformers
- **Potential Energy Surfaces**: Map energy vs. dihedral angle
- **Reaction Pathways**: Explore rotational barriers
- **Photochemistry**: Study TICT states
- **Solvent Effects**: Compare gas phase vs. solution

### 10.5 Best Practices

1. **Identify Axis**: Choose the bond around which rotation occurs
2. **Define Branches**: Clearly identify atoms in each branch
3. **Step Size**: Balance between resolution and computational cost
4. **Validation**: Check first few geometries manually
5. **Naming**: Use descriptive output directory names

---

## 11. PySOC Integration

### 11.1 Overview

PySOC (Python Spin-Orbit Coupling) integration automates the preparation of calculations for spin-orbit coupling analysis, which is crucial for understanding intersystem crossing and phosphorescence.

### 11.2 Setup Requirements

**Gaussian Settings**:
- **State Type**: Mixed (50-50) or enable SOC checkbox
- **Route Card**: Automatically adds `6D 10F GFInput`
- **Checkpoint**: Saves RWF file (`%rwf=filename.rwf`)

**Why 50-50?**
- PySOC requires both singlet and triplet states
- 50-50 mode calculates both simultaneously
- More efficient than separate calculations

### 11.3 Workflow

#### Step 1: Generate Input Files
1. Enable "Prepare for PySOC" checkbox
2. Configure normal TD-DFT settings
3. Generate files (Steps 1-7)
4. Run Gaussian calculations
5. Ensure all jobs complete successfully

#### Step 2: Generate PySOC Scripts
1. Navigate to Generate tab
2. Click "Generate PySOC Scripts"
3. Select directory containing completed calculations
4. Scripts generated automatically

#### Step 3: Run PySOC
1. Execute generated PySOC scripts
2. PySOC processes RWF files
3. Calculates spin-orbit coupling matrix elements
4. Generates results files

#### Step 4: Analyze Results
1. Results combined into Excel files
2. SOC matrix elements tabulated
3. Intersystem crossing rates calculated
4. Phosphorescence lifetimes determined

### 11.4 Generated Files

**Gaussian Input Files**:
- Include `%rwf` line for RWF file saving
- Route contains `6D 10F GFInput` keywords
- Standard workflow steps (1-7)

**PySOC Scripts**:
- Bash scripts for running PySOC
- Automatic RWF file detection
- Result file organization
- Excel file generation

### 11.5 Best Practices

1. **Complete Workflow**: Run all steps before PySOC
2. **Check RWF Files**: Ensure RWF files are generated
3. **Resource Management**: PySOC can be memory-intensive
4. **Validation**: Compare with literature values
5. **Documentation**: Keep track of settings used

---

## 12. Best Practices

### 12.1 File Organization

**Recommended Structure**:
```
Project/
├── Input/
│   ├── molecules.com
│   └── geometries.xyz
├── Output/
│   ├── Step1/
│   ├── Step2/
│   └── ...
└── Results/
    ├── log_files/
    └── analysis/
```

### 12.2 Naming Conventions

**Input Files**:
- Use descriptive names: `molecule_functional_basis.com`
- Include method information
- Avoid special characters

**Output Files**:
- Auto-generated names include step number
- Format: `01molecule_method_basis_solvent.com`
- Consistent and searchable

### 12.3 Computational Resources

**CPU Cores**:
- Match to available resources
- Consider parallel efficiency
- Typical: 32-64 cores for medium molecules

**Memory**:
- Estimate based on molecule size
- Basis set dependent
- Typical: 64-128 GB for standard calculations

**Walltime**:
- Estimate based on previous calculations
- Add buffer for convergence issues
- Typical: 24-48 hours for full workflow

### 12.4 Quality Control

**Before Generation**:
1. Review all settings
2. Check input files
3. Verify paths and directories
4. Preview generated routes

**After Generation**:
1. Spot-check generated files
2. Verify route cards
3. Check file names
4. Validate geometry extraction

**During Calculation**:
1. Monitor job status
2. Check for convergence
3. Review log files
4. Adjust if necessary

### 12.5 Workflow Optimization

**Efficiency Tips**:
- Use geometry chaining when possible
- Batch process similar molecules
- Reuse optimized geometries
- Archive successful calculations

**Error Prevention**:
- Validate input files
- Test with small examples first
- Keep backups of working configurations
- Document custom settings

---

## 13. Troubleshooting

### 13.1 Common Issues

#### Issue: Files Not Found
**Symptoms**: "No .com files found" error
**Solutions**:
- Check file path is correct
- Verify file extensions (.com, .log, .xyz)
- Ensure files are in specified directory
- Check for hidden files or permissions

#### Issue: Geometry Extraction Fails
**Symptoms**: "No coordinates found" error
**Solutions**:
- Verify file format is correct
- Check for empty lines in coordinate section
- Ensure charge/multiplicity line is present
- Try different input file format

#### Issue: Route Card Errors
**Symptoms**: Gaussian rejects input file
**Solutions**:
- Review route card syntax
- Check method/basis set compatibility
- Verify keyword spelling
- Test route in Gaussian separately

#### Issue: Widget Values Not Updating
**Symptoms**: Generated files use old values
**Solutions**:
- Click in widget to ensure focus
- Type value and press Enter
- Use dropdown to select from list
- Check status message for current values

### 13.2 File Reading Issues

**Windows Paths**:
- Use forward slashes or double backslashes
- Avoid special characters in paths
- Keep paths reasonably short

**File Encoding**:
- Use UTF-8 encoding
- Avoid special characters in file names
- Check for BOM markers

### 13.3 AI Assistant Issues

#### Ollama Not Detected
**Solutions**:
- Verify Ollama is running
- Check `ollama list` shows models
- Restart application
- Check firewall settings

#### Gemini API Errors
**Solutions**:
- Verify API key is correct
- Check API quota limits
- Try different model
- Use Ollama as fallback

### 13.4 Performance Issues

**Slow GUI Response**:
- Reduce number of files in preview
- Disable route caching temporarily
- Close other applications
- Check system resources

**Large File Processing**:
- Process in smaller batches
- Use more efficient input formats
- Consider command-line alternatives
- Optimize file reading

---

## 14. Examples and Case Studies

### 14.1 Example 1: Simple Absorption Calculation

**Goal**: Calculate vertical absorption of formaldehyde in water

**Steps**:
1. Create input file: `formaldehyde.com` with geometry
2. Select Gaussian software
3. Input type: `.com` files
4. Method: B3LYP
5. Basis: 6-31G*
6. Solvent: SMD, Water
7. Mode: Single, Step 2
8. Generate files

**Result**: `02formaldehyde_b3lyp_6-31G*_water.com` with TD-DFT route

### 14.2 Example 2: Complete Workflow

**Goal**: Full photophysical characterization of a fluorophore

**Steps**:
1. Input: SMILES string or .com file
2. Method: wB97XD
3. Basis: def2-TZVP
4. Solvent: SMD, Acetonitrile
5. TD-DFT: 10 states, Root 1
6. Mode: Full (Steps 1-7)
7. Generate all files

**Result**: Complete set of 7 input files for full workflow

### 14.3 Example 3: TICT Scan

**Goal**: Study rotation barrier of a TICT molecule

**Steps**:
1. Input: Optimized geometry file
2. Navigate to TICT Rotation tab
3. Define axis: atoms 11-12
4. Define branches: atoms 1-10 (Branch A), 13-17 (Branch B)
5. Set rotation: 5° increments, 36 steps (0-180°)
6. Generate rotated geometries

**Result**: 36 input files with rotated geometries

### 14.4 Example 4: Bulk Processing

**Goal**: Process 50 molecules with same settings

**Steps**:
1. Place all .com files in folder
2. Input: Folder path or `*.com` pattern
3. Configure settings once
4. Generate all files

**Result**: All 50 molecules processed automatically

### 14.5 Example 5: AI-Assisted Generation

**Goal**: Generate input files using natural language

**Conversation**:
```
User: "I need to calculate the emission energy of coumarin in methanol using CAM-B3LYP/cc-pVDZ"

AI: "I'll help you set up an emission calculation. For emission, we need to:
1. Optimize the excited state (Step 4)
2. Calculate emission from optimized geometry (Step 6)
3. Calculate ground state at excited geometry (Step 7)

Please provide the input file path or SMILES string."

User: "C:\Molecules\coumarin.com"

AI: [Generates configuration and files]
    "Generated files for coumarin emission calculation:
    - Step 4: Excited state optimization
    - Step 6: Emission energy with cLR
    - Step 7: Ground state at excited geometry
    Files saved to: C:\Molecules\Output\"
```

---

## Appendix A: Quick Reference

### A.1 Keyboard Shortcuts

- **Tab**: Navigate between fields
- **Enter**: Confirm selection
- **Ctrl+C**: Copy (in text fields)
- **Ctrl+V**: Paste (in text fields)

### A.2 File Naming Convention

**Format**: `[step][base]_[method]_[basis]_[solvent].[ext]`

**Example**: `01molecule_wb97xd_def2tzvp_dmso.com`

### A.3 Default Settings

**Gaussian**:
- Functional: m062x
- Basis: def2SVP
- Solvent: None
- NPROC: 64
- MEM: 128GB

**ORCA**:
- Method: m06-2x
- Basis: def2-TZVP
- Solvent: None
- NPROCS: 8
- MAXCORE: 2000 MB

---

## Appendix B: Glossary

- **cLR**: Corrected Linear Response method
- **TICT**: Twisted Intramolecular Charge Transfer
- **TD-DFT**: Time-Dependent Density Functional Theory
- **PCM**: Polarizable Continuum Model
- **SMD**: Solvation Model based on Density
- **SOC**: Spin-Orbit Coupling
- **RWF**: Read-Write File (Gaussian checkpoint format)
- **SMILES**: Simplified Molecular Input Line Entry System

---

## Appendix C: References

### Software
- Gaussian: https://gaussian.com
- ORCA: https://orcaforum.kofo.mpg.de
- RDKit: https://www.rdkit.org
- PySOC: See Gaussian documentation

### Documentation
- Gaussian User's Reference
- ORCA Manual
- This tutorial and application documentation

---

## Version History

**Version 2.0** (January 2025)
- Unified Gaussian and ORCA support
- AI Assistant integration
- TICT rotation module
- Web-based interface
- Enhanced file reading capabilities
- Improved widget value handling

**Version 1.0** (Initial Release)
- Basic Gaussian step maker
- GUI interface
- Standard workflow support

---

**Document Prepared By**: Quantum Chemistry Input Generator Development Team  
**For Support**: Open an issue on GitHub  
**License**: MIT License

---

*This tutorial is continuously updated. Please refer to the latest version for the most current information.*


