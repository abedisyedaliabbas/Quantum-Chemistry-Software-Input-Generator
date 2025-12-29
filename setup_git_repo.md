# Git Setup Instructions

Follow these steps to push your code to GitHub:

## Step 1: Initialize Git Repository

```bash
# Initialize git repository
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: Quantum Chemical Calculations Step Maker v2.0

- Unified GUI supporting Gaussian and ORCA
- Web-based version with Flask backend
- SMILES input with ChemDraw SVG support
- Log file parsing
- PySOC integration
- Performance optimizations
- Comprehensive documentation"
```

## Step 2: Connect to GitHub Repository

```bash
# Add your GitHub repository as remote
git remote add origin https://github.com/abedisyedaliabbas/Quantum-Chemistry-Software-Input-Generator.git

# Verify remote was added
git remote -v
```

## Step 3: Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

## Alternative: If Repository Already Exists on GitHub

If your repository already has files, you may need to pull first:

```bash
# Pull existing files (if any)
git pull origin main --allow-unrelated-histories

# Resolve any conflicts, then:
git add .
git commit -m "Merge with existing repository"

# Push
git push -u origin main
```

## Step 4: Update Repository Settings on GitHub

1. Go to: https://github.com/abedisyedaliabbas/Quantum-Chemistry-Software-Input-Generator/settings
2. Scroll to "Features" section
3. Enable:
   - ✅ Issues
   - ✅ Discussions (optional)
   - ✅ Wiki (optional)
4. Scroll to "General" → "Repository visibility"
5. Make sure it's set to **Public** (for open source)

## Step 5: Update Repository Description

Go to repository settings → General → and update:

**Description:**
```
Python utility to auto-generate Gaussian and ORCA input files & job submission scripts (PBS/SLURM/local) for ground/excited-state workflows. Available as desktop GUI and web application. Supports SMILES, log file parsing, PySOC integration, and more.
```

**Topics (add these):**
- quantum-chemistry
- gaussian
- orca
- computational-chemistry
- python
- gui
- flask
- quantum-mechanics
- dft
- tddft

## Step 6: Create a Release (Optional but Recommended)

1. Go to: https://github.com/abedisyedaliabbas/Quantum-Chemistry-Software-Input-Generator/releases/new
2. Tag: `v2.0.0`
3. Title: `Version 2.0.0 - Unified GUI with Web Support`
4. Description: Copy from CHANGELOG.md
5. Publish release

## Files to Include

✅ **Include:**
- `quantum_steps_gui.py` - Main desktop GUI
- `quantum_steps_web.py` - Web server
- `gaussian_steps_gui.py` - Gaussian backend
- `templates/` - Web UI templates
- `static/` - Web static files
- `README.md` - Main documentation
- `LICENSE` - MIT License
- `CONTRIBUTING.md` - Contribution guidelines
- `CHANGELOG.md` - Version history
- `DEPLOYMENT_GUIDE.md` - Web deployment guide
- `requirements.txt` - Desktop dependencies
- `requirements_web.txt` - Web dependencies
- `.gitignore` - Git ignore rules

❌ **Exclude (already in .gitignore):**
- `__pycache__/`
- `build/`
- `dist/`
- `*.exe`
- `*.spec`
- Temporary files

## Quick Commands Summary

```bash
# Complete setup in one go
git init
git add .
git commit -m "Initial commit: Quantum Chemical Calculations Step Maker v2.0"
git remote add origin https://github.com/abedisyedaliabbas/Quantum-Chemistry-Software-Input-Generator.git
git branch -M main
git push -u origin main
```

## Troubleshooting

**If you get "remote origin already exists":**
```bash
git remote remove origin
git remote add origin https://github.com/abedisyedaliabbas/Quantum-Chemistry-Software-Input-Generator.git
```

**If you get authentication errors:**
- Use GitHub Personal Access Token instead of password
- Or use SSH: `git@github.com:abedisyedaliabbas/Quantum-Chemistry-Software-Input-Generator.git`

**If files are too large:**
- Check `.gitignore` is working
- Remove large files manually if needed


