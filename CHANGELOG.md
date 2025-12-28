# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Web-based version with Flask backend
- iOS Safari and mobile browser support
- API endpoints for programmatic access
- Deployment guide for cloud hosting
- Comprehensive README with examples
- Contributing guidelines

## [2.0.0] - 2025-01-XX

### Added
- Unified application supporting both Gaussian and ORCA
- Software selector in GUI
- Dynamic UI that adapts to selected software
- SMILES input support with ChemDraw SVG name extraction
- Log file parsing for Gaussian (.log files)
- PySOC integration for Spin-Orbit Coupling calculations
- Triplet state support for TD-DFT
- Multiple input types: .com, .log, .xyz, SMILES
- Performance optimizations (debouncing, caching)
- Help sections with workflow guides
- Reset to Defaults button
- Auto-open output folder after generation

### Changed
- Refactored codebase for better modularity
- Improved GUI consistency between Gaussian and ORCA
- Enhanced scrolling support across all tabs
- Updated default settings (NPROC=64, MEM=128GB)

### Fixed
- SVG name extraction AttributeError
- Prefix/suffix removal for log files
- Route preview updates
- Widget recreation issues when switching software
- PySOC script generation
- Empty script generation issues

## [1.0.0] - Initial Release

### Added
- Basic Gaussian step maker
- GUI interface
- Support for steps 1-7
- PBS/SLURM/Local scheduler support
- Geometry chaining
- Solvent models

---

[Unreleased]: https://github.com/abedisyedaliabbas/Quantum-Chemistry-Software-Input-Generator/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/abedisyedaliabbas/Quantum-Chemistry-Software-Input-Generator/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/abedisyedaliabbas/Quantum-Chemistry-Software-Input-Generator/releases/tag/v1.0.0


