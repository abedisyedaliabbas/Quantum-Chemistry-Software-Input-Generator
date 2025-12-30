// Quantum Chemical Calculations Step Maker - Web App JavaScript

class QuantumStepsApp {
    constructor() {
        this.software = 'gaussian';
        this.config = {};
        this.files = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadDefaults();
    }

    setupEventListeners() {
        // Software selection
        document.querySelectorAll('input[name="software"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.software = e.target.value;
                this.updateSoftwareStatus();
                this.loadDefaults();
            });
        });

        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tabName = e.target.dataset.tab;
                this.switchTab(tabName);
            });
        });

        // Mode selection
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const mode = e.target.dataset.mode;
                this.setMode(mode);
            });
        });

        // Step selection
        document.querySelectorAll('.step-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const step = parseInt(e.target.dataset.step);
                this.setStep(step);
            });
        });

        // Input type selection
        document.querySelectorAll('input[name="input-type"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.onInputTypeChange(e.target.value);
            });
        });

        // File input
        document.getElementById('file-input').addEventListener('change', (e) => {
            this.handleFileInput(e.target.files);
        });

        // SMILES processing
        document.getElementById('smiles-text').addEventListener('input', (e) => {
            this.processSMILES(e.target.value);
        });

        // Buttons
        document.getElementById('preview-btn').addEventListener('click', () => {
            this.previewFiles();
        });

        document.getElementById('generate-btn').addEventListener('click', () => {
            this.generateFiles();
        });

        document.getElementById('load-svg-btn').addEventListener('click', () => {
            this.loadNamesFromSVG();
        });

        document.getElementById('chemdraw-help-btn').addEventListener('click', () => {
            this.showChemDrawHelp();
        });

        // Config updates
        this.setupConfigListeners();
    }

    setupConfigListeners() {
        // Method & Basis
        document.getElementById('functional-select').addEventListener('change', (e) => {
            this.config.FUNCTIONAL = e.target.value;
        });

        document.getElementById('basis-select').addEventListener('change', (e) => {
            this.config.BASIS = e.target.value;
        });

        // Solvent
        document.getElementById('solvent-model-select').addEventListener('change', (e) => {
            this.config.SOLVENT_MODEL = e.target.value;
        });

        document.getElementById('solvent-name-select').addEventListener('change', (e) => {
            this.config.SOLVENT_NAME = e.target.value;
        });

        // TD-DFT
        document.querySelectorAll('input[name="state-type"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.config.STATE_TYPE = e.target.value;
            });
        });

        document.getElementById('td-nstates').addEventListener('change', (e) => {
            this.config.TD_NSTATES = parseInt(e.target.value);
        });

        document.getElementById('td-root').addEventListener('change', (e) => {
            this.config.TD_ROOT = parseInt(e.target.value);
        });

        // Resources
        document.getElementById('nproc-input').addEventListener('change', (e) => {
            this.config.NPROC = parseInt(e.target.value);
        });

        document.getElementById('mem-input').addEventListener('change', (e) => {
            this.config.MEM = e.target.value;
        });

        // Advanced
        document.getElementById('pop-full').addEventListener('change', (e) => {
            this.config.POP_FULL = e.target.checked;
        });

        document.getElementById('dispersion').addEventListener('change', (e) => {
            this.config.DISPERSION = e.target.checked;
        });

        document.getElementById('soc-enable').addEventListener('change', (e) => {
            this.config.SOC_ENABLE = e.target.checked;
        });
    }

    updateSoftwareStatus() {
        const statusEl = document.getElementById('software-status');
        const softwareName = this.software.charAt(0).toUpperCase() + this.software.slice(1);
        statusEl.textContent = `✔ ${softwareName} support active`;
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab panes
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.remove('active');
        });
        document.getElementById(`tab-${tabName}`).classList.add('active');
    }

    setMode(mode) {
        this.config.MODE = mode;
        
        // Update UI
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-mode="${mode}"]`).classList.add('active');

        // Show/hide step selection
        const stepSelection = document.getElementById('step-selection');
        if (mode === 'full') {
            stepSelection.style.display = 'none';
        } else {
            stepSelection.style.display = 'block';
        }
    }

    setStep(step) {
        this.config.STEP = step;
        
        // Update UI
        document.querySelectorAll('.step-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-step="${step}"]`).classList.add('active');
    }

    onInputTypeChange(type) {
        const fileGroup = document.getElementById('file-input-group');
        const smilesGroup = document.getElementById('smiles-input-group');

        if (type === 'smiles') {
            fileGroup.style.display = 'none';
            smilesGroup.style.display = 'block';
        } else {
            fileGroup.style.display = 'block';
            smilesGroup.style.display = 'none';
        }
    }

    async loadDefaults() {
        try {
            const response = await fetch(`/api/defaults/${this.software}`);
            const defaults = await response.json();
            
            // Merge with current config
            this.config = { ...defaults, ...this.config };
            
            // Update UI with defaults
            this.updateUIFromConfig();
        } catch (error) {
            console.error('Error loading defaults:', error);
        }
    }

    updateUIFromConfig() {
        // Update form fields from config
        if (this.config.FUNCTIONAL) {
            const funcSelect = document.getElementById('functional-select');
            if (funcSelect) funcSelect.value = this.config.FUNCTIONAL;
        }
        
        if (this.config.BASIS) {
            const basisSelect = document.getElementById('basis-select');
            if (basisSelect) basisSelect.value = this.config.BASIS;
        }

        if (this.config.MODE) {
            this.setMode(this.config.MODE);
        }

        if (this.config.STEP) {
            this.setStep(this.config.STEP);
        }
    }

    async handleFileInput(files) {
        this.files = [];
        
        for (let file of files) {
            const text = await file.text();
            
            if (file.name.endsWith('.com')) {
                // Parse .com file
                const parsed = this.parseComFile(text);
                this.files.push(parsed);
            } else if (file.name.endsWith('.log')) {
                // Parse .log file
                const response = await fetch('/api/parse-log', {
                    method: 'POST',
                    body: this.createFormData({ file: file })
                });
                const result = await response.json();
                if (!result.error) {
                    this.files.push({
                        name: file.name.replace(/\.(com|log)$/, ''),
                        coords: result.coords,
                        charge: result.charge,
                        mult: result.mult
                    });
                }
            }
        }
        
        this.showMessage(`Loaded ${this.files.length} file(s)`, 'success');
    }

    parseComFile(text) {
        // Simple .com parser - extract charge, mult, and coordinates
        const lines = text.split('\n');
        let charge = 0, mult = 1;
        let coords = [];
        let inCoords = false;
        let name = 'molecule';

        for (let line of lines) {
            line = line.trim();
            if (!line) continue;

            // Extract charge and multiplicity
            const cmMatch = line.match(/^(\d+)\s+(\d+)$/);
            if (cmMatch && !inCoords) {
                charge = parseInt(cmMatch[1]);
                mult = parseInt(cmMatch[2]);
                inCoords = true;
                continue;
            }

            // Extract coordinates
            if (inCoords) {
                const coordMatch = line.match(/^(\w+)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)$/);
                if (coordMatch) {
                    coords.push(line);
                } else if (coords.length > 0) {
                    break; // End of coordinates
                }
            }
        }

        return { name, coords, charge, mult };
    }

    async processSMILES(smilesText) {
        if (!smilesText.trim()) return;

        const charge = parseInt(document.getElementById('charge-input').value) || 0;
        const mult = parseInt(document.getElementById('mult-input').value) || 1;

        try {
            const response = await fetch('/api/smiles-to-coords', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ smiles: smilesText, charge, mult })
            });

            const result = await response.json();
            if (!result.error) {
                this.files = result.coords;
                this.showMessage(`Processed ${this.files.length} SMILES`, 'success');
            } else {
                this.showMessage(`Error: ${result.error}`, 'error');
            }
        } catch (error) {
            this.showMessage(`Error processing SMILES: ${error.message}`, 'error');
        }
    }

    async previewFiles() {
        if (this.files.length === 0) {
            this.showMessage('Please load files or enter SMILES first', 'error');
            return;
        }

        const previewArea = document.getElementById('preview-content');
        previewArea.innerHTML = '<div class="loading"></div> Loading preview...';

        try {
            const file = this.files[0]; // Preview first file
            const response = await fetch('/api/preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    software: this.software,
                    config: this.config,
                    file: file
                })
            });

            const result = await response.json();
            if (!result.error) {
                this.displayPreview(result.previews);
            } else {
                previewArea.innerHTML = `<div class="message error">Error: ${result.error}</div>`;
            }
        } catch (error) {
            previewArea.innerHTML = `<div class="message error">Error: ${error.message}</div>`;
        }
    }

    displayPreview(previews) {
        const previewArea = document.getElementById('preview-content');
        let html = '';

        previews.forEach(preview => {
            html += `
                <div class="preview-item">
                    <h3>${preview.filename}</h3>
                    <pre>${preview.content}</pre>
                </div>
            `;
        });

        previewArea.innerHTML = html || '<div class="message">No preview available</div>';
    }

    async generateFiles() {
        if (this.files.length === 0) {
            this.showMessage('Please load files or enter SMILES first', 'error');
            return;
        }

        this.showMessage('Generating files...', 'success');

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    software: this.software,
                    config: this.config,
                    files: this.files
                })
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${this.software}_jobs.zip`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                this.showMessage('Files generated successfully!', 'success');
            } else {
                const error = await response.json();
                this.showMessage(`Error: ${error.error}`, 'error');
            }
        } catch (error) {
            this.showMessage(`Error: ${error.message}`, 'error');
        }
    }

    async loadNamesFromSVG() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.svg';
        input.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/api/extract-names-from-svg', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                if (!result.error && result.names.length > 0) {
                    // Match names with SMILES
                    const smilesText = document.getElementById('smiles-text').value;
                    this.matchNamesWithSMILES(result.names, smilesText);
                    this.showMessage(`Loaded ${result.names.length} names from SVG`, 'success');
                } else {
                    this.showMessage('No names found in SVG', 'error');
                }
            } catch (error) {
                this.showMessage(`Error: ${error.message}`, 'error');
            }
        };
        input.click();
    }

    matchNamesWithSMILES(names, smilesText) {
        const lines = smilesText.split('\n');
        const matched = [];

        lines.forEach((line, idx) => {
            const trimmed = line.trim();
            if (!trimmed || trimmed.startsWith('#')) return;

            let name = names[idx] || `molecule_${idx + 1}`;
            let smiles = trimmed;

            if (trimmed.includes(':')) {
                const parts = trimmed.split(':');
                name = parts[0].trim();
                smiles = parts.slice(1).join(':').trim();
            }

            matched.push(`${name}:${smiles}`);
        });

        document.getElementById('smiles-text').value = matched.join('\n');
    }

    showChemDrawHelp() {
        alert(`How to Use SMILES with Names from ChemDraw:

STEP 1 - Copy SMILES:
1. In ChemDraw, select ALL structures (Ctrl+A or drag select)
2. Go to: Edit → Copy As → SMILES (or Alt+Ctrl+C)
3. Paste into the SMILES Input box above

STEP 2 - Get Names from SVG:
1. In ChemDraw: File → Save As → SVG format
2. Click "Load Names from SVG" button above
3. Select your SVG file
4. Names will be automatically matched with your SMILES!

ALTERNATIVE - Manual Format:
You can also type manually:
  • Just SMILES: CCO
  • With name: FLIMBD_1:CCO
  • Tab-separated: FLIMBD_1<TAB>CCO
  • Comments: # This is a comment`);
    }

    showMessage(text, type = 'success') {
        // Remove existing messages
        const existing = document.querySelector('.message');
        if (existing) existing.remove();

        // Create new message
        const message = document.createElement('div');
        message.className = `message ${type}`;
        message.textContent = text;
        
        // Insert at top of current tab
        const tabPane = document.querySelector('.tab-pane.active');
        tabPane.insertBefore(message, tabPane.firstChild);

        // Auto-remove after 5 seconds
        setTimeout(() => message.remove(), 5000);
    }

    createFormData(data) {
        const formData = new FormData();
        Object.keys(data).forEach(key => {
            formData.append(key, data[key]);
        });
        return formData;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new QuantumStepsApp();
});



