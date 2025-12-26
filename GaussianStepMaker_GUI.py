"""
Syed Ali Abbas Abedi

"""

import tkinter as tk
from tkinter import filedialog, ttk
import sys
import GaussianStepMaker as script # Replace 'your_script_name' with the actual file name

class GaussianGUI:
    def __init__(self, master):
        self.master = master
        master.title("Gaussian Job Generator")

        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        style = ttk.Style()
        style.theme_use('clam')

        # --- Variables to hold settings ---
        self.mode = tk.StringVar(value=script.MODE)
        self.step = tk.IntVar(value=script.STEP)
        self.inputs = tk.StringVar(value=script.INPUTS)
        self.out_dir = tk.StringVar(value=script.OUT_DIR)
        
        self.functional = tk.StringVar(value=script.FUNCTIONAL)
        self.basis = tk.StringVar(value=script.BASIS)
        
        self.solvent_model = tk.StringVar(value=script.SOLVENT_MODEL)
        self.solvent_name = tk.StringVar(value=script.SOLVENT_NAME)
        
        self.td_nstates = tk.IntVar(value=script.TD_NSTATES)
        self.td_root = tk.IntVar(value=script.TD_ROOT)
        self.pop_full = tk.BooleanVar(value=script.POP_FULL)
        self.dispersion = tk.BooleanVar(value=script.DISPERSION)
        
        self.nproc = tk.IntVar(value=script.NPROC)
        self.mem = tk.StringVar(value=script.MEM)
        self.scheduler = tk.StringVar(value=script.SCHEDULER)
        self.queue = tk.StringVar(value=script.QUEUE)
        self.walltime = tk.StringVar(value=script.WALLTIME)
        self.project = tk.StringVar(value=script.PROJECT)
        self.account = tk.StringVar(value=script.ACCOUNT)
        
        # Geometry-specific variables
        self.inline_step4 = tk.BooleanVar(value=4 in script.INLINE_STEPS) # New boolean variable
        self.inline_source = tk.IntVar(value=script.INLINE_SOURCE_5TO7)
        self.charge = tk.StringVar(value="")
        self.multiplicity = tk.StringVar(value="")
        
        # --- Create widgets and lay them out ---
        self.create_file_frame(main_frame)
        self.create_calc_settings_frame(main_frame)
        self.create_hpc_settings_frame(main_frame)
        self.create_adv_settings_frame(main_frame)
        self.create_action_frame(main_frame)
        
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

    def create_file_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="File & Mode Selection", padding="10")
        frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        
        ttk.Label(frame, text="Mode:").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Radiobutton(frame, text="Single Step", variable=self.mode, value="single", command=self.update_ui_state).grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(frame, text="Full Steps", variable=self.mode, value="full", command=self.update_ui_state).grid(row=0, column=2, sticky="w")
        
        ttk.Label(frame, text="Step:").grid(row=1, column=0, sticky="w", padx=5)
        self.step_entry = ttk.Combobox(frame, textvariable=self.step, values=[1, 2, 3, 4, 5, 6, 7])
        self.step_entry.grid(row=1, column=1, sticky="w")

        ttk.Label(frame, text="Input Folder:").grid(row=2, column=0, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=self.inputs).grid(row=2, column=1, sticky="ew")
        ttk.Button(frame, text="Browse", command=self.browse_inputs).grid(row=2, column=2, sticky="w", padx=5)

        ttk.Label(frame, text="Output Directory:").grid(row=3, column=0, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=self.out_dir).grid(row=3, column=1, sticky="ew")
        ttk.Button(frame, text="Browse", command=self.browse_out_dir).grid(row=3, column=2, sticky="w", padx=5)
        
        frame.grid_columnconfigure(1, weight=1)

    def create_calc_settings_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Calculation Settings", padding="10")
        frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        ttk.Label(frame, text="Functional:").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=self.functional).grid(row=0, column=1, sticky="ew")
        ttk.Label(frame, text="Basis:").grid(row=1, column=0, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=self.basis).grid(row=1, column=1, sticky="ew")

        ttk.Label(frame, text="Solvent Model:").grid(row=2, column=0, sticky="w", padx=5)
        ttk.Combobox(frame, textvariable=self.solvent_model, values=["none", "SMD", "PCM", "IEFPCM", "CPCM"]).grid(row=2, column=1, sticky="ew")
        ttk.Label(frame, text="Solvent Name:").grid(row=3, column=0, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=self.solvent_name).grid(row=3, column=1, sticky="ew")
        
        ttk.Label(frame, text="TD-DFT NStates:").grid(row=4, column=0, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=self.td_nstates).grid(row=4, column=1, sticky="ew")
        ttk.Label(frame, text="TD-DFT Root:").grid(row=5, column=0, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=self.td_root).grid(row=5, column=1, sticky="ew")
        
        ttk.Checkbutton(frame, text="Populations (full)", variable=self.pop_full).grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(frame, text="Dispersion (GD3BJ)", variable=self.dispersion).grid(row=7, column=0, columnspan=2, sticky="w", padx=5)

        frame.grid_columnconfigure(1, weight=1)
        
    def create_hpc_settings_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="HPC & Resource Settings", padding="10")
        frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        ttk.Label(frame, text="Nproc:").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=self.nproc).grid(row=0, column=1, sticky="ew")
        ttk.Label(frame, text="Mem:").grid(row=1, column=0, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=self.mem).grid(row=1, column=1, sticky="ew")

        ttk.Label(frame, text="Scheduler:").grid(row=2, column=0, sticky="w", padx=5)
        ttk.Combobox(frame, textvariable=self.scheduler, values=["pbs", "slurm", "local"]).grid(row=2, column=1, sticky="ew")
        ttk.Label(frame, text="Queue:").grid(row=3, column=0, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=self.queue).grid(row=3, column=1, sticky="ew")
        ttk.Label(frame, text="Walltime:").grid(row=4, column=0, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=self.walltime).grid(row=4, column=1, sticky="ew")
        ttk.Label(frame, text="Project:").grid(row=5, column=0, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=self.project).grid(row=5, column=1, sticky="ew")
        ttk.Label(frame, text="Account:").grid(row=6, column=0, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=self.account).grid(row=6, column=1, sticky="ew")
        
        frame.grid_columnconfigure(1, weight=1)

    def create_adv_settings_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Advanced Geometry Settings (Full Mode)", padding="10")
        frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)

        # Inline Step 4 with description
        ttk.Checkbutton(frame, text="Inline Coordinates for Step 4", variable=self.inline_step4).grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(frame, text="  (Forces Step 4 to use the input geometry instead of a checkpoint file.)", font=("Helvetica", 8, "italic")).grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=(0,5))
        
        # Inline Source with description
        ttk.Label(frame, text="Inline Source for Steps 5-7:").grid(row=2, column=0, sticky="w", padx=5)
        ttk.Radiobutton(frame, text="Step 1", variable=self.inline_source, value=1).grid(row=2, column=1, sticky="w")
        ttk.Radiobutton(frame, text="Step 4", variable=self.inline_source, value=4).grid(row=2, column=2, sticky="w")
        ttk.Label(frame, text="  (This is only used if you manually edit your script to inline Steps 5-7.)", font=("Helvetica", 8, "italic")).grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=(0,5))

        # Charge & Multiplicity Override
        ttk.Label(frame, text="Charge Override:").grid(row=4, column=0, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=self.charge).grid(row=4, column=1, sticky="ew")
        ttk.Label(frame, text="Multiplicity Override:").grid(row=5, column=0, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=self.multiplicity).grid(row=5, column=1, sticky="ew")
        ttk.Label(frame, text="  (Set both to override the values from your input file.)", font=("Helvetica", 8, "italic")).grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=(0,5))
        
        frame.grid_columnconfigure(1, weight=1)
    
    def create_action_frame(self, parent):
        frame = ttk.Frame(parent, padding="10")
        frame.grid(row=3, column=0, columnspan=2, sticky="ew")
        
        self.status_label = ttk.Label(frame, text="Ready.")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(frame, text="Generate Jobs", command=self.run_script).pack(side=tk.RIGHT, padx=5)

    # --- Helper methods ---
    def update_ui_state(self):
        if self.mode.get() == "single":
            self.step_entry.config(state='!disabled')
        else:
            self.step_entry.config(state='disabled')

    def browse_inputs(self):
        directory = filedialog.askdirectory()
        if directory:
            self.inputs.set(directory)

    def browse_out_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.out_dir.set(directory)

    def run_script(self):
        try:
            script.MODE = self.mode.get()
            script.STEP = self.step.get()
            script.INPUTS = self.inputs.get()
            script.OUT_DIR = self.out_dir.get()
            
            script.FUNCTIONAL = self.functional.get()
            script.BASIS = self.basis.get()
            
            script.SOLVENT_MODEL = self.solvent_model.get()
            script.SOLVENT_NAME = self.solvent_name.get()
            
            script.TD_NSTATES = self.td_nstates.get()
            script.TD_ROOT = self.td_root.get()
            script.POP_FULL = self.pop_full.get()
            script.DISPERSION = self.dispersion.get()
            
            script.NPROC = self.nproc.get()
            script.MEM = self.mem.get()
            script.SCHEDULER = self.scheduler.get()
            script.QUEUE = self.queue.get()
            script.WALLTIME = self.walltime.get()
            script.PROJECT = self.project.get()
            script.ACCOUNT = self.account.get()
            
            # Update INLINE_STEPS based on the new Checkbutton
            script.INLINE_STEPS = [4] if self.inline_step4.get() else []
            script.INLINE_SOURCE_5TO7 = self.inline_source.get()
            
            script.CHARGE = int(self.charge.get()) if self.charge.get() else None
            script.MULT = int(self.multiplicity.get()) if self.multiplicity.get() else None

            script.main()
            self.status_label.config(text="Jobs generated successfully! 🎉")
            
        except Exception as e:
            self.status_label.config(text=f"Error: {e}", foreground="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = GaussianGUI(root)
    app.update_ui_state()
    root.mainloop()
