"""
AI Assistant Module for Quantum Chemistry Calculations
Supports Ollama (local, free, no API key) and Google Gemini (requires API key)
"""
import json
import re
import subprocess
from typing import Dict, Optional, Tuple, List
from pathlib import Path

# Check for Ollama (local, free, no API key needed)
OLLAMA_AVAILABLE = False
OLLAMA_SERVER_RUNNING = False

def check_ollama_installed() -> bool:
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True, timeout=2)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False

def check_ollama_server_running() -> bool:
    """Check if Ollama server is actually running on port 11434"""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', 11434))
        sock.close()
        return result == 0
    except Exception:
        return False

# Check if Ollama is installed
OLLAMA_AVAILABLE = check_ollama_installed()
# Check if Ollama server is running (will be checked dynamically)
OLLAMA_SERVER_RUNNING = check_ollama_server_running()

# Check for Gemini (requires API key)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def load_gemini_api_key(key_file: Path = None) -> Optional[str]:
    """Load Gemini API key from file or environment"""
    if key_file is None:
        key_file = Path.home() / ".gemini_api_key.txt"
    
    # Try file first
    if key_file.exists():
        try:
            with open(key_file, 'r') as f:
                key = f.read().strip()
                if key:
                    return key
        except Exception:
            pass
    
    # Try environment variable
    import os
    key = os.getenv('GEMINI_API_KEY')
    if key:
        return key
    
    return None


def save_gemini_api_key(key: str, key_file: Path = None):
    """Save Gemini API key to file"""
    if key_file is None:
        key_file = Path.home() / ".gemini_api_key.txt"
    
    try:
        # Ensure the key is not empty
        key = key.strip()
        if not key:
            return False, "API key cannot be empty"
        
        # Ensure parent directory exists (important for Windows)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the key to file
        with open(key_file, 'w') as f:
            f.write(key)
        
        # Set file permissions to be readable only by the user (Unix/Linux/Mac)
        import os
        import stat
        try:
            os.chmod(key_file, stat.S_IRUSR | stat.S_IWUSR)
        except (AttributeError, OSError):
            # Windows doesn't support chmod the same way, ignore
            pass
        
        return True, None
    except PermissionError as e:
        return False, f"Permission denied: {str(e)}"
    except OSError as e:
        return False, f"Failed to save API key: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def get_quantum_chemistry_system_prompt(software: str = "gaussian") -> str:
    """Get system prompt for quantum chemistry assistant"""
    
    if software.lower() == "orca":
        software_info = """
ORCA Workflow Steps:
- Step 1: Ground state optimization
- Step 2: Vertical excitation (Franck-Condon state)
- Step 4: Excited state optimization
- Step 7: Ground state at excited state geometry
- Step 9: Custom step (user-defined)

Common ORCA Methods: m06-2x, b3lyp, pbe0, wb97x-d3
Common ORCA Basis Sets: def2-SVP, def2-TZVP, def2-QZVP, cc-pVDZ, cc-pVTZ
Solvent Models: SMD, CPCM, PCM
"""
    else:  # gaussian
        software_info = """
Gaussian Workflow Steps:
- Step 1: Ground state geometry optimization + frequency
- Step 2: Vertical excitation (Franck-Condon state)
- Step 3: cLR correction of vertical excitation energy
- Step 4: Excited state geometry optimization
- Step 5: Density calculation at optimized excited state geometry
- Step 6: cLR correction of excited state energy
- Step 7: Ground state energy at excited state geometry

Common Gaussian Methods: m062x, b3lyp, pbe0, wb97xd, cam-b3lyp
Common Gaussian Basis Sets: def2SVP, def2TZVP, def2QZVP, 6-31G(d), 6-311+G(d,p), cc-pVDZ, cc-pVTZ
Solvent Models: SMD, PCM, IEFPCM, CPCM
"""
    
    return f"""You are an expert AI assistant for quantum chemistry calculations using {software.upper()}.

Your role is to help users set up quantum chemistry calculations by conducting a structured conversation to gather all necessary information, then automatically generating the input files.

CONVERSATION FLOW:
1. START by asking: "What molecule/system do you want to calculate? Please provide:
   - Input file path (.com/.xyz/.log), folder path, glob pattern, OR SMILES string
     (You can specify a single file, a folder containing multiple files, or a glob pattern like '*.com')
     NOTE: The system can read these files automatically to extract geometry, charge, multiplicity, and calculate dihedral angles
   - Output directory where files should be created
   - What type of calculation you want (e.g., 'full workflow', 'optimization only', 'excitation energies', 'torsional scan')"

2. Then ask systematically for:
   - Charge and spin state (multiplicity) - if not in input file
   - Method and basis set (provide recommendations if unsure)
   - Solvent (if any)
   - Which calculation steps to run (single step, multiple steps, or full workflow)
   - Computational resources (processors, memory)
   - Scheduler settings (PBS/SLURM/Local)

3. After gathering all information, summarize the configuration and say:
   "I have all the information needed. I'll now generate the input files for you. Please use the 'Generate Files' button or say 'generate files' to proceed."

4. When the user confirms or says "generate files", respond with a special marker:
   "GENERATE_FILES_START
   [JSON configuration here]
   GENERATE_FILES_END"

{software_info}

IMPORTANT GUIDELINES:
- Be conversational and friendly
- Ask one question at a time to avoid overwhelming the user
- Provide recommendations when users are uncertain
- Always confirm file paths and output directories before generating
- If information is missing, ask for it clearly
- Explain your recommendations briefly (e.g., "I recommend M06-2X/def2-TZVP for good accuracy-speed balance")

SPECIAL CALCULATION TYPES:

For TICT/torsional/dihedral scans:
- If the user requests a torsional scan, dihedral scan, or TICT scan, include these REQUIRED fields in the JSON:
  - "CALCULATION_TYPE": "torsional_scan" or "dihedral_scan" or "tict_scan" (REQUIRED)
  - "INPUT_FILE": "full path to input file" (ABSOLUTELY REQUIRED - extract from conversation, e.g., "C:\\Users\\Admin\\Desktop\\SMILES\\01CH3_WB97XD_water.com")
  - "OUTPUT_DIR": "full path to output directory" (ABSOLUTELY REQUIRED - if user says "same folder" or "same directory", use the input file's directory)
  - "DIHEDRAL_ATOMS": "atom1 atom2 atom3 atom4" (REQUIRED, e.g., "12 11 10 3") - the four atoms defining the dihedral
  - "SCAN_RANGE": "start end" or "current end" (e.g., "0 90" or "current 90") - REQUIRED
    - CRITICAL: When the user asks "what is the current angle?" or "from current angle", respond with:
      "The system will automatically calculate the current dihedral angle from your input file when generating the scan files. You don't need to provide it manually. Just tell me what angle you want to scan TO (e.g., 'to 90 degrees'), and I'll set it up to scan from the current angle in your file to that target angle."
    - If "current" is specified or range is empty, the backend system will automatically read the input file and calculate the starting dihedral angle from the coordinates - YOU don't need to calculate it
    - The user can say "from current angle to 90" or "rotate from current to 90" and you should use "current 90" in SCAN_RANGE
    - Always use "current [end_angle]" format when user wants to scan from current angle - the system handles the calculation automatically
  - "NUM_STEPS": number (e.g., 10) - REQUIRED, number of steps in the scan

EXAMPLE JSON for TICT scan (use this format):
{{"CALCULATION_TYPE": "tict_scan", "INPUT_FILE": "C:\\\\Users\\\\Admin\\\\Desktop\\\\SMILES\\\\01CH3_WB97XD_water.com", "OUTPUT_DIR": "C:\\\\Users\\\\Admin\\\\Desktop\\\\SMILES", "DIHEDRAL_ATOMS": "12 11 10 3", "SCAN_RANGE": "current 90", "NUM_STEPS": 10}}

- These scans use the TICT rotation module and don't require the regular Gaussian workflow steps
- The system supports bulk processing: you can provide a folder path or glob pattern, and it will process all matching files
- The system can automatically read and parse .com, .xyz, and .inp files to extract geometry and calculate dihedral angles - you don't need to do this manually
- CRITICAL: You MUST extract the INPUT_FILE path from the conversation - look for file paths like "C:\\...", quotes around paths, or .com/.xyz file extensions. If the user says "same folder" or "same directory" for output, use the input file's directory as OUTPUT_DIR

PARAMETER EXTRACTION:
Extract and use these parameters (provide defaults if not specified):
- Input file path or SMILES string (REQUIRED)
- Output directory (REQUIRED)
- Charge (default: 0)
- Multiplicity (default: 1)
- Method (default: m062x for Gaussian, m06-2x for ORCA)
- Basis set (default: def2TZVP for Gaussian, def2-TZVP for ORCA)
- Solvent model and name (default: none)
- Calculation mode: 'single', 'multiple', or 'full'
- Step number(s) if mode is 'single' or 'multiple'
- Number of processors (default: 8)
- Memory (default: reasonable based on system size)
- Scheduler: 'pbs', 'slurm', or 'local' (default: 'local')

Always be helpful, accurate, and explain your recommendations. If unsure, ask for clarification."""


class QuantumChemistryAssistant:
    """AI Assistant for quantum chemistry calculations
    Prefers Ollama (free, local, no API key) over Gemini (requires API key)
    """
    
    def __init__(self, api_key: Optional[str] = None, software: str = "gaussian", use_ollama: bool = True):
        self.software = software.lower()
        self.conversation_history = []
        self.use_ollama = use_ollama
        self.ollama_model = "llama3.1:8b"  # Default Ollama model
        
        # Prefer Ollama if available (free, no API key needed)
        if use_ollama and OLLAMA_AVAILABLE:
            self.backend = "ollama"
            self._check_ollama_model()
        elif GEMINI_AVAILABLE:
            self.backend = "gemini"
            # Load API key
            if api_key is None:
                api_key = load_gemini_api_key()
            
            if not api_key:
                if OLLAMA_AVAILABLE:
                    # Fallback to Ollama if Gemini key not available
                    self.backend = "ollama"
                    self.use_ollama = True
                    self._check_ollama_model()
                else:
                    raise ValueError("No API key found and Ollama not available. Please install Ollama (recommended, free) or set Gemini API key.")
            else:
                # Configure Gemini
                genai.configure(api_key=api_key)
                # Try different model names in order of preference
                # Updated to use correct model names that actually exist
                model_names = [
                    'gemini-2.5-flash',      # Latest and fastest
                    'gemini-2.0-flash',      # Newer version
                    'gemini-1.5-flash',      # Stable version
                    'gemini-2.5-pro',        # Latest pro version
                    'gemini-1.5-pro',        # Stable pro version
                    'gemini-pro',            # Older but reliable
                    'gemini-flash-latest',  # Latest flash (generic)
                    'gemini-pro-latest'     # Latest pro (generic)
                ]
                model_configured = False
                last_error = None
                
                for model_name in model_names:
                    try:
                        self.model = genai.GenerativeModel(
                            model_name=model_name,
                            system_instruction=get_quantum_chemistry_system_prompt(self.software)
                        )
                        self.model_name = model_name
                        model_configured = True
                        break
                    except Exception as e:
                        last_error = str(e)
                        continue
                
                if not model_configured:
                    # Try to list available models for better error message
                    available_models = []
                    try:
                        for model in genai.list_models():
                            if 'generateContent' in model.supported_generation_methods:
                                # Remove 'models/' prefix if present
                                model_name = model.name.replace('models/', '')
                                available_models.append(model_name)
                    except Exception:
                        pass
                    
                    error_msg = f"Gemini API error: Could not initialize any model.\n"
                    if last_error:
                        error_msg += f"Last error: {last_error}\n"
                    if available_models:
                        error_msg += f"Available models: {', '.join(available_models[:10])}\n"
                    error_msg += "Please check your API key or use Ollama instead."
                    raise ValueError(error_msg)
                
                self.chat = self.model.start_chat(history=[])
        else:
            raise ImportError("Neither Ollama nor Gemini available. Install Ollama (recommended, free) or google-generativeai")
    
    def _check_ollama_model(self):
        """Check if Ollama model is available, pull if needed"""
        try:
            # Check if model exists
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
            if self.ollama_model not in result.stdout:
                # Model not found, but don't auto-pull (let user do it manually)
                pass
        except Exception:
            pass
    
    def send_message(self, user_message: str) -> Tuple[str, bool]:
        """
        Send a message to the AI assistant
        
        Returns:
            Tuple of (response_text, success)
        """
        try:
            if self.backend == "ollama":
                return self._send_message_ollama(user_message)
            else:
                try:
                    response = self.chat.send_message(user_message)
                    return response.text, True
                except Exception as e:
                    error_str = str(e)
                    # Check for quota/rate limit errors
                    if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower() or "exceeded" in error_str.lower():
                        # Try to fallback to Ollama if available
                        if OLLAMA_AVAILABLE and self.use_ollama:
                            # Switch to Ollama backend
                            try:
                                self.backend = "ollama"
                                self._check_ollama_model()
                                return self._send_message_ollama(user_message)
                            except Exception:
                                return f"⚠️ Gemini quota exceeded. Please install Ollama (free) for unlimited local use:\n\nWindows: Download from https://ollama.com/download\nMac/Linux: curl -fsSL https://ollama.com/install.sh | sh\n\nOr wait and try again later.\n\nOriginal error: {error_str}", False
                        else:
                            return f"⚠️ Gemini quota exceeded. Please install Ollama (free) for unlimited local use:\n\nWindows: Download from https://ollama.com/download\nMac/Linux: curl -fsSL https://ollama.com/install.sh | sh\n\nOr wait and try again later.\n\nOriginal error: {error_str}", False
                    return f"Error: {error_str}", False
        except Exception as e:
            return f"Error: {str(e)}", False
    
    def _send_message_ollama(self, user_message: str) -> Tuple[str, bool]:
        """Send message using Ollama (local, free)"""
        # First check if server is running
        if not check_ollama_server_running():
            # Try to start Ollama server in background (macOS/Linux)
            # Note: On macOS, Ollama usually runs as a service, but we try anyway
            try:
                import platform
                import os
                if platform.system() != 'Windows':
                    # Try to start Ollama in background (this may not work if it's already managed by systemd/launchd)
                    try:
                        subprocess.Popen(['ollama', 'serve'], 
                                       stdout=subprocess.DEVNULL, 
                                       stderr=subprocess.DEVNULL,
                                       preexec_fn=os.setsid if hasattr(os, 'setsid') else None)
                        # Wait a bit for server to start
                        import time
                        for _ in range(5):  # Try for up to 5 seconds
                            time.sleep(1)
                            if check_ollama_server_running():
                                break
                    except Exception:
                        pass
            except Exception:
                pass
            
            # Check again
            if not check_ollama_server_running():
                return (
                    "⚠️ Ollama server is not running.\n\n"
                    "To start the Ollama server, open Terminal and run ONE of these commands:\n\n"
                    "Option 1 (Recommended - will start server and test it):\n"
                    f"  ollama run {self.ollama_model}\n\n"
                    "Option 2 (Start server in background):\n"
                    "  ollama serve\n\n"
                    "After running one of these commands, wait a few seconds, then try again.\n\n"
                    "Note: On macOS, Ollama may already be installed but the server needs to be started manually.\n\n"
                    "Alternatively, you can use Gemini API (requires API key).",
                    False
                )
        
        try:
            import requests
            import json as json_lib
            
            # Build prompt with system instruction
            system_prompt = get_quantum_chemistry_system_prompt(self.software)
            
            # Add conversation history
            conversation_text = ""
            for i, msg in enumerate(self.conversation_history):
                role = "User" if i % 2 == 0 else "Assistant"
                conversation_text += f"{role}: {msg}\n\n"
            
            # Combine into full prompt
            full_prompt = f"{system_prompt}\n\n{conversation_text}User: {user_message}\n\nAssistant:"
            
            # Call Ollama API
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.ollama_model,
                    'prompt': full_prompt,
                    'stream': False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                # Add to conversation history
                self.conversation_history.append(user_message)
                self.conversation_history.append(response_text)
                return response_text, True
            else:
                error_msg = response.text
                if "model" in error_msg.lower() and "not found" in error_msg.lower():
                    return f"Model '{self.ollama_model}' not found. Please run: ollama pull {self.ollama_model}", False
                return f"Ollama API error: {response.status_code} - {error_msg}", False
                
        except ImportError:
            # Fallback to subprocess if requests not available
            # First check if server is running
            if not check_ollama_server_running():
                return (
                    "⚠️ Ollama server is not running.\n\n"
                    "To start the Ollama server, open Terminal and run:\n"
                    f"  ollama run {self.ollama_model}\n\n"
                    "This will start the server and test it. Wait a few seconds, then try again.",
                    False
                )
            
            try:
                system_prompt = get_quantum_chemistry_system_prompt(self.software)
                conversation_text = "\n".join([f"{'User' if i % 2 == 0 else 'Assistant'}: {msg}" 
                                              for i, msg in enumerate(self.conversation_history)])
                full_prompt = f"{system_prompt}\n\n{conversation_text}\n\nUser: {user_message}\n\nAssistant:"
                
                process = subprocess.run(
                    ['ollama', 'run', self.ollama_model, full_prompt],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if process.returncode == 0:
                    response_text = process.stdout.strip()
                    self.conversation_history.append(user_message)
                    self.conversation_history.append(response_text)
                    return response_text, True
                else:
                    error_msg = process.stderr
                    if "model" in error_msg.lower() and "not found" in error_msg.lower():
                        return f"Model '{self.ollama_model}' not found. Please run: ollama pull {self.ollama_model}", False
                    return f"Ollama error: {error_msg}", False
            except Exception as e:
                error_str = str(e)
                if "model" in error_str.lower() and "not found" in error_str.lower():
                    return f"Model '{self.ollama_model}' not found. Please run: ollama pull {self.ollama_model}", False
                return f"Error: {error_str}. Make sure Ollama server is running and model {self.ollama_model} is installed.", False
        except Exception as e:
            error_str = str(e)
            if "Connection" in error_str or "refused" in error_str.lower() or "connect" in error_str.lower():
                return (
                    "⚠️ Cannot connect to Ollama server.\n\n"
                    "The Ollama server is not running. To start it, open Terminal and run:\n\n"
                    f"  ollama run {self.ollama_model}\n\n"
                    "This will start the server. Wait a few seconds, then try again.",
                    False
                )
            return f"Error: {error_str}", False
    
    def reset_conversation(self):
        """Reset the conversation history"""
        if self.backend == "gemini":
            self.chat = self.model.start_chat(history=[])
        self.conversation_history = []
    
    def extract_parameters(self, conversation_text: str) -> Dict:
        """
        Extract calculation parameters from conversation
        
        This is a simple parser - in production, you might want to use
        more sophisticated NLP or ask the AI to output structured JSON
        """
        params = {
            'method': None,
            'basis': None,
            'solvent_model': None,
            'solvent_name': None,
            'charge': None,
            'mult': None,
            'steps': [],
            'td_nstates': 3,
            'td_root': 1,
        }
        
        # Simple keyword extraction (can be enhanced)
        text_lower = conversation_text.lower()
        
        # Method detection
        methods = ['m062x', 'm06-2x', 'b3lyp', 'pbe0', 'wb97xd', 'cam-b3lyp']
        for method in methods:
            if method in text_lower:
                params['method'] = method.replace('-', '')
                break
        
        # Basis set detection
        basis_sets = ['def2svp', 'def2tzvp', 'def2qzvp', '6-31g', '6-311+g', 'cc-pvdz', 'cc-pvtz']
        for basis in basis_sets:
            if basis in text_lower:
                params['basis'] = basis
                break
        
        # Solvent detection
        if 'smd' in text_lower:
            params['solvent_model'] = 'SMD'
        elif 'pcm' in text_lower:
            params['solvent_model'] = 'PCM'
        
        solvents = ['dmso', 'water', 'acetonitrile', 'methanol', 'ethanol', 'toluene', 'chloroform']
        for solvent in solvents:
            if solvent in text_lower:
                params['solvent_name'] = solvent.upper() if solvent == 'dmso' else solvent.capitalize()
                break
        
        # Charge and multiplicity
        charge_match = re.search(r'charge[:\s]+(-?\d+)', text_lower)
        if charge_match:
            params['charge'] = int(charge_match.group(1))
        
        mult_match = re.search(r'(?:multiplicity|mult|spin)[:\s]+(\d+)', text_lower)
        if mult_match:
            params['mult'] = int(mult_match.group(1))
        
        return params
    
    def extract_generation_config(self, response_text: str) -> Optional[Dict]:
        """
        Extract JSON configuration from AI response that contains GENERATE_FILES_START/END markers
        Returns None if markers not found, or the parsed config dict
        """
        try:
            start_marker = "GENERATE_FILES_START"
            end_marker = "GENERATE_FILES_END"
            
            start_idx = response_text.find(start_marker)
            end_idx = response_text.find(end_marker)
            
            if start_idx == -1 or end_idx == -1:
                return None
            
            json_str = response_text[start_idx + len(start_marker):end_idx].strip()
            config = json.loads(json_str)
            return config
        except Exception:
            return None
