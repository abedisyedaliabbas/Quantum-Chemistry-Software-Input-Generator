# Setting Up Ollama (Free, Local AI Assistant)

Ollama is a free, local AI solution that doesn't require API keys or have usage limits. It's the recommended option for unlimited AI assistance.

## Installation

### Windows

1. **Download Ollama:**
   - Go to https://ollama.com/download
   - Download the Windows installer
   - Run the installer and follow the setup wizard

2. **Verify Installation:**
   - Open a new Command Prompt or PowerShell window
   - Type: `ollama --version`
   - You should see the version number

3. **Download a Model (Required):**
   - In Command Prompt/PowerShell, run:
     ```
     ollama pull llama3.1:8b
     ```
   - This downloads the Llama 3.1 8B model (recommended, ~4.7GB)
   - Wait for the download to complete

4. **Start Ollama Server:**
   - On Windows, Ollama usually runs as a background service automatically
   - To check if it's running:
     ```
     ollama list
     ```
   - If it works, the server is running!

### Mac/Linux

1. **Install Ollama:**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Download a Model:**
   ```bash
   ollama pull llama3.1:8b
   ```

3. **Start Server (if needed):**
   - On Mac, Ollama usually starts automatically
   - To start manually:
     ```bash
     ollama serve
     ```

## Using Ollama in the Application

Once Ollama is installed and a model is downloaded:

1. **Restart the Quantum Steps GUI application**
2. The AI Assistant tab should show "✓ Available" next to Ollama
3. When you use the AI Assistant, it will automatically use Ollama instead of Gemini
4. No API keys needed - completely free and local!

## Benefits of Ollama

- ✅ **Free** - No API costs
- ✅ **Unlimited** - No quota or rate limits
- ✅ **Local** - Your data stays on your computer
- ✅ **Private** - No data sent to external servers
- ✅ **Fast** - Runs on your local machine

## Troubleshooting

**"Ollama server is not running":**
- Windows: Restart the Ollama service or restart your computer
- Mac/Linux: Run `ollama serve` in a terminal

**"Model not found":**
- Run `ollama pull llama3.1:8b` to download the model

**Connection errors:**
- Make sure Ollama is running
- Try restarting the application
- Check firewall settings if issues persist


