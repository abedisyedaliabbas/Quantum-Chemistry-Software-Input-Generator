# Deployment Guide - Sharing Your Web App

## Option 1: Local Network (Same WiFi) - EASIEST ✅

**Best for:** Testing with friends on the same network

### Steps:

1. **Start the server:**
   ```bash
   python quantum_steps_web.py
   ```
   Or double-click `start_web_server.bat` (Windows)

2. **Get your IP address:**
   ```bash
   python get_ip.py
   ```
   This will show your IP (e.g., `192.168.1.100`)

3. **Share with your friend:**
   - Give them the URL: `http://YOUR_IP:5000`
   - Example: `http://192.168.1.100:5000`
   - They open it in their browser (iOS Safari, Chrome, etc.)

4. **Firewall settings:**
   - Windows: Allow Python through firewall when prompted
   - Or manually allow port 5000 in Windows Firewall

**Pros:**
- ✅ No code sharing needed
- ✅ No internet required (works offline)
- ✅ Free
- ✅ Easy setup

**Cons:**
- ❌ Friend must be on same WiFi
- ❌ Your computer must be running
- ❌ Only works while server is active

---

## Option 2: Deploy to Cloud (Public URL) - BEST FOR SHARING 🌐

**Best for:** Sharing with anyone, anywhere

### Free Options:

#### A. **PythonAnywhere** (Free tier available)
1. Sign up at https://www.pythonanywhere.com
2. Upload your files
3. Get a public URL like: `yourname.pythonanywhere.com`
4. Friend can access from anywhere!

#### B. **Heroku** (Free tier discontinued, but paid is cheap)
1. Install Heroku CLI
2. Create `Procfile`:
   ```
   web: python quantum_steps_web.py
   ```
3. Deploy: `heroku create` then `git push heroku main`
4. Get public URL

#### C. **Railway** (Free tier available)
1. Sign up at https://railway.app
2. Connect GitHub repo or upload files
3. Deploy automatically
4. Get public URL

#### D. **Render** (Free tier available)
1. Sign up at https://render.com
2. Create new Web Service
3. Connect repo or upload files
4. Get public URL

**Pros:**
- ✅ Works from anywhere (not just same WiFi)
- ✅ No code sharing needed
- ✅ Always available (if you keep it running)
- ✅ Professional URL

**Cons:**
- ❌ Requires internet
- ❌ Some setup required
- ❌ Free tiers have limitations

---

## Option 3: Share the Code (Self-Hosted)

**Best for:** Friends who want to run it themselves

1. Share the entire project folder
2. Friend installs dependencies: `pip install -r requirements_web.txt`
3. Friend runs: `python quantum_steps_web.py`
4. Friend accesses: `http://localhost:5000`

**Pros:**
- ✅ Friend has full control
- ✅ Works offline

**Cons:**
- ❌ Friend needs Python installed
- ❌ Friend needs to install dependencies
- ❌ More setup for friend

---

## Quick Start (Local Network)

**Windows:**
```bash
# Double-click this file:
start_web_server.bat
```

**Mac/Linux:**
```bash
python quantum_steps_web.py
```

Then share the URL shown (e.g., `http://192.168.1.100:5000`)

---

## Troubleshooting

**Friend can't connect:**
- ✅ Check both are on same WiFi
- ✅ Check firewall allows port 5000
- ✅ Check server is running
- ✅ Try `http://0.0.0.0:5000` instead of localhost

**Firewall issues:**
- Windows: Search "Windows Defender Firewall" → Allow an app → Python
- Or temporarily disable firewall for testing

**Port already in use:**
- Change port in `quantum_steps_web.py`:
  ```python
  app.run(debug=True, host='0.0.0.0', port=5001)  # Use 5001 instead
  ```

---

## Recommended: Start with Option 1 (Local Network)

It's the easiest and requires no code sharing. Just share your IP address!


