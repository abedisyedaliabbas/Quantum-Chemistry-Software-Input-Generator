@echo off
echo ============================================================
echo Quantum Chemical Calculations Step Maker - Git Setup
echo ============================================================
echo.

echo Step 1: Initializing Git repository...
git init
if %errorlevel% neq 0 (
    echo ERROR: Git is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Step 2: Adding all files...
git add .

echo.
echo Step 3: Making initial commit...
git commit -m "Initial commit: Quantum Chemical Calculations Step Maker v2.0 - Unified GUI with Web Support"

echo.
echo Step 4: Adding GitHub remote...
git remote add origin https://github.com/abedisyedaliabbas/Quantum-Chemistry-Software-Input-Generator.git
if %errorlevel% neq 0 (
    echo WARNING: Remote might already exist. Continuing...
    git remote set-url origin https://github.com/abedisyedaliabbas/Quantum-Chemistry-Software-Input-Generator.git
)

echo.
echo Step 5: Setting main branch...
git branch -M main

echo.
echo ============================================================
echo Setup complete!
echo ============================================================
echo.
echo Next steps:
echo 1. Review the changes: git status
echo 2. Push to GitHub: git push -u origin main
echo.
echo Note: You may need to authenticate with GitHub
echo       Use a Personal Access Token if prompted for password
echo.
pause

