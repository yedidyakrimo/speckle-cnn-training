@echo off
echo ========================================
echo Speckle CNN Training - GitHub Upload
echo ========================================
echo.

REM Check if git is initialized
if not exist ".git" (
    echo Error: Git repository not initialized
    echo Please run: git init
    pause
    exit /b 1
)

REM Check if there are commits
git log --oneline -1 >nul 2>&1
if errorlevel 1 (
    echo Error: No commits found
    echo Please run: git add . && git commit -m "Initial commit"
    pause
    exit /b 1
)

echo Current git status:
git status --short
echo.

echo ========================================
echo Step 1: Create GitHub Repository
echo ========================================
echo.
echo 1. Go to https://github.com
echo 2. Click "New" or "+" button
echo 3. Fill in repository details:
echo    - Name: speckle-cnn-training
echo    - Description: GPU-optimized CNN training for speckle pattern recognition
echo    - Choose Public or Private
echo    - DO NOT check "Add a README file"
echo    - DO NOT check "Add .gitignore"
echo 4. Click "Create repository"
echo.

set /p GITHUB_URL="Enter your GitHub repository URL (e.g., https://github.com/username/repo.git): "

if "%GITHUB_URL%"=="" (
    echo Error: No URL provided
    pause
    exit /b 1
)

echo.
echo Setting remote URL to: %GITHUB_URL%
git remote set-url origin %GITHUB_URL%

echo.
echo ========================================
echo Step 2: Upload to GitHub
echo ========================================
echo.

echo Pushing to GitHub...
git push -u origin main

if errorlevel 1 (
    echo.
    echo Error: Failed to push to GitHub
    echo Please check:
    echo 1. Repository URL is correct
    echo 2. You have access to the repository
    echo 3. You are authenticated with GitHub
    echo.
    echo For authentication, you may need to:
    echo 1. Use GitHub CLI: gh auth login
    echo 2. Or use personal access token
    echo 3. Or use SSH keys
    pause
    exit /b 1
)

echo.
echo ========================================
echo Success!
echo ========================================
echo.
echo Your project has been uploaded to GitHub!
echo Repository URL: %GITHUB_URL%
echo.
echo You can now:
echo 1. Share the repository with others
echo 2. Clone it on other machines
echo 3. Continue developing and pushing changes
echo.

pause
