# 🚀 GitHub Upload - Detailed Instructions

## Step 1: Create Repository on GitHub

1. **Go to GitHub**: https://github.com
2. **Click "New"** or "+" in the top right corner
3. **Fill in details**:
   - **Repository name**: `speckle-cnn-training`
   - **Description**: `GPU-optimized CNN training for speckle pattern recognition with RTX 3090 Ti support`
   - **Visibility**: Choose Public or Private
   - **❌ Don't check** "Add a README file" (we already have one)
   - **❌ Don't check** "Add .gitignore" (we already have one)
   - **❌ Don't check** "Choose a license" (optional)
4. **Click "Create repository"**

## Step 2: Automatic Upload

### Option A: Windows (Double-click)
```bash
upload_to_github.bat
```

### Option B: Python (All Platforms)
```bash
python upload_to_github.py
```

### Option C: Manual
```bash
# 1. Set the URL (replace YOUR_USERNAME and REPO_NAME)
git remote set-url origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 2. Upload to GitHub
git push -u origin main
```

## Step 3: Authentication

If you encounter authentication errors, you have several options:

### Option 1: GitHub CLI (Recommended)
```bash
# Install GitHub CLI
# Windows: winget install GitHub.cli
# Or download from: https://cli.github.com/

# Login
gh auth login

# Upload
git push -u origin main
```

### Option 2: Personal Access Token
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Select scopes: `repo`, `workflow`
4. Copy the token
5. Use the token as password when uploading

### Option 3: SSH Keys
```bash
# Create SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to GitHub
# Copy the content of ~/.ssh/id_ed25519.pub
# Add at: https://github.com/settings/keys

# Change to SSH URL
git remote set-url origin git@github.com:USERNAME/REPO.git
```

## Step 4: Verification

After uploading, check:
1. **Repository URL**: https://github.com/YOUR_USERNAME/speckle-cnn-training
2. **Files**: All our files should be there
3. **README**: Should display our documentation

## Files to be Uploaded

- ✅ `train_speckles_cnn.py` - Main code
- ✅ `config.json` - Basic configuration
- ✅ `config_high_performance.json` - For powerful computers
- ✅ `config_cpu.json` - For CPU-only training
- ✅ `check_gpu.py` - GPU check
- ✅ `run_training.py` - Run with options
- ✅ `setup.py` - Automatic installation
- ✅ `examples.py` - Usage examples
- ✅ `run_gpu.bat` / `run_cpu.bat` - Run scripts
- ✅ `README.md` - Detailed documentation
- ✅ `PERFORMANCE_TIPS.md` - Performance tips
- ✅ `requirements.txt` - Required packages
- ✅ `.gitignore` - Files to ignore

## Common Issues

### "Repository not found"
- Check that the URL is correct
- Ensure you have access to the repository

### "Authentication failed"
- Use GitHub CLI or Personal Access Token
- Check that SSH key is configured correctly

### "Permission denied"
- Ensure you have write access to the repository
- Check that the repository is not private if you're not logged in

### "Remote origin already exists"
```bash
git remote remove origin
git remote add origin YOUR_NEW_URL
```

## Continued Development

After the first upload, you can:

```bash
# Add changes
git add .
git commit -m "Your commit message"
git push

# Clone repository elsewhere
git clone https://github.com/YOUR_USERNAME/speckle-cnn-training.git

# Update existing repository
git pull origin main
```

## Support

If you encounter issues:
1. Check error messages
2. Use `upload_to_github.py` for automatic checking
3. Check GitHub documentation
