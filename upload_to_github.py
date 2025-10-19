#!/usr/bin/env python3
"""
Script to help upload the project to GitHub
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and return success status"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False

def check_git_status():
    """Check if git is properly initialized"""
    print("=== Checking Git Status ===")
    
    # Check if git is initialized
    if not os.path.exists('.git'):
        print("✗ Git repository not initialized")
        print("Please run: git init")
        return False
    
    # Check if there are commits
    result = subprocess.run('git log --oneline -1', shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("✗ No commits found")
        print("Please run: git add . && git commit -m 'Initial commit'")
        return False
    
    print("✓ Git repository is ready")
    return True

def show_github_instructions():
    """Show instructions for creating GitHub repository"""
    print("\n=== GitHub Repository Setup ===")
    print("1. Go to https://github.com")
    print("2. Click 'New' or '+' button")
    print("3. Fill in repository details:")
    print("   - Name: speckle-cnn-training")
    print("   - Description: GPU-optimized CNN training for speckle pattern recognition")
    print("   - Choose Public or Private")
    print("   - DO NOT check 'Add a README file'")
    print("   - DO NOT check 'Add .gitignore'")
    print("4. Click 'Create repository'")
    print("5. Copy the repository URL")

def get_repository_url():
    """Get repository URL from user"""
    print("\n=== Repository URL ===")
    url = input("Enter your GitHub repository URL (e.g., https://github.com/username/repo.git): ").strip()
    
    if not url:
        print("Error: No URL provided")
        return None
    
    if not url.startswith('https://github.com/') and not url.startswith('git@github.com:'):
        print("Error: Invalid GitHub URL format")
        return None
    
    return url

def upload_to_github(url):
    """Upload the project to GitHub"""
    print(f"\n=== Uploading to GitHub ===")
    print(f"Repository URL: {url}")
    
    # Set remote URL
    if not run_command(f'git remote set-url origin {url}', 'Setting remote URL'):
        return False
    
    # Push to GitHub
    if not run_command('git push -u origin main', 'Pushing to GitHub'):
        print("\nTroubleshooting tips:")
        print("1. Check if repository URL is correct")
        print("2. Ensure you have access to the repository")
        print("3. Authenticate with GitHub:")
        print("   - Use GitHub CLI: gh auth login")
        print("   - Or use personal access token")
        print("   - Or use SSH keys")
        return False
    
    print("\n✓ Successfully uploaded to GitHub!")
    print(f"Repository URL: {url}")
    return True

def main():
    """Main function"""
    print("Speckle CNN Training - GitHub Upload Helper")
    print("=" * 50)
    
    # Check git status
    if not check_git_status():
        sys.exit(1)
    
    # Show GitHub instructions
    show_github_instructions()
    
    # Get repository URL
    url = get_repository_url()
    if not url:
        sys.exit(1)
    
    # Upload to GitHub
    if upload_to_github(url):
        print("\n=== Upload Complete ===")
        print("Your project is now on GitHub!")
        print("You can now:")
        print("1. Share the repository with others")
        print("2. Clone it on other machines")
        print("3. Continue developing and pushing changes")
    else:
        print("\n=== Upload Failed ===")
        print("Please check the error messages above and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()
