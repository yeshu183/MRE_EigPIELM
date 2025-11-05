# PowerShell script to push MRE-PINN to GitHub for Lightning AI
# This is the easiest way to get your code into Lightning AI Studios

Write-Host "=== MRE-PINN GitHub Setup ===" -ForegroundColor Cyan
Write-Host "This will help you push your code to GitHub" -ForegroundColor Yellow
Write-Host "Then you can easily clone it in Lightning AI!" -ForegroundColor Yellow
Write-Host ""

# Navigate to repo root
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "Repository location: $repoRoot" -ForegroundColor Green
Write-Host ""

# Check if git is installed
try {
    git --version | Out-Null
} catch {
    Write-Host "Error: Git is not installed!" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Check if already a git repo
if (-not (Test-Path ".git")) {
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "Git repository initialized!" -ForegroundColor Green
    Write-Host ""
}

# Create or update .gitignore
$gitignorePath = ".gitignore"
$gitignoreContent = @"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data files (will be downloaded on Lightning AI)
data/BIOQIC/downloads/*.mat
data/BIOQIC/fem_box/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Zip files
*.zip
"@

Set-Content -Path $gitignorePath -Value $gitignoreContent
Write-Host "Created/Updated .gitignore" -ForegroundColor Green
Write-Host ""

# Check current status
Write-Host "Current Git status:" -ForegroundColor Cyan
git status --short
Write-Host ""

# Ask user for GitHub repo URL
Write-Host "=== GitHub Repository Setup ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Steps to create a GitHub repository:" -ForegroundColor Yellow
Write-Host "1. Go to https://github.com/new" -ForegroundColor White
Write-Host "2. Repository name: MRE-PINN" -ForegroundColor White
Write-Host "3. Set to Public or Private (your choice)" -ForegroundColor White
Write-Host "4. Do NOT initialize with README (we already have files)" -ForegroundColor White
Write-Host "5. Click 'Create repository'" -ForegroundColor White
Write-Host "6. Copy the repository URL (e.g., https://github.com/YourUsername/MRE-PINN.git)" -ForegroundColor White
Write-Host ""

$repoUrl = Read-Host "Enter your GitHub repository URL (or press Enter to skip)"

if ($repoUrl) {
    # Check if remote already exists
    $remotes = git remote
    if ($remotes -contains "origin") {
        Write-Host "Remote 'origin' already exists, updating..." -ForegroundColor Yellow
        git remote set-url origin $repoUrl
    } else {
        Write-Host "Adding remote 'origin'..." -ForegroundColor Yellow
        git remote add origin $repoUrl
    }
    Write-Host "Remote configured!" -ForegroundColor Green
    Write-Host ""

    # Add and commit files
    Write-Host "Adding files to Git..." -ForegroundColor Yellow
    git add .

    Write-Host "Creating commit..." -ForegroundColor Yellow
    $commitMsg = "Add Lightning AI training setup with GPU support"
    git commit -m $commitMsg

    Write-Host "Commit created!" -ForegroundColor Green
    Write-Host ""

    # Ask about branch
    $currentBranch = git branch --show-current
    Write-Host "Current branch: $currentBranch" -ForegroundColor Cyan

    if ($currentBranch -ne "main" -and $currentBranch -ne "master") {
        $response = Read-Host "Rename branch to 'main'? (y/n)"
        if ($response -eq "y") {
            git branch -M main
            $currentBranch = "main"
            Write-Host "Branch renamed to 'main'" -ForegroundColor Green
        }
    }

    # Push to GitHub
    Write-Host ""
    Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
    Write-Host "(You may be asked for GitHub credentials)" -ForegroundColor White

    try {
        git push -u origin $currentBranch
        Write-Host ""
        Write-Host "=== SUCCESS! ===" -ForegroundColor Green
        Write-Host "Your code is now on GitHub!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next steps for Lightning AI:" -ForegroundColor Cyan
        Write-Host "1. Open Lightning AI Studio terminal" -ForegroundColor White
        Write-Host "2. Run: git clone $repoUrl" -ForegroundColor Yellow
        Write-Host "3. Run: cd MRE-PINN/lightning-ai-training" -ForegroundColor Yellow
        Write-Host "4. Open: lightning-ai-simulation-training.ipynb" -ForegroundColor Yellow
        Write-Host ""
    } catch {
        Write-Host ""
        Write-Host "Push failed. Possible reasons:" -ForegroundColor Red
        Write-Host "- Authentication issue (need to setup GitHub credentials)" -ForegroundColor Yellow
        Write-Host "- Network issue" -ForegroundColor Yellow
        Write-Host "- Branch protection rules" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Try pushing manually:" -ForegroundColor White
        Write-Host "  git push -u origin $currentBranch" -ForegroundColor Yellow
    }

} else {
    Write-Host "Skipping GitHub push." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To push later, run:" -ForegroundColor Cyan
    Write-Host "  git remote add origin YOUR_GITHUB_URL" -ForegroundColor Yellow
    Write-Host "  git add ." -ForegroundColor Yellow
    Write-Host "  git commit -m 'Add Lightning AI setup'" -ForegroundColor Yellow
    Write-Host "  git push -u origin main" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Script complete!" -ForegroundColor Green
