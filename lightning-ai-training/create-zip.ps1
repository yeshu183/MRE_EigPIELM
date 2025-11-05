# PowerShell script to create a Lightning AI-ready zip file
# This excludes data files and other unnecessary files to reduce size

$sourcePath = Split-Path -Parent $PSScriptRoot
$zipPath = Join-Path (Split-Path -Parent $sourcePath) "MRE-PINN-lightning.zip"

Write-Host "Creating Lightning AI zip file..." -ForegroundColor Green
Write-Host "Source: $sourcePath" -ForegroundColor Cyan
Write-Host "Destination: $zipPath" -ForegroundColor Cyan

# Folders and files to exclude
$excludePatterns = @(
    "data\*",           # Exclude all data (will download fresh on Lightning AI)
    ".git\*",           # Exclude git repository
    "__pycache__\*",    # Exclude Python cache
    "*.pyc",            # Exclude compiled Python files
    ".ipynb_checkpoints\*",  # Exclude Jupyter checkpoints
    "*.zip",            # Exclude existing zip files
    ".vscode\*",        # Exclude VS Code settings
    ".idea\*",          # Exclude PyCharm settings
    "*.egg-info\*",     # Exclude package info
    "dist\*",           # Exclude distribution files
    "build\*"           # Exclude build files
)

# Create a temporary directory for files to zip
$tempDir = Join-Path $env:TEMP "MRE-PINN-temp-$(Get-Random)"
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

try {
    Write-Host "`nCopying files (excluding data and cache)..." -ForegroundColor Yellow

    # Copy everything except excluded patterns
    $filesToCopy = Get-ChildItem -Path $sourcePath -Recurse -File | Where-Object {
        $file = $_
        $relativePath = $file.FullName.Substring($sourcePath.Length + 1)

        # Check if file matches any exclude pattern
        $shouldExclude = $false
        foreach ($pattern in $excludePatterns) {
            if ($relativePath -like $pattern) {
                $shouldExclude = $true
                break
            }
        }
        -not $shouldExclude
    }

    $totalFiles = $filesToCopy.Count
    $current = 0

    foreach ($file in $filesToCopy) {
        $current++
        $relativePath = $file.FullName.Substring($sourcePath.Length + 1)
        $destPath = Join-Path $tempDir "MRE-PINN\$relativePath"
        $destFolder = Split-Path -Parent $destPath

        if (!(Test-Path $destFolder)) {
            New-Item -ItemType Directory -Path $destFolder -Force | Out-Null
        }

        Copy-Item -Path $file.FullName -Destination $destPath -Force

        if ($current % 10 -eq 0) {
            Write-Progress -Activity "Copying files" -Status "$current of $totalFiles" -PercentComplete (($current / $totalFiles) * 100)
        }
    }

    Write-Progress -Activity "Copying files" -Completed
    Write-Host "Copied $totalFiles files" -ForegroundColor Green

    # Create the zip file
    Write-Host "`nCreating zip archive..." -ForegroundColor Yellow
    if (Test-Path $zipPath) {
        Remove-Item $zipPath -Force
    }

    Compress-Archive -Path (Join-Path $tempDir "MRE-PINN") -DestinationPath $zipPath -CompressionLevel Optimal

    $zipSize = (Get-Item $zipPath).Length / 1MB
    Write-Host "`nSuccess! Created: $zipPath" -ForegroundColor Green
    Write-Host "Size: $([math]::Round($zipSize, 2)) MB" -ForegroundColor Cyan
    Write-Host "`nThis zip is ready to upload to Lightning AI!" -ForegroundColor Green
    Write-Host "Data will be downloaded fresh on Lightning AI (saves upload time)." -ForegroundColor Yellow

} finally {
    # Clean up temp directory
    if (Test-Path $tempDir) {
        Remove-Item -Path $tempDir -Recurse -Force
    }
}

Write-Host "`nNext steps:" -ForegroundColor Magenta
Write-Host "1. Go to lightning.ai and create a new Studio with T4 GPU" -ForegroundColor White
Write-Host "2. Upload MRE-PINN-lightning.zip" -ForegroundColor White
Write-Host "3. Extract: unzip MRE-PINN-lightning.zip" -ForegroundColor White
Write-Host "4. Navigate: cd MRE-PINN/lightning-ai-training" -ForegroundColor White
Write-Host "5. Open and run: lightning-ai-simulation-training.ipynb" -ForegroundColor White
