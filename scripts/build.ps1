# Build exe with PyInstaller (ASCII-only to avoid encoding issues)
$ErrorActionPreference = "Stop"

# Ensure we run from repo root
$root = Split-Path -Parent $PSScriptRoot
if (-not $root) { $root = Get-Location }
Set-Location $root

python --version

# Collect Qt/SoundFile/Numpy assets; include ffmpeg if available
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
$ffprobe = Get-Command ffprobe -ErrorAction SilentlyContinue

# Try to locate ffmpeg/ffprobe in common Winget location if not in PATH
if (-not $ffmpeg) {
  try {
    $base = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'
    if (Test-Path $base) {
      $ffexe = Get-ChildItem -Path $base -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1
      if ($ffexe) { $ffmpeg = @{ Path = $ffexe.FullName } }
    }
  } catch {}
}
if (-not $ffprobe) {
  try {
    $base = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'
    if (Test-Path $base) {
      $fpexe = Get-ChildItem -Path $base -Recurse -Filter ffprobe.exe -ErrorAction SilentlyContinue | Select-Object -First 1
      if ($fpexe) { $ffprobe = @{ Path = $fpexe.FullName } }
    }
  } catch {}
}

# Clean build artifacts to guarantee fresh exe
try { Remove-Item -Recurse -Force (Join-Path $root 'build') -ErrorAction SilentlyContinue } catch {}
try { Remove-Item -Recurse -Force (Join-Path $root 'dist') -ErrorAction SilentlyContinue } catch {}

$pyArgs = @(
  '--noconfirm',
  '--clean',
  '--onefile',
  '--windowed',
  '--name', 'AudioTransformer',
  '--paths', 'src',
  '--collect-all', 'PySide6',
  '--collect-all', 'soundfile',
  '--collect-submodules', 'numpy',
  # Единый UI (ui2)
  '--hidden-import', 'app_ui.main_window',
  '--hidden-import', 'app_ui.shared',
  '--hidden-import', 'ui2.main_window',
  '--hidden-import', 'src.ui2.main_window',
  # Processing modules
  '--hidden-import', 'src.processing.audio_ops',
  '--hidden-import', 'src.processing.fwht',
  '--runtime-hook', 'hooks/runtime_ffmpeg.py',
  'src/app.py'
)

if ($ffmpeg) {
  Write-Host "Including ffmpeg: $($ffmpeg.Path)"
  $pyArgs = @('--add-binary', ("{0};." -f $ffmpeg.Path)) + $pyArgs
  # Try include ffprobe next to ffmpeg or via PATH
  if (-not $ffprobe) {
    $cand = $ffmpeg.Path -replace 'ffmpeg\.exe$', 'ffprobe.exe'
    if (Test-Path $cand) { $ffprobe = @{ Path = $cand } }
  }
}
if ($ffprobe) {
  Write-Host "Including ffprobe: $($ffprobe.Path)"
  $pyArgs = @('--add-binary', ("{0};." -f $ffprobe.Path)) + $pyArgs
}

# Run PyInstaller and tee output to log
$log = Join-Path $root 'pyinstaller_build.log'
python -m PyInstaller @pyArgs *>&1 | Tee-Object -FilePath $log

if (Test-Path (Join-Path $root 'dist/AudioTransformer.exe')) {
  Write-Host "Build complete: dist/AudioTransformer.exe"
} else {
  Write-Error "Build failed. See $log"
  exit 1
}
