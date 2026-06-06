# 一键运行微调流程（PowerShell）
# 用法: .\training\scripts\run_all.ps1
# 可选: .\training\scripts\run_all.ps1 -MaxRows 500 -Task refiner

param(
    [int]$MaxRows = 0,
    [ValidateSet("refiner", "visual", "all")]
    [string]$Task = "refiner"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)

Write-Host "== 1/4 安装微调依赖 ==" -ForegroundColor Cyan
pip install -r training/requirements.txt

Write-Host "== 2/4 下载数据集 ==" -ForegroundColor Cyan
$dlArgs = @("training/data/download.py")
if ($MaxRows -gt 0) { $dlArgs += @("--max-rows", $MaxRows) }
python @dlArgs

Write-Host "== 3/4 构建 SFT ==" -ForegroundColor Cyan
python training/data/build_sft.py

Write-Host "== 4/4 LoRA 训练 ==" -ForegroundColor Cyan
if ($Task -eq "refiner" -or $Task -eq "all") {
    python training/train/train_lora.py --config training/configs/train_refiner.yaml
}
if ($Task -eq "visual" -or $Task -eq "all") {
    python training/train/train_lora.py --config training/configs/train_visual.yaml
}

Write-Host "完成。适配器在 training/outputs/ 下。" -ForegroundColor Green
