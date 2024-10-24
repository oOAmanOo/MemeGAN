$csvFile = "90pin_gpu_stats.csv"

# 檢查是否已經存在 csv 檔案，沒有的話就新增檔案並加入標題行
if (-not (Test-Path $csvFile)) {
    nvidia-smi --query-gpu=index,gpu_name,memory.total,memory.used,memory.free,temperature.gpu,pstate,utilization.gpu,utilization.memory --format=csv | Out-File -Append -Encoding utf8 $csvFile
}

while ($true) {
    nvidia-smi --query-gpu=index,gpu_name,memory.total,memory.used,memory.free,temperature.gpu,pstate,utilization.gpu,utilization.memory --format=csv,noheader | Out-File -Append -Encoding utf8 $csvFile
    Start-Sleep -Seconds 3
}
