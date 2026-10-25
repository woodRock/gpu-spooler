
if ! command -v nvidia-smi >/dev/null 2>&1; then echo "NO_NVIDIA_SMI"; exit 0; fi
GPU_OUT=$(nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu,compute_mode \
    --format=csv,noheader,nounits 2>/dev/null) || { echo "ERROR: nvidia-smi failed"; exit 1; }
PMON_OUT=$(nvidia-smi pmon -c 1 2>/dev/null)
APPS_OUT=$(nvidia-smi --query-compute-apps=gpu_index,pid --format=csv,noheader 2>/dev/null)
printf 'GPU_INFO\n%s\nPMON_INFO\n%s\nAPPS_INFO\n%s\nPROC_INFO\n' "$GPU_OUT" "$PMON_OUT" "$APPS_OUT"
PIDS=$( { printf '%s\n' "$PMON_OUT" | awk '!/^#/ && $2 ~ /^[0-9]+$/ {print $2}'
          printf '%s\n' "$APPS_OUT" | awk -F', *' '$2 ~ /^[0-9]+$/ {print $2}'; } | sort -u | tr '\n' ',')
PIDS="${PIDS%,}"
if [ -n "$PIDS" ]; then
    ps -o pid=,user= -p "$PIDS" 2>/dev/null || true
fi
printf 'END\n'
