D="$HOME/.gpu-spooler/jobs/12-a1b2c3"
[ -f "$D/pid" ] || { echo NO_JOB; exit 0; }
P=$(cat "$D/pid")
if kill -TERM -- -"$P" 2>/dev/null; then
  echo KILLED
  ( trap '' HUP; i=0
    while [ $i -lt 10 ]; do sleep 1; kill -0 -- -"$P" 2>/dev/null || exit 0; i=$((i+1)); done
    kill -KILL -- -"$P" 2>/dev/null ) > /dev/null 2>&1 < /dev/null &
else
  echo NOT_RUNNING
fi
