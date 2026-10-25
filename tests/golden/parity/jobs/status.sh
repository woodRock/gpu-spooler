for spec in 1-aaaaaa:0 22-bbbbbb:1048576 '3-c'"'"'c:5'; do
  k=${spec%%:*}; off=${spec##*:}; D="$HOME/.gpu-spooler/jobs/$k"
  printf '@@JOB %s\n' "$k"
  if [ -f "$D/rc" ]; then printf 'STATE done %s\n' "$(cat "$D/rc")"
  elif [ ! -d "$D" ]; then echo 'STATE missing'
  elif [ -f "$D/pid" ] && kill -0 "$(cat "$D/pid")" 2>/dev/null; then echo 'STATE running'
  elif [ -f "$D/rc" ]; then printf 'STATE done %s\n' "$(cat "$D/rc")"
  else echo 'STATE gone'; fi
  printf 'SIZE %s\n' "$(wc -c < "$D/out.log" 2>/dev/null | tr -d ' ' || echo 0)"
  printf 'DATA '; tail -c +$((off+1)) "$D/out.log" 2>/dev/null | head -c 1048576 | base64 | tr -d '\n'; echo
done
echo '@@END'
