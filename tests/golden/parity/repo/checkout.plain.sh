command -v git >/dev/null 2>&1 || PATH="$(bash -lc 'printf %s "$PATH"' </dev/null 2>/dev/null):$PATH"
cd /vol/me/proj || exit 1
git checkout -q --detach 0123456789abcdef0123456789abcdef01234567 || exit 5
echo SYNCED
