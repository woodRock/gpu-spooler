command -v python3 >/dev/null 2>&1 || { echo "python3 not found" >&2; exit 3; }
python3 - eyJiYXNlIjogIi92b2wvbWUvcHJvai9yZXN1bHRzIiwgIm1lbWJlcnMiOiBbImEuY3N2IiwgInN1Yi9iIGMudHh0IiwgImxpbmsiXX0= <<'TS_PY'

import base64, json, os, sys, tarfile
req = json.loads(base64.b64decode(sys.argv[1]))
os.chdir(os.path.expanduser(req["base"]))
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as tf:
    for rel in req["members"]:
        tf.add(rel, recursive=False)

TS_PY
