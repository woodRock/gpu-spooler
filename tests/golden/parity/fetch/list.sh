command -v python3 >/dev/null 2>&1 || { echo "python3 not found" >&2; exit 3; }
python3 - WyIvdm9sL21lL3Byb2ovcmVzdWx0cyIsICJ+L3ggeSIsICIvdm9sL2l0J3MiXQ== <<'TS_PY'

import base64, hashlib, json, os, sys
paths = json.loads(base64.b64decode(sys.argv[1]))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
out = []
for path in paths:
    p = os.path.expanduser(path)
    if os.path.islink(p):
        out.append({"kind": "link", "target": os.readlink(p)})
    elif os.path.isfile(p):
        out.append({"kind": "file", "sha": sha(p)})
    elif os.path.isdir(p):
        files, links = {}, {}
        for root, dirs, names in os.walk(p):        # does not descend into linked dirs
            for n in dirs + names:
                full = os.path.join(root, n)
                rel = os.path.relpath(full, p)
                if os.path.islink(full):
                    links[rel] = os.readlink(full)
                elif os.path.isfile(full):
                    files[rel] = sha(full)
        out.append({"kind": "dir", "files": files, "links": links})
    else:
        out.append({"kind": "missing"})
print("TS_FETCH " + json.dumps(out))

TS_PY
