
import base64, json, os, sys, tarfile
req = json.loads(base64.b64decode(sys.argv[1]))
os.chdir(os.path.expanduser(req["base"]))
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as tf:
    for rel in req["members"]:
        tf.add(rel, recursive=False)
