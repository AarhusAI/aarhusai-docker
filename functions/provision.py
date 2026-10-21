import json, os, time, urllib.request, urllib.error

URL = os.environ["OWUI_URL"]

def call(method, path, data=None, token=None):
    req = urllib.request.Request(
        URL + path, method=method,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read() or "null")

for _ in range(60):  # wait for Open WebUI
    try: call("GET", "/health"); break
    except Exception: time.sleep(5)

token = call("POST", "/api/v1/auths/signin",
             {"email": os.environ["OWUI_EMAIL"],
              "password": os.environ["OWUI_PASSWORD"]})["token"]

FUNCS = {"presidio_guard": "Presidio PII Guard",
         "presidio_anonymize": "Presidio PII Anonymizer"}

for fid, name in FUNCS.items():
    content = open(f"/functions/{fid}.py").read()
    body = {"id": fid, "name": name, "content": content,
            "meta": {"description": name}}
    try:
        call("POST", "/api/v1/functions/create", body, token)
        call("POST", f"/api/v1/functions/id/{fid}/toggle", token=token)  # activate
        print("created", fid)
    except urllib.error.HTTPError:
        call("POST", f"/api/v1/functions/id/{fid}/update", body, token)  # already exists
        print("updated", fid)
