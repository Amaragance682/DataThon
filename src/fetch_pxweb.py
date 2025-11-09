import argparse, json, requests, pathlib, sys

def get_meta(url: str) -> dict:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def post_pxweb(url: str, query: dict) -> dict:
    r = requests.post(url, json=query, timeout=60)
    if r.status_code >= 400:
        sys.stderr.write(f"\nPXWeb error {r.status_code}:\n{r.text}\n")
        r.raise_for_status()
    return r.json()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--query", required=False)
    p.add_argument("--out", required=False)
    p.add_argument("--inspect", action="store_true", help="Print variable codes & allowed values, then exit")
    args = p.parse_args()

    if args.inspect:
        meta = get_meta(args.url)
        print("VARIABLES:")
        for v in meta.get("variables", []):
            print(f"- code={v['code']}  first_values={v['values'][:5]}")
        return

    raw = json.loads(pathlib.Path(args.query).read_text(encoding="utf-8"))
    payload = raw.get("queryObj", raw)       # accept both shapes
    data = post_pxweb(args.url, payload)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved raw response to {args.out}")

if __name__ == "__main__":
    main()