import sys
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import json
import urllib.error
import urllib.request
from app.config import Config


def check_models():
    api_key = Config.GEMINI_API_KEY
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set in .env")
        return

    masked_key = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
    print(f"Testing Gemini API connectivity with key: {masked_key} (length: {len(api_key)})")

    for version in ["v1beta", "v1"]:
        url = f"https://generativelanguage.googleapis.com/{version}/models?key={api_key}"
        print(f"\n--- Checking {version} Endpoint: {url.replace(api_key, masked_key)} ---")
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                gen_models = [
                    m["name"].replace("models/", "")
                    for m in models
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
                print(f"Success! Found {len(gen_models)} models supporting generateContent on {version}:")
                for m in gen_models:
                    print(f"  • {m}")
                return

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8") if e.fp else ""
            print(f"HTTP Error {e.code}: {e.reason}")
            try:
                err_json = json.loads(err_body)
                print("Details:", json.dumps(err_json, indent=2))
            except Exception:
                print("Details:", err_body)

        except Exception as e:
            print(f"Error connecting: {e}")

    print("\nTip: A standard Google AI Studio API key usually begins with 'AIzaSy...'.")
    print("Obtain a free direct key at: https://aistudio.google.com/app/apikey")


if __name__ == "__main__":
    check_models()
