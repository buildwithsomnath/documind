import sys
from pathlib import Path

# Ensure backend and documind root are on sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
documind_root = backend_dir.parent
if str(documind_root) not in sys.path:
    sys.path.insert(0, str(documind_root))

from app import app, init_db

if __name__ == "__main__":
    # Attempt to initialize tables on startup
    try:
        init_db()
    except Exception as e:
        app.logger.warning(
            f"Could not auto-initialize database on startup (PostgreSQL may still be starting): {e}"
        )

    app.run(host="127.0.0.1", port=5000, debug=True)
