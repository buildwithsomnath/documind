import sys
from pathlib import Path

# Ensure backend and documind root are in sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
documind_root = backend_dir.parent
if str(documind_root) not in sys.path:
    sys.path.insert(0, str(documind_root))

from app import app, init_db


def main():
    args = sys.argv[1:]
    command = args[0] if args else "run"

    if command in ("run", "runserver", "start"):
        print("Starting DocuMind development server on http://127.0.0.1:5000 ...")
        try:
            init_db()
        except Exception as e:
            print(f"Notice: Database auto-initialization skipped or deferred: {e}")
        app.run(host="127.0.0.1", port=5000, debug=True)

    elif command in ("init-db", "init_db"):
        print("Initializing PostgreSQL database schema and pgvector extension...")
        init_db()
        print("Database schema initialized successfully.")

    elif command == "test":
        import pytest
        print("Running DocuMind test suite...")
        sys.exit(pytest.main(["tests", "-v"]))

    elif command in ("evaluate", "eval", "benchmark"):
        from evaluation.run_evaluation import main as run_eval_main
        run_eval_main()

    else:
        print(f"Unknown command: '{command}'")
        print("\nUsage:")
        print("  python manage.py run        # Start Flask development server")
        print("  python manage.py init-db    # Create tables and pgvector extension")
        print("  python manage.py test       # Run pytest test suite")
        print("  python manage.py evaluate   # Run RAG evaluation benchmark")


if __name__ == "__main__":
    main()
