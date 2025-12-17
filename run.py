# Entry-point runner for refactored code
from v3_project import main

if __name__ == "__main__":
    main.main() if hasattr(main, "main") else None
