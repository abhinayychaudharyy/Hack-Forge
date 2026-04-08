import sys
from pathlib import Path

# Add project root to sys.path for server imports
sys.path.append(str(Path(__file__).parent))

from server.app import app, main

if __name__ == "__main__":
    main()
