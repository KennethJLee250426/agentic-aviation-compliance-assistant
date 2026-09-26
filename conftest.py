import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is in sys.path
root_dir = Path(__file__).parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Load environment variables from .env before any app modules are imported
load_dotenv(root_dir / ".env")
if (len(os.environ.get("AUTH_TOKEN", "").strip()) < 32
        or os.environ.get("AUTH_TOKEN", "").strip() == "change-me-in-production"):
    os.environ["AUTH_TOKEN"] = "test-only-token-that-is-not-used-outside-pytest"