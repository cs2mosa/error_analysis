import sys
from pathlib import Path

# make the `ea` package importable when pytest runs from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
