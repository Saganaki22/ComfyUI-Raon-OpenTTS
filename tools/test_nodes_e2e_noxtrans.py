"""Run test_nodes_e2e.py with x_transformers blocked, to prove the vendored rotary
is what actually executes inside the model forward. Usage same as test_nodes_e2e.py."""

import sys
import types

block = types.ModuleType("x_transformers")
block.__path__ = []
sys.modules["x_transformers"] = block

from pathlib import Path

sys.argv = [str(Path(__file__))] + sys.argv[1:]
exec(compile(Path(Path(__file__).with_name("test_nodes_e2e.py")).read_text(encoding="utf-8"),
             "test_nodes_e2e.py", "exec"), {"__name__": "__main__", "__file__": str(Path(__file__).with_name("test_nodes_e2e.py"))})
