"""Probe what it would take to add WebShop transfer experiments.

This script does NOT install WebShop — it just reports what's missing
and how much work would be required. WebShop has heavy non-Python
dependencies (Java/Selenium) and a curated product catalog (~6 GB),
so a full setup is out-of-scope for the gate experiment.
"""
import importlib
import shutil
from pathlib import Path

print("=== WebShop transfer probe ===\n")

# 1. Check pip-discoverable packages
print("--- pip packages ---")
for m in ("webshop", "webshop_lite", "webshop_minimal"):
    try:
        x = importlib.import_module(m)
        print(f"  {m}: installed at {x.__file__}")
    except ImportError:
        print(f"  {m}: not installed")

# 2. Java?
java = shutil.which("java")
print(f"\n--- system tools ---")
print(f"  java:      {java or 'MISSING'}")
print(f"  selenium:  {'install ok' if importlib.util.find_spec('selenium') else 'MISSING'}")

# 3. Source repos that might be on disk
print(f"\n--- on-disk WebShop repos ---")
candidates = [
    Path.home() / "WebShop",
    Path.home() / "webshop",
    Path("/data1/zhiyuanzhai/WebShop"),
    Path("/data1/zhiyuanzhai/webshop"),
    Path("/data/WebShop"),
]
found = [p for p in candidates if p.exists()]
if found:
    for p in found:
        print(f"  found: {p}")
else:
    print("  no on-disk repos in standard locations")

# 4. Estimated cost
print(f"\n--- cost estimate to add WebShop transfer experiment ---")
print("  - clone github.com/princeton-nlp/WebShop  (~30 min, requires git LFS)")
print("  - install conda env (selenium, java, scraping deps)  (~30 min)")
print("  - download products.json (~6 GB) + curated split  (~20 min)")
print("  - port src/env.py to a WebShop GroupEnv  (~2 hours)")
print("  - port src/prompts.py for WebShop ReAct format  (~1 hour)")
print("  - re-run 100 rollouts with G=8 + compute metrics  (~30 min on 4 GPUs)")
print("  - re-fit gate thresholds + write up  (~1 hour)")
print("  TOTAL: ~5-6 hours of focused work; mostly env porting, not")
print("         a genuine generalisation result.")
print()
print("Recommendation: defer to follow-up work; the ALFWorld + G=16")
print("results in the paper already cover within-distribution and")
print("group-size robustness.")
