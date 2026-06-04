"""Standalone test harness for ``ProcessingEngine.compare_expert``.

Point ``current_run`` and ``other_run`` at two run folders produced by the
engine — each must contain a ``RunInfo.json`` manifest. Comparison artifacts
(side-by-side images) land in a fresh temp directory; the run folders
themselves are never touched.

Usage: edit the three constants below, then ``python test_expert.py``.
"""
import json
import tempfile

from src.processing_engine import ProcessingEngine

current_run = ""
other_run = ""
metric_name = "ENTRANCE_VECTORS"

engine = ProcessingEngine()
output_dir = tempfile.mkdtemp(prefix="compare_expert_")
print(f"Comparison artifacts: {output_dir}")

result = engine.compare_expert(
    metric_name,
    current_run,
    other_run,
    output_dir=output_dir,
)
print(json.dumps(json.loads(result), indent=2))
