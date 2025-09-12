from __future__ import annotations

import os
from agent import app

transcript = """
[00:00] Alice: Kickoff and goals. We need clarity on Q4 scope.
[00:03] Bob: Scope options: A (ship MVP), B (expand features). KPI at 99.9% uptime.
[00:07] Carol: Dependencies on Data team; PRD v2 is in Drive.
[00:12] Bob: Decision proposal: Ship MVP v1 by Oct 15, then iterate.
[00:15] Alice: Action items: Bob drafts plan; Carol confirms Data API SLA.
[00:20] Dave: Risk: test infra flaky; need to upgrade CI.
"""

config = {
    "configurable": {
        "require_approval": False,
        "require_final_approval": False,
        "provider": "google",
        "model": "gemini-2.5-pro",
        "temperature": 0.3,
    }
}

state = {
    "messages": [],
    "transcript": transcript,
    "metadata": {
        "attendees": ["Alice", "Bob", "Carol", "Dave"],
        "contact": "pm@example.com",
        "date": "2025-09-12",
        "duration": "30m",
    },
}

try:
    result = app.invoke(state, config=config)
except Exception as e:
    # Fallback to 1.5 pro if 2.5 is unavailable
    config["configurable"]["model"] = "gemini-1.5-pro"
    result = app.invoke(state, config=config)

final_doc = result.get("final_doc", {})
print("themes:", len(result.get("themes", [])))
print("theme_notes:", len(result.get("theme_notes", {})))
print("actions:", len(result.get("actions", [])))
print("decisions:", len(result.get("decisions", [])))
print("open_questions:", len(result.get("open_questions", [])))
print("exec_summary_words:", (final_doc.get("exec_summary") or {}).get("word_count"))
print("body_words:", (final_doc.get("body") or {}).get("word_count"))
print("appendix_keys:", list((final_doc.get("appendices") or {}).keys()))

