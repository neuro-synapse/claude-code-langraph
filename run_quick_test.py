#!/usr/bin/env python3
"""
Quick test for Meeting Transcript Processing - just theme extraction step.
"""

from __future__ import annotations
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
        "model": "gemini-1.5-pro",  # Using 1.5-pro instead of 2.5-pro
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

print("🚀 Starting LangGraph meeting processing workflow...")

try:
    result = app.invoke(state, config=config)
    
    print("\n✅ Workflow completed successfully!")
    print("\n📊 Results Summary:")
    print(f"Themes extracted: {len(result.get('themes', []))}")
    print(f"Theme notes generated: {len(result.get('theme_notes', {}))}")
    print(f"Actions identified: {len(result.get('actions', []))}")
    print(f"Decisions recorded: {len(result.get('decisions', []))}")
    print(f"Open questions: {len(result.get('open_questions', []))}")
    
    final_doc = result.get("final_doc", {})
    if final_doc:
        exec_summary = final_doc.get("exec_summary", {})
        body = final_doc.get("body", {})
        print(f"Executive summary words: {exec_summary.get('word_count', 0)}")
        print(f"Body words: {body.get('word_count', 0)}")
        
        # Show a sample of the exec summary
        if exec_summary.get('text'):
            print(f"\n📝 Executive Summary Sample:")
            print(f"'{exec_summary['text'][:100]}...'")
    
except Exception as e:
    print(f"❌ Workflow failed: {e}")
    import traceback
    traceback.print_exc()