#!/usr/bin/env python3
"""
Test partial execution up to theme expansion to verify fixes work.
"""

from __future__ import annotations
import agent as ag

# Monkeypatch to stop after expand_themes to avoid timeout
original_add_edge = ag.graph_builder.add_edge

def limited_add_edge(from_node, to_node):
    # Skip connections after expand_themes to test just the first steps
    if from_node == "expand_themes":
        ag.graph_builder.add_edge(from_node, ag.END)
        return
    original_add_edge(from_node, to_node)

# Apply monkeypatch
ag.graph_builder = ag.StateGraph(ag.MeetingState)
ag.graph_builder.add_node("preprocess", ag.preprocess)
ag.graph_builder.add_node("theme_extract", ag.extract_themes)
ag.graph_builder.add_node("expand_themes", ag.expand_themes)
ag.graph_builder.add_edge(ag.START, "preprocess")
ag.graph_builder.add_edge("preprocess", "theme_extract")
ag.graph_builder.add_edge("expand_themes", ag.END)

# Rebuild the app with limited workflow
limited_app = ag.graph_builder.compile()

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
        "provider": "google",
        "model": "gemini-1.5-pro",
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

print("🚀 Testing theme extraction and expansion...")

try:
    result = limited_app.invoke(state, config=config)
    
    print("\n✅ Partial workflow completed!")
    print(f"Themes: {len(result.get('themes', []))}")
    print(f"Theme notes: {len(result.get('theme_notes', {}))}")
    
    # Show sample theme
    themes = result.get('themes', [])
    if themes:
        sample_theme = themes[0]
        print(f"\n📝 Sample theme: {sample_theme.get('title', 'N/A')}")
        print(f"Tier: {sample_theme.get('tier', 'N/A')}")
        print(f"Time %: {sample_theme.get('time_percent', 'N/A')}")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()