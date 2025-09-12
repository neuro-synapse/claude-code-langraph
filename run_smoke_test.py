"""
Smoke test for Meeting Transcript Processing graph without calling external LLMs.

It monkeypatches agent._get_llm to a stub that returns structured
outputs expected by each stage. This validates graph wiring, state
updates, and final document assembly.
"""

from __future__ import annotations

from typing import Any, List, Dict

import agent as ag


class _StubInvoker:
    def __init__(self, model_cls, stage: str):
        self.model_cls = model_cls
        self.stage = stage

    def invoke(self, messages: List[Any]):
        # Theme extraction
        if self.model_cls is ag.ThemeList:
            themes = []
            tiers = ["PRIMARY", "PRIMARY", "SECONDARY", "SECONDARY", "TANGENTIAL"]
            for i in range(5):
                themes.append(
                    ag.Theme(
                        id=i + 1,
                        title=f"Theme {i+1} Title",
                        summary=f"Summary for theme {i+1}.",
                        time_percent=20.0,
                        participants=["Alice", "Bob" if i % 2 == 0 else "Carol"],
                        tier=tiers[i],
                        flags=ag.ThemeFlags(
                            decision=(i % 2 == 0), action=(i % 3 == 0), question=(i == 4), dependency=False
                        ),
                    )
                )
            return ag.ThemeList(themes=themes)

        # Theme expansion
        if self.model_cls is ag.ThemeNotes:
            # Pull theme id from message text heuristically
            text = "\n".join([getattr(m, "content", "") for m in messages])
            tid = 1
            title = "Unknown"
            for line in text.splitlines():
                if line.startswith("Focus on Theme "):
                    try:
                        tid = int(line.split(":")[0].split()[-1])
                        title = line.split(": ", 1)[-1].strip()
                    except Exception:
                        pass
                    break
            notes_md = (
                "Overview: This section covers detailed discussion points for the theme.\n\n"
                "- Main Discussion Points\n"
                "  • Primary point A\n"
                "    ◦ Supporting detail A1\n"
                "      ▪ \"Exact quote illustrating A1\" - Alice\n"
                "  • Primary point B\n"
                "    ◦ Supporting detail B1 with KPI 99.9% on 2025-01-01\n"
                "      ▪ \"We should track this weekly\" - Bob\n"
                "- Key Insights or Decisions\n"
                "  • Decision noted and rationale provided.\n"
                "- Questions Raised but Not Answered\n"
                "  • What is the final budget source?\n"
                "- Related Resources Mentioned\n"
                "  • Dashboard link and PRD document.\n"
            )
            # Ensure 300-500 words roughly by padding
            base_words = len(notes_md.split())
            pad = " lorem" * max(0, 320 - base_words)
            notes_md = notes_md + pad
            return ag.ThemeNotes(
                theme_id=tid,
                title=title,
                notes_md=notes_md,
                word_count=len(notes_md.split()),
                quotes=[ag.Quote(text="Exact quote illustrating A1", speaker="Alice")],
                related_resources=[ag.ResourceRef(label="Dashboard", url="https://example.com")],
                key_insights_or_decisions="Decision captured.",
                open_questions="Budget source pending.",
            )

        # Outcomes extraction
        if self.model_cls is ag.Outcomes:
            actions = [
                ag.ActionItem(what=f"Do task {i}", owner=("Alice" if i % 2 == 0 else "Bob"), due="Short-term", why="Required", priority="SHORT_TERM", depends_on=[])
                for i in range(1, 7)
            ]
            decisions = [ag.Decision(statement=f"Decision {i}", rationale="Because", impact="Medium") for i in range(1, 4)]
            open_q = [ag.OpenQuestion(question=f"Open question {i}", context="Context", suggested_owner="Carol") for i in range(1, 3)]
            return ag.Outcomes(actions=actions, decisions=decisions, open_questions=open_q)

        # QC result (pass-through with minimal adjustments)
        if self.model_cls is ag.QCResult:
            themes = []
            for i in range(1, 6):
                themes.append(
                    ag.Theme(
                        id=i,
                        title=f"Theme {i} Title",
                        summary=f"Summary for theme {i}.",
                        time_percent=20.0,
                        participants=["Alice", "Bob"],
                        tier=("PRIMARY" if i < 3 else ("SECONDARY" if i < 5 else "TANGENTIAL")),
                        flags=ag.ThemeFlags(),
                    )
                )
            theme_notes = []
            for i in range(1, 5):
                theme_notes.append(
                    ag.ThemeNotes(
                        theme_id=i,
                        title=f"Theme {i} Title",
                        notes_md=f"Notes for theme {i}...",
                        word_count=320,
                        quotes=[ag.Quote(text="Example quote", speaker="Alice")],
                        related_resources=[ag.ResourceRef(label="Doc", url="https://example.com/doc")],
                    )
                )
            actions = [ag.ActionItem(what="Do task 1", owner="Alice", due="Immediate", priority="IMMEDIATE")]
            decisions = [ag.Decision(statement="Ship v1", rationale="Timeline", impact="High")]
            open_q = [ag.OpenQuestion(question="Who owns QA?", context="Staffing", suggested_owner="Bob")]
            out = ag.QCOutput(
                themes=themes,
                theme_notes=theme_notes,
                actions=actions,
                decisions=decisions,
                open_questions=open_q,
                resources=[ag.ResourceRef(label="Dashboard", url="https://example.com")],
                metadata={"date": "2025-01-02", "attendees": ["Alice", "Bob", "Carol"], "duration": "60m"},
            )
            return ag.QCResult(corrections=out, findings=[ag.QCFinding(issue="Minor name fix", fix="Bob -> Robert", severity="minor")])

        # Final exec summary
        if self.model_cls is ag.ExecSummary:
            text = "Purpose: Align on roadmap and decisions. Outcome: Go for v1. Key decisions: Ship v1; adopt metric. Critical actions: Task 1 (Alice), Task 2 (Bob). Next steps: finalize scope this week."
            return ag.ExecSummary(text=text, word_count=len(text.split()))

        # Final body
        if self.model_cls is ag.FinalBody:
            body_md = (
                "# Detailed Notes\n\n"
                "Themes in order with transitions.\n\n"
                "### Theme 1 (PRIMARY)\nDetails...\n\n"
                "### Theme 2 (PRIMARY)\nMore details...\n"
            )
            return ag.FinalBody(detailed_notes_md=body_md, word_count=len(body_md.split()))

        raise RuntimeError(f"Stub does not handle model {self.model_cls}")


class _StubLLM:
    def __init__(self, stage: str = ""):
        self.stage = stage

    def with_structured_output(self, model_cls):
        return _StubInvoker(model_cls, self.stage)


def _stub_get_llm(temperature: float = 0.0, model: str | None = None, provider: str | None = None):
    return _StubLLM()


def main():
    # Monkeypatch LLM resolver
    ag._get_llm = _stub_get_llm  # type: ignore

    transcript = "Alice: Kickoff. Bob: We need to finalize the scope. Carol: Metrics at 99.9%."
    state = {
        "messages": [],
        "transcript": transcript,
        "metadata": {"attendees": ["Alice", "Bob", "Carol"], "contact": "pm@example.com"},
    }
    cfg = {"configurable": {"require_approval": False, "require_final_approval": False}}
    result = ag.app.invoke(state, config=cfg)

    print("=== Smoke Test Results ===")
    print("Themes:", len(result.get("themes", [])))
    print("Theme Notes:", len(result.get("theme_notes", {})))
    print("Actions:", len(result.get("actions", [])))
    print("Decisions:", len(result.get("decisions", [])))
    print("Open Questions:", len(result.get("open_questions", [])))
    final_doc: Dict[str, Any] = result.get("final_doc", {})
    print("Exec Summary words:", (final_doc.get("exec_summary") or {}).get("word_count"))
    print("Body words:", (final_doc.get("body") or {}).get("word_count"))
    appx = final_doc.get("appendices") or {}
    print("Appendix sections:", list(appx.keys()))


if __name__ == "__main__":
    main()

