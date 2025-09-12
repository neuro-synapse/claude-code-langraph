from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

from agent import app


def final_doc_to_markdown(final_doc: Dict[str, Any], metadata: Dict[str, Any] | None = None) -> str:
    meta = metadata or {}
    es = final_doc.get("exec_summary", {}) or {}
    body = final_doc.get("body", {}) or {}
    appx = final_doc.get("appendices", {}) or {}
    footer = final_doc.get("footer", {}) or {}

    lines = []
    title = meta.get("title") or "Meeting Summary (AI-generated)"
    date = meta.get("date") or footer.get("generation_date") or ""
    lines.append(f"# {title}")
    if date:
        lines.append(f"_Date: {date}_\n")

    # Metadata block (optional)
    attendees = meta.get("attendees")
    duration = meta.get("duration")
    if attendees or duration:
        lines.append("**Metadata**:")
        if attendees:
            lines.append(f"- Attendees: {', '.join(attendees)}")
        if duration:
            lines.append(f"- Duration: {duration}")
        lines.append("")

    # Executive Summary
    if es.get("text"):
        lines.append("## Executive Summary")
        lines.append(es["text"].strip())
        lines.append("")

    # Detailed notes (already markdown)
    if body.get("detailed_notes_md"):
        lines.append("## Detailed Meeting Notes")
        lines.append(body["detailed_notes_md"].rstrip())
        lines.append("")

    # Appendices
    if appx:
        lines.append("## Appendices")
        if appx.get("actions_register_md"):
            lines.append(appx["actions_register_md"].rstrip())
            lines.append("")
        if appx.get("decision_log_md"):
            lines.append(appx["decision_log_md"].rstrip())
            lines.append("")
        if appx.get("open_questions_md"):
            lines.append(appx["open_questions_md"].rstrip())
            lines.append("")
        if appx.get("resources_md"):
            lines.append(appx["resources_md"].rstrip())
            lines.append("")

    # Footer
    if footer:
        lines.append("## Footer")
        if footer.get("disclaimer"):
            lines.append(footer["disclaimer"])
        if footer.get("generation_date"):
            lines.append(f"Generated on: {footer['generation_date']}")
        if footer.get("contact"):
            lines.append(f"Contact: {footer['contact']}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "require_approval": not args.approval,
        "require_final_approval": not args.final_approval,
        "temperature": args.temperature,
    }
    if args.provider:
        cfg["provider"] = args.provider
    if args.model:
        cfg["model"] = args.model
    return {"configurable": cfg}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile meeting transcript into a markdown summary (LangGraph)")
    parser.add_argument("--transcript", required=True, help="Path to raw transcript text file")
    parser.add_argument("--metadata", help="Path to metadata JSON (optional)")
    parser.add_argument("--output", default="output.md", help="Output markdown file path")
    parser.add_argument("--provider", choices=["google", "anthropic", "openai"], help="LLM provider override")
    parser.add_argument("--model", help="LLM model override (e.g., gemini-2.5-pro)")
    parser.add_argument("--temperature", type=float, default=0.3, help="LLM temperature")
    parser.add_argument("--approval", action="store_true", help="Enable approval interrupts during run")
    parser.add_argument("--final-approval", action="store_true", help="Enable final approval interrupt")

    args = parser.parse_args()

    # Read transcript
    try:
        with open(args.transcript, "r", encoding="utf-8") as f:
            transcript = f.read()
    except Exception as e:
        print(f"Failed to read transcript: {e}", file=sys.stderr)
        return 1

    # Read metadata JSON (optional)
    metadata: Dict[str, Any] | None = None
    if args.metadata:
        try:
            with open(args.metadata, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Failed to read metadata JSON: {e}", file=sys.stderr)
            return 1

    state: Dict[str, Any] = {
        "messages": [],
        "transcript": transcript,
    }
    if metadata:
        state["metadata"] = metadata

    config = build_config(args)

    try:
        result = app.invoke(state, config=config)
    except Exception as e:
        print(f"Graph execution failed: {e}", file=sys.stderr)
        return 2

    final_doc = result.get("final_doc")
    if not final_doc:
        print("No final document produced (did an approval interrupt pause the run?).", file=sys.stderr)
        return 3

    md = final_doc_to_markdown(final_doc, metadata)

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md)
    except Exception as e:
        print(f"Failed to write output: {e}", file=sys.stderr)
        return 4

    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

