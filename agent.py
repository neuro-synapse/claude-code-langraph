"""
Meeting Transcript Processing v2 — Step 1 (Theme Extraction)

Deployment-first LangGraph scaffold:
- Exports compiled graph as `app`
- No checkpointer
- Minimal state + Step 1 nodes (Preprocess -> Theme Extract -> Interrupt)

Notes:
- Uses Anthropic by default, falling back to OpenAI then Google.
- Theme extraction uses structured output (Pydantic models) per guidelines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from typing import TypedDict

from pydantic import BaseModel, Field
from datetime import datetime

from langgraph.graph import MessagesState, StateGraph, START, END
try:
    from langgraph.func import interrupt
except Exception:
    # Fallback no-op for environments lacking interrupt (for local smoke tests)
    def interrupt(*args, **kwargs):  # type: ignore
        return None

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig


# =====================
# Models (Structured IO)
# =====================

class ThemeFlags(BaseModel):
    decision: bool = False
    action: bool = False
    question: bool = False
    dependency: bool = False


class Theme(BaseModel):
    id: int = Field(..., ge=1, description="Ordinal number for the theme")
    title: str = Field(..., min_length=3, max_length=64)
    summary: str = Field(..., min_length=8)
    time_percent: float = Field(..., ge=0, le=100)
    participants: List[str] = Field(default_factory=list)
    tier: Literal["PRIMARY", "SECONDARY", "TANGENTIAL"]
    flags: ThemeFlags = Field(default_factory=ThemeFlags)


class ThemeList(BaseModel):
    themes: List[Theme]


# ==============
# App State Type
# ==============

class MeetingState(MessagesState):
    """Graph state.

    Fields beyond `messages` are optional and filled as the graph progresses.
    """

    # Required for this workflow (input)
    transcript: str

    # Optional metadata supplied by caller
    metadata: Dict[str, Any]

    # Output of Step 1
    themes: List[Theme]

    # Output of Step 2
    theme_notes: Dict[int, Dict[str, Any]]

    # Output of Step 3
    actions: List[Dict[str, Any]]
    decisions: List[Dict[str, Any]]
    open_questions: List[Dict[str, Any]]

    # Aggregated resources (optional, collated during QC)
    resources: List[Dict[str, Any]]

    # QC findings and corrections summary
    qc_findings: List[Dict[str, Any]]

    # Output of Step 5
    final_doc: Dict[str, Any]


# ======================
# Model selection helper
# ======================

def _get_default_llm(temperature: float = 0.2):
    """Choose Anthropic > OpenAI > Google, per preferences.

    Assumes API keys are available in environment when actually invoked.
    """
    try:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=temperature)
    except Exception:
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model="gpt-4o", temperature=temperature)
        except Exception:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=temperature)


def _get_llm(
    temperature: float = 0.2,
    model: Optional[str] = None,
    provider: Optional[str] = None,
):
    """Resolve LLM based on explicit model/provider, else fallback to default order.

    - If `provider` is set, choose its default model when `model` is None.
    - If only `model` is set, infer provider by name prefix.
    - Else use Anthropic > OpenAI > Google fallback.
    """
    # Helper imports inside function to avoid mandatory deps when unused
    def _google(m: str):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=m, temperature=temperature)

    def _anthropic(m: str):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=m, temperature=temperature)

    def _openai(m: str):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=m, temperature=temperature)

    if provider:
        p = provider.lower()
        if p == "google":
            return _google(model or "gemini-2.5-pro")
        if p == "anthropic":
            return _anthropic(model or "claude-3-5-sonnet-20241022")
        if p == "openai":
            return _openai(model or "gpt-4o")

    if model:
        low = model.lower()
        if "gemini" in low:
            return _google(model)
        if "claude" in low:
            return _anthropic(model)
        if low.startswith("gpt") or "-o" in low:
            return _openai(model)

    # Fallback preference chain
    return _get_default_llm(temperature=temperature)


# =====================
# Node: Preprocess input
# =====================

def preprocess(state: MeetingState) -> Dict[str, Any]:
    """Light validation/normalization placeholder.

    For now, just validate presence of `transcript`. Future enhancements can
    segment speakers, collect URLs, etc., but keep Step 1 minimal.
    """
    transcript = state.get("transcript", "") or ""
    if not transcript.strip():
        raise ValueError("Missing 'transcript' in state.")
    return {}


# =======================
# Node: Extract the themes
# =======================

MASTER_THEME_PROMPT = (
    "You are an expert meeting analyst specializing in identifying key discussion "
    "threads and decision points. Analyze the provided meeting transcript with the "
    "following objectives:\n\n"
    "1. Identify ALL major themes, topics, and discussion areas covered in the meeting\n"
    "2. For each theme, provide:\n"
    "   - A clear, descriptive title (3-7 words)\n"
    "   - A one-sentence summary of what was discussed\n"
    "   - Approximate percentage of meeting time devoted to this theme\n"
    "   - Key participants who contributed to this theme\n\n"
    "3. Organize themes by:\n"
    "   - Primary themes (core meeting objectives)\n"
    "   - Secondary themes (supporting discussions)\n"
    "   - Tangential themes (brief mentions or off-topic discussions)\n\n"
    "4. Flag any themes that contain:\n"
    "   - Decisions made\n"
    "   - Action items assigned\n"
    "   - Unresolved questions\n"
    "   - Dependencies or blockers\n\n"
    "Present your analysis using the structured JSON schema you have been provided. "
    "Ensure there are 5-10 themes and the time percentages are approximate but realistic."
)


def extract_themes(state: MeetingState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """LLM call producing structured ThemeList output.

    Respects model preference order and uses low temperature for consistent extraction.
    """
    transcript = state["transcript"]

    # Allow temperature override via config if desired
    temperature = 0.2
    require_approval = True
    if config and isinstance(config, dict):
        cfg = config.get("configurable", {}) or {}
        temperature = float(cfg.get("temperature", temperature))
        require_approval = bool(cfg.get("require_approval", require_approval))
        model = cfg.get("model")
        provider = cfg.get("provider")

    llm = _get_llm(temperature=temperature, model=model, provider=provider)

    messages = [
        SystemMessage(content=MASTER_THEME_PROMPT),
        HumanMessage(content=f"Transcript:\n{transcript}"),
    ]

    # Ask the model to return the Pydantic-validated structure
    structured = llm.with_structured_output(ThemeList)
    result = structured.invoke(messages)

    # Normalize to dict
    themes: List[Theme]
    if isinstance(result, ThemeList):
        themes = result.themes
    else:
        # If the LC wrapper returns a raw dict
        themes = [Theme(**t) for t in result.get("themes", [])]

    # Lightweight validation: ensure 5-10 items
    if len(themes) < 5 or len(themes) > 10:
        # Minimal retry with stronger instruction; production could add more controls
        strict_prompt = MASTER_THEME_PROMPT + "\n\nSTRICT: Return between 5 and 10 themes inclusive."
        messages[0] = SystemMessage(content=strict_prompt)
        result2 = structured.invoke(messages)
        if isinstance(result2, ThemeList):
            themes = result2.themes
        else:
            themes = [Theme(**t) for t in result2.get("themes", [])]

    # Pause for human approval/corrections before proceeding to downstream steps
    if require_approval:
        interrupt(
            "Please review extracted themes (titles, tiers, participants, time%).\n"
            "Provide any corrections before continuing."
        )

    return {"themes": themes}


# ==================
# Build and export app
# ==================

graph_builder = StateGraph(MeetingState)
graph_builder.add_node("preprocess", preprocess)
graph_builder.add_node("theme_extract", extract_themes)
 
# =========================
# Step 2: Theme Expansion
# =========================

class Quote(BaseModel):
    text: str
    speaker: Optional[str] = None
    context: Optional[str] = None


class ResourceRef(BaseModel):
    label: Optional[str] = None
    url: Optional[str] = None
    reference: Optional[str] = None


class ThemeNotes(BaseModel):
    theme_id: int
    title: str
    notes_md: str = Field(
        ..., description="300-500 words markdown with required structure and hierarchy"
    )
    word_count: int
    quotes: List[Quote] = Field(default_factory=list)
    related_resources: List[ResourceRef] = Field(default_factory=list)
    key_insights_or_decisions: Optional[str] = None
    open_questions: Optional[str] = None


THEME_EXPANSION_PROMPT = (
    "You are creating comprehensive documentation for a single meeting theme. "
    "Follow these requirements strictly and output in the provided structured schema.\n\n"
    "Structure your notes_md as:\n"
    "- Overview paragraph (2-3 sentences setting context)\n"
    "- Main Discussion Points (hierarchical bullets)\n"
    "  • Primary point\n"
    "    ◦ Supporting detail\n"
    "      ▪ Specific example or \"quote\" - Speaker Name\n"
    "- Key Insights or Decisions (if any)\n"
    "- Questions Raised but Not Answered (if any)\n"
    "- Related Resources Mentioned (documents, tools, links)\n\n"
    "Requirements:\n"
    "1) COMPLETENESS: Include every relevant point from the transcript for this theme\n"
    "2) QUOTES: Preserve exact quotes with attribution using \"quote\" - Speaker Name\n"
    "3) CONTEXT: Include context around decisions and discussions\n"
    "4) ACRONYMS: Expand on first use as 'Expanded Form (ACRONYM)'\n"
    "5) TECHNICAL DETAILS: Preserve numbers, metrics, dates, specs exactly as stated\n"
    "6) LENGTH: Keep notes_md between 300 and 500 words\n"
)


def _count_words(text: str) -> int:
    return len([w for w in text.strip().split() if w])


def expand_themes(state: MeetingState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Generate detailed notes for each theme (300-500 words, with quotes and resources)."""
    transcript = state["transcript"]
    themes = state.get("themes", []) or []
    if not themes:
        return {"theme_notes": {}}

    # Allow overrides
    temperature = 0.4
    if config and isinstance(config, dict):
        cfg = config.get("configurable", {}) or {}
        temperature = float(cfg.get("temperature", temperature))
        model = cfg.get("model")
        provider = cfg.get("provider")

    llm = _get_llm(temperature=temperature, model=model, provider=provider)
    structured = llm.with_structured_output(ThemeNotes)

    notes_by_id: Dict[int, Dict[str, Any]] = {}

    for t in themes:
        # Handle Theme instance or dict
        theme_obj = t if isinstance(t, Theme) else Theme(**t)

        sys = SystemMessage(content=THEME_EXPANSION_PROMPT)
        human = HumanMessage(
            content=(
                f"Focus on Theme {theme_obj.id}: {theme_obj.title}\n\n"
                f"Theme summary: {theme_obj.summary}\n"
                f"Tier: {theme_obj.tier}; Time: {theme_obj.time_percent}%\n"
                f"Participants: {', '.join(theme_obj.participants)}\n\n"
                f"Original transcript (full reference):\n{transcript}"
            )
        )

        result = structured.invoke([sys, human])

        # Normalize to ThemeNotes instance
        if not isinstance(result, ThemeNotes):
            result = ThemeNotes(**result)

        wc = result.word_count or _count_words(result.notes_md)
        # Retry if out of range
        if wc < 300 or wc > 500:
            sys2 = SystemMessage(
                content=THEME_EXPANSION_PROMPT
                + "\n\nSTRICT: Ensure notes_md stays within 300-500 words."
            )
            result2 = structured.invoke([sys2, human])
            if not isinstance(result2, ThemeNotes):
                result2 = ThemeNotes(**result2)
            result = result2

        # Finalize and store as plain dict for state friendliness
        data = result.model_dump()
        # Ensure theme_id is correct
        data["theme_id"] = theme_obj.id
        # Backfill word count if missing
        if not data.get("word_count"):
            data["word_count"] = _count_words(data.get("notes_md", ""))

        notes_by_id[theme_obj.id] = data

    return {"theme_notes": notes_by_id}


graph_builder.add_node("expand_themes", expand_themes)

graph_builder.add_edge(START, "preprocess")
graph_builder.add_edge("preprocess", "theme_extract")
graph_builder.add_edge("theme_extract", "expand_themes")

# =========================
# Step 3: Actions/Decisions
# =========================

class ActionItem(BaseModel):
    what: str
    owner: Optional[str] = None
    due: Optional[str] = None  # ISO date or timeframe (Immediate/Short-term/Long-term)
    why: Optional[str] = None
    # Be flexible in QC: accept arbitrary strings and normalize later
    priority: Optional[str] = None
    depends_on: List[str] = Field(default_factory=list)


class Decision(BaseModel):
    statement: str
    rationale: Optional[str] = None
    dissent: Optional[str] = None
    impact: Optional[str] = None


class OpenQuestion(BaseModel):
    question: str
    context: Optional[str] = None
    suggested_owner: Optional[str] = None


class Outcomes(BaseModel):
    actions: List[ActionItem]
    decisions: List[Decision]
    open_questions: List[OpenQuestion]


ACTION_EXTRACTION_PROMPT = (
    "You are a project management specialist extracting actionable outcomes from a meeting.\n\n"
    "Analyze the transcript and expanded notes to identify:\n"
    "1) EXPLICIT ACTION ITEMS (what, owner, due, why, dependencies)\n"
    "2) IMPLICIT ACTION ITEMS (convert questions and problems into tasks)\n"
    "3) DECISIONS MADE (statement, rationale, dissent, impact)\n"
    "4) OPEN QUESTIONS (question, context, suggested owner)\n\n"
    "Targets: 5-15 action items, 3-8 decisions, 2-5 open questions.\n"
    "Be comprehensive and prefer inclusion over omission. Use the provided structured schema."
)


def _compile_theme_notes_md(theme_notes: Dict[int, Dict[str, Any]]) -> str:
    if not theme_notes:
        return ""
    ordered_ids = sorted(theme_notes.keys())
    parts: List[str] = []
    for tid in ordered_ids:
        item = theme_notes[tid]
        title = item.get("title") or f"Theme {tid}"
        md = item.get("notes_md", "")
        parts.append(f"## {tid}. {title}\n\n{md}\n")
    return "\n".join(parts)


def _canonicalize_owner(name: Optional[str], attendees: Optional[List[str]]) -> Optional[str]:
    if not name or not attendees:
        return name
    # Simple case-insensitive exact match; do not over-engineer here.
    for a in attendees:
        if a.lower() == name.lower():
            return a
    return name


def extract_outcomes(state: MeetingState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    transcript = state.get("transcript", "")
    theme_notes = state.get("theme_notes", {}) or {}
    metadata = state.get("metadata", {}) or {}
    attendees = metadata.get("attendees") if isinstance(metadata, dict) else None

    temperature = 0.2
    require_approval = True
    min_actions, min_decisions, min_questions = 5, 3, 2
    if config and isinstance(config, dict):
        cfg = config.get("configurable", {}) or {}
        temperature = float(cfg.get("temperature", temperature))
        require_approval = bool(cfg.get("require_approval", require_approval))
        min_actions = int(cfg.get("min_actions", min_actions))
        min_decisions = int(cfg.get("min_decisions", min_decisions))
        min_questions = int(cfg.get("min_questions", min_questions))
        model = cfg.get("model")
        provider = cfg.get("provider")

    llm = _get_llm(temperature=temperature, model=model, provider=provider)
    structured = llm.with_structured_output(Outcomes)

    compiled_md = _compile_theme_notes_md(theme_notes)

    sys = SystemMessage(content=ACTION_EXTRACTION_PROMPT)
    human = HumanMessage(
        content=(
            "Use both the original transcript and the expanded theme notes.\n\n"
            "Transcript (verbatim):\n" + transcript + "\n\n"
            "Expanded Theme Notes (markdown):\n" + compiled_md
        )
    )

    result = structured.invoke([sys, human])
    if not isinstance(result, Outcomes):
        result = Outcomes(**result)

    # Validate counts; retry with stricter instruction if below targets
    if (
        len(result.actions) < min_actions
        or len(result.decisions) < min_decisions
        or len(result.open_questions) < min_questions
    ):
        sys2 = SystemMessage(
            content=(
                ACTION_EXTRACTION_PROMPT
                + "\n\nSTRICT: Ensure at least the minimum counts. "
                + f"Actions>={min_actions}, Decisions>={min_decisions}, OpenQuestions>={min_questions}. "
                + "Include implicit items where necessary."
            )
        )
        result2 = structured.invoke([sys2, human])
        if not isinstance(result2, Outcomes):
            result2 = Outcomes(**result2)
        result = result2

    # Normalize owners with attendee list if provided
    actions: List[ActionItem] = []
    for a in result.actions:
        if not isinstance(a, ActionItem):
            a = ActionItem(**a)
        a.owner = _canonicalize_owner(a.owner, attendees)
        actions.append(a)

    # Optional approval interrupt before finalizing
    if require_approval:
        interrupt(
            "Please review ACTIONS/DECISIONS/OPEN QUESTIONS.\n"
            "Confirm owners, due dates, priorities, and dependencies."
        )

    # Return as simple dicts for state
    return {
        "actions": [x.model_dump() for x in actions],
        "decisions": [d.model_dump() if isinstance(d, BaseModel) else d for d in result.decisions],
        "open_questions": [q.model_dump() if isinstance(q, BaseModel) else q for q in result.open_questions],
    }


graph_builder.add_node("actions_decisions", extract_outcomes)
graph_builder.add_edge("expand_themes", "actions_decisions")

# =============================
# Step 4: Quality Control (QC)
# =============================

class QCFinding(BaseModel):
    issue: str
    fix: str
    evidence: Optional[str] = None
    severity: Optional[Literal["minor", "major"]] = None
    scope: Optional[str] = None


class QCOutput(BaseModel):
    themes: List[Theme]
    theme_notes: List[ThemeNotes]
    actions: List[ActionItem]
    decisions: List[Decision]
    open_questions: List[OpenQuestion]
    resources: List[ResourceRef] = Field(default_factory=list)
    # Some models may emit a formatted string; accept both and sanitize later
    metadata: Optional[Dict[str, Any] | str] = None


class QCResult(BaseModel):
    corrections: QCOutput
    findings: List[QCFinding]


SELF_REVIEW_PROMPT = (
    "You are performing a quality control pass on a meeting summary.\n\n"
    "Apply the following checklist and produce corrected artifacts using the structured schema provided:\n"
    "ATTRIBUTION FIXES:\n"
    "- Replace room names with actual speaker names; correct phonetic spellings; verify quote attribution.\n"
    "CONTENT COMPLETENESS:\n"
    "- Add links/URLs mentioned; include document names or project codes; expand acronyms on first use.\n"
    "ACTION ITEM REFINEMENT:\n"
    "- Eliminate duplicates; break vague items into specific tasks; assign provisional owners; add realistic timeframes.\n"
    "READABILITY:\n"
    "- Ensure consistent formatting; fix grammar in paraphrases; maintain logical flow and transitions.\n"
    "METADATA:\n"
    "- Add meeting date, attendees, duration; include AI-generated disclaimer; note technical difficulties or missing segments.\n\n"
    "Rules:\n"
    "- Preserve exact quotes verbatim with correct attribution.\n"
    "- Only polish paraphrased content; do not alter numbers/dates/specs.\n"
    "- If uncertain about a fix, propose and flag in findings with severity 'minor'.\n"
)


def _gather_resources_from_notes(theme_notes: Dict[int, Dict[str, Any]]) -> List[ResourceRef]:
    results: List[ResourceRef] = []
    for v in (theme_notes or {}).values():
        for r in v.get("related_resources", []) or []:
            try:
                rr = ResourceRef(**r) if not isinstance(r, ResourceRef) else r
                results.append(rr)
            except Exception:
                # Skip malformed resource entries
                continue
    return results


def qc_enhance(state: MeetingState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    transcript = state.get("transcript", "")
    themes_in = state.get("themes", []) or []
    theme_notes_in = state.get("theme_notes", {}) or {}
    actions_in = state.get("actions", []) or []
    decisions_in = state.get("decisions", []) or []
    open_q_in = state.get("open_questions", []) or []
    metadata_in = state.get("metadata", {}) or {}

    temperature = 0.3
    require_approval = True
    if config and isinstance(config, dict):
        cfg = config.get("configurable", {}) or {}
        temperature = float(cfg.get("temperature", temperature))
        require_approval = bool(cfg.get("require_approval", require_approval))
        model = cfg.get("model")
        provider = cfg.get("provider")

    llm = _get_llm(temperature=temperature, model=model, provider=provider)
    structured = llm.with_structured_output(QCResult)

    # Prepare a compact, readable input view
    def _fmt_themes(lst: List[Any]) -> str:
        lines = []
        for t in lst:
            T = t if isinstance(t, Theme) else Theme(**t)
            flags = []
            if T.flags.decision:
                flags.append("decision")
            if T.flags.action:
                flags.append("action")
            if T.flags.question:
                flags.append("question")
            if T.flags.dependency:
                flags.append("dependency")
            lines.append(
                f"[{T.tier}] {T.id}. {T.title} — {T.time_percent}% — Participants: {', '.join(T.participants)} — Flags: {', '.join(flags) if flags else 'none'}"
            )
        return "\n".join(lines)

    def _fmt_actions(lst: List[Any]) -> str:
        lines = []
        for a in lst:
            A = a if isinstance(a, ActionItem) else ActionItem(**a)
            lines.append(
                f"[Priority: {A.priority or 'UNSET'}] {A.what} | Owner: {A.owner or 'TBD'} | Due: {A.due or 'TBD'} | Why: {A.why or ''} | Depends on: {', '.join(A.depends_on) if A.depends_on else '—'}"
            )
        return "\n".join(lines)

    def _fmt_decisions(lst: List[Any]) -> str:
        lines = []
        for d in lst:
            D = d if isinstance(d, Decision) else Decision(**d)
            lines.append(
                f"✓ {D.statement} | Rationale: {D.rationale or ''} | Impact: {D.impact or ''} | Dissent: {D.dissent or ''}"
            )
        return "\n".join(lines)

    def _fmt_open_questions(lst: List[Any]) -> str:
        lines = []
        for q in lst:
            Q = q if isinstance(q, OpenQuestion) else OpenQuestion(**q)
            lines.append(
                f"? {Q.question} | Context: {Q.context or ''} | Suggested owner: {Q.suggested_owner or 'TBD'}"
            )
        return "\n".join(lines)

    compiled_notes_md = _compile_theme_notes_md(theme_notes_in)
    compiled_resources = _gather_resources_from_notes(theme_notes_in)

    sys = SystemMessage(content=SELF_REVIEW_PROMPT)
    human = HumanMessage(
        content=(
            "Review the following materials, then return corrected artifacts using the structured schema.\n\n"
            "Metadata (may be incomplete):\n" + str(metadata_in) + "\n\n"
            "Themes:\n" + _fmt_themes(themes_in) + "\n\n"
            "Theme Notes (markdown):\n" + compiled_notes_md + "\n\n"
            "Action Items:\n" + _fmt_actions(actions_in) + "\n\n"
            "Decisions:\n" + _fmt_decisions(decisions_in) + "\n\n"
            "Open Questions:\n" + _fmt_open_questions(open_q_in) + "\n\n"
            "Existing Resources (from notes):\n" + str([r.model_dump() for r in compiled_resources]) + "\n\n"
            "Original Transcript (for verification):\n" + transcript
        )
    )

    result = structured.invoke([sys, human])
    if not isinstance(result, QCResult):
        result = QCResult(**result)

    # Convert theme_notes list back into mapping by theme_id
    notes_map: Dict[int, Dict[str, Any]] = {}
    for tn in result.corrections.theme_notes:
        tnx = tn if isinstance(tn, ThemeNotes) else ThemeNotes(**tn)
        notes_map[tnx.theme_id] = tnx.model_dump()

    themes_corr = [t.model_dump() if isinstance(t, Theme) else Theme(**t).model_dump() for t in result.corrections.themes]
    # Normalize priorities to canonical buckets when possible
    def _norm_priority(p: Optional[str]) -> Optional[str]:
        if not p:
            return p
        s = str(p).strip().upper()
        if s in {"IMMEDIATE", "URGENT", "P0", "P1", "HIGH"}:
            return "IMMEDIATE"
        if s in {"SHORT-TERM", "SHORT_TERM", "P2", "MEDIUM", "SOON"}:
            return "SHORT_TERM"
        if s in {"LONG-TERM", "LONG_TERM", "P3", "LOW", "LATER"}:
            return "LONG_TERM"
        return s

    actions_corr = []
    for a in result.corrections.actions:
        A = a if isinstance(a, ActionItem) else ActionItem(**a)
        d = A.model_dump()
        d["priority"] = _norm_priority(d.get("priority"))
        actions_corr.append(d)
    decisions_corr = [d.model_dump() if isinstance(d, Decision) else Decision(**d).model_dump() for d in result.corrections.decisions]
    open_q_corr = [q.model_dump() if isinstance(q, OpenQuestion) else OpenQuestion(**q).model_dump() for q in result.corrections.open_questions]
    resources_corr = [r.model_dump() if isinstance(r, ResourceRef) else ResourceRef(**r).model_dump() for r in result.corrections.resources]
    findings = [f.model_dump() if isinstance(f, QCFinding) else QCFinding(**f).model_dump() for f in result.findings]

    if require_approval:
        interrupt(
            "QC pass completed. Please review the proposed corrections and findings.\n"
            "Confirm name/attribution changes, added links, and formatting adjustments before finalizing."
        )

    updates: Dict[str, Any] = {
        "themes": themes_corr or themes_in,
        "theme_notes": notes_map or theme_notes_in,
        "actions": actions_corr or actions_in,
        "decisions": decisions_corr or decisions_in,
        "open_questions": open_q_corr or open_q_in,
        "resources": resources_corr or [r.model_dump() for r in compiled_resources],
        "qc_findings": findings,
    }

    # Allow metadata corrections to flow back
    if result.corrections.metadata:
        # Only accept dict-like metadata; ignore formatted strings
        if isinstance(result.corrections.metadata, dict):
            updates["metadata"] = result.corrections.metadata

    return updates


graph_builder.add_node("qc_enhance", qc_enhance)
graph_builder.add_edge("actions_decisions", "qc_enhance")

# ==============================
# Step 5: Final Compilation
# ==============================

class ExecSummary(BaseModel):
    text: str
    word_count: int


class FinalAppendices(BaseModel):
    actions_register_md: str
    decision_log_md: str
    open_questions_md: str
    resources_md: str


class FinalFooter(BaseModel):
    disclaimer: str = "This summary was AI-generated from the meeting transcript."
    generation_date: str
    contact: Optional[str] = None


class FinalBody(BaseModel):
    detailed_notes_md: str
    word_count: int


class FinalDocument(BaseModel):
    exec_summary: ExecSummary
    body: FinalBody
    appendices: FinalAppendices
    footer: FinalFooter
    total_word_count: int


FINAL_COMPILATION_PROMPT = (
    "Create the final meeting documentation package.\n\n"
    "1) EXECUTIVE SUMMARY (<=150 words): Meeting purpose/outcome in 2 sentences; 3-5 key decisions/achievements; critical action items with owners; next steps and timeline.\n"
    "2) DETAILED MEETING NOTES: Order themes logically (PRIMARY -> SECONDARY -> TANGENTIAL), add smooth transitions, preserve content accuracy.\n"
    "3) Maintain consistent formatting and hierarchy.\n"
    "Use the structured schema when responding."
)


def _tier_rank(tier: str) -> int:
    order = {"PRIMARY": 0, "SECONDARY": 1, "TANGENTIAL": 2}
    return order.get(tier, 3)


def _order_themes(themes: List[Any]) -> List[Theme]:
    items: List[Theme] = [t if isinstance(t, Theme) else Theme(**t) for t in themes]
    return sorted(items, key=lambda t: (_tier_rank(t.tier), -t.time_percent, t.id))


def _compile_theme_notes_md_ordered(themes: List[Any], theme_notes: Dict[int, Dict[str, Any]]) -> str:
    ordered = _order_themes(themes)
    parts: List[str] = []
    for t in ordered:
        tn = theme_notes.get(t.id) or {}
        title = tn.get("title") or t.title
        md = tn.get("notes_md", "")
        parts.append(f"### {t.title} ({t.tier})\n\n{md}\n")
    return "\n".join(parts)


def _mk_actions_register_md(actions: List[Dict[str, Any]]) -> str:
    lines = ["## Action Items\n"]
    for a in actions:
        A = a if isinstance(a, ActionItem) else ActionItem(**a)
        pr = A.priority or ""
        dep = ", ".join(A.depends_on) if A.depends_on else "—"
        lines.append(
            f"- [ ] [{pr}] {A.what} | Owner: {A.owner or 'TBD'} | Due: {A.due or 'TBD'} | Why: {A.why or ''} | Depends on: {dep}"
        )
    return "\n".join(lines)


def _mk_decision_log_md(decisions: List[Dict[str, Any]]) -> str:
    lines = ["## Decisions\n"]
    for d in decisions:
        D = d if isinstance(d, Decision) else Decision(**d)
        lines.append(
            f"- ✓ {D.statement} | Rationale: {D.rationale or ''} | Impact: {D.impact or ''} | Dissent: {D.dissent or ''}"
        )
    return "\n".join(lines)


def _mk_open_questions_md(questions: List[Dict[str, Any]]) -> str:
    lines = ["## Open Questions\n"]
    for q in questions:
        Q = q if isinstance(q, OpenQuestion) else OpenQuestion(**q)
        lines.append(
            f"- ? {Q.question} | Context: {Q.context or ''} | Suggested owner: {Q.suggested_owner or 'TBD'}"
        )
    return "\n".join(lines)


def _mk_resources_md(resources: List[Dict[str, Any]]) -> str:
    lines = ["## Resources & Links\n"]
    for r in resources or []:
        label = r.get("label") if isinstance(r, dict) else getattr(r, "label", None)
        url = r.get("url") if isinstance(r, dict) else getattr(r, "url", None)
        ref = r.get("reference") if isinstance(r, dict) else getattr(r, "reference", None)
        display = label or ref or url or "Resource"
        link = f" ({url})" if url else (f" — {ref}" if ref else "")
        lines.append(f"- {display}{link}")
    return "\n".join(lines)


def final_compile(state: MeetingState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    transcript = state.get("transcript", "")
    themes = state.get("themes", []) or []
    theme_notes = state.get("theme_notes", {}) or {}
    actions = state.get("actions", []) or []
    decisions = state.get("decisions", []) or []
    open_q = state.get("open_questions", []) or []
    resources = state.get("resources", []) or []
    metadata = state.get("metadata", {}) or {}

    # Config
    temperature = 0.3
    require_final_approval = False
    if config and isinstance(config, dict):
        cfg = config.get("configurable", {}) or {}
        temperature = float(cfg.get("temperature", temperature))
        require_final_approval = bool(cfg.get("require_final_approval", require_final_approval))
        model = cfg.get("model")
        provider = cfg.get("provider")

    llm = _get_llm(temperature=temperature, model=model, provider=provider)

    # Prepare ordered notes and appendix markdown
    ordered_notes_md = _compile_theme_notes_md_ordered(themes, theme_notes)
    actions_md = _mk_actions_register_md(actions)
    decisions_md = _mk_decision_log_md(decisions)
    questions_md = _mk_open_questions_md(open_q)
    resources_md = _mk_resources_md(resources)

    # 1) Executive summary
    es_struct = llm.with_structured_output(ExecSummary)
    es_sys = SystemMessage(content=FINAL_COMPILATION_PROMPT)
    es_human = HumanMessage(
        content=(
            "Craft an executive summary (<=150 words).\n\n"
            "Metadata: " + str(metadata) + "\n\n"
            "Decisions: \n" + "\n".join([d["statement"] if isinstance(d, dict) else d.statement for d in decisions]) + "\n\n"
            "Top Actions: \n" + "\n".join([a["what"] if isinstance(a, dict) else a.what for a in actions[:5]]) + "\n\n"
            "Transcript signal (context only):\n" + transcript[:4000]
        )
    )
    es = es_struct.invoke([es_sys, es_human])
    if not isinstance(es, ExecSummary):
        es = ExecSummary(**es)
    # Normalize/count words
    es_wc = _count_words(es.text)
    es = ExecSummary(text=es.text, word_count=es_wc)
    if es.word_count > 150:
        # Retry with stricter constraint
        es_sys2 = SystemMessage(content=FINAL_COMPILATION_PROMPT + "\n\nSTRICT: Keep summary <=150 words.")
        es = es_struct.invoke([es_sys2, es_human])
        if not isinstance(es, ExecSummary):
            es = ExecSummary(**es)
        es = ExecSummary(text=es.text, word_count=_count_words(es.text))

    # 2) Detailed body
    body_struct = llm.with_structured_output(FinalBody)
    body_sys = SystemMessage(content=FINAL_COMPILATION_PROMPT)
    body_human = HumanMessage(
        content=(
            "Compose DETAILED MEETING NOTES using the provided theme notes, reordered as PRIMARY→SECONDARY→TANGENTIAL, with smooth transitions and consistent hierarchy.\n\n"
            "Ordered Theme Notes (markdown):\n" + ordered_notes_md + "\n\n"
            "Ensure readability; preserve exact quotes and technical details."
        )
    )
    body = body_struct.invoke([body_sys, body_human])
    if not isinstance(body, FinalBody):
        body = FinalBody(**body)
    # Normalize word count
    body = FinalBody(detailed_notes_md=body.detailed_notes_md, word_count=_count_words(body.detailed_notes_md))

    # 3) Footer
    gen_date = datetime.utcnow().strftime("%Y-%m-%d")
    contact = metadata.get("contact") if isinstance(metadata, dict) else None
    footer = FinalFooter(generation_date=gen_date, contact=contact)

    total_wc = (es.word_count or 0) + (body.word_count or _count_words(body.detailed_notes_md))

    final_doc = FinalDocument(
        exec_summary=es,
        body=body,
        appendices=FinalAppendices(
            actions_register_md=actions_md,
            decision_log_md=decisions_md,
            open_questions_md=questions_md,
            resources_md=resources_md,
        ),
        footer=footer,
        total_word_count=total_wc,
    )

    if require_final_approval:
        interrupt(
            "Final document compiled. Review executive summary, detailed notes, and appendices before distribution."
        )

    return {"final_doc": final_doc.model_dump()}


graph_builder.add_node("final_compile", final_compile)
graph_builder.add_edge("qc_enhance", "final_compile")
graph_builder.add_edge("final_compile", END)

app = graph_builder.compile()
