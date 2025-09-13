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
from langgraph.checkpoint.memory import MemorySaver
try:
    from langgraph.func import interrupt
except Exception:
    # Fallback no-op for environments lacking interrupt (for local smoke tests)
    def interrupt(*args, **kwargs):  # type: ignore
        return None

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path


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

    # Memory tracking fields
    meeting_id: str  # Unique identifier for this meeting
    meeting_date: str  # ISO date of the meeting

    # Output of Step 1
    themes: List[Theme]

    # Output of Step 2
    theme_notes: Dict[int, Dict[str, Any]]

    # Output of Step 3
    actions: List[Dict[str, Any]]
    decisions: List[Dict[str, Any]]
    open_questions: List[Dict[str, Any]]

    # Memory integration
    previous_action_items: List[Dict[str, Any]]  # Action items from previous meetings
    escalated_items: List[Dict[str, Any]]  # Items that need attention
    memory_insights: List[str]  # AI-generated insights about recurring items

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

    # Best-effort meeting_id/date derivation if absent
    metadata = state.get("metadata", {}) or {}

    def _slug(s: str) -> str:
        keep = [c.lower() if c.isalnum() else "-" for c in str(s)]
        slug = "".join(keep)
        while "--" in slug:
            slug = slug.replace("--", "-")
        return slug.strip("-") or "meeting"

    updates: Dict[str, Any] = {}
    mdate = state.get("meeting_date") or metadata.get("date")
    if not mdate:
        mdate = datetime.utcnow().strftime("%Y-%m-%d")
        updates["meeting_date"] = mdate
    else:
        updates["meeting_date"] = mdate

    mid = state.get("meeting_id")
    if not mid:
        title = metadata.get("title") or "meeting"
        updates["meeting_id"] = f"{mdate}-{_slug(title)[:40]}"

    return updates


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
    themes: List[Theme] = []
    parsed_ok = False
    try:
        result = structured.invoke(messages)
        if isinstance(result, ThemeList):
            themes = result.themes
        else:
            themes = [Theme(**t) for t in result.get("themes", [])]
        parsed_ok = len(themes) > 0
    except Exception:
        parsed_ok = False

    # Fallback: force JSON-only response and parse manually if needed
    if not parsed_ok:
        schema_hint = (
            "Return ONLY JSON matching this exact schema, no extra text.\n"
            "{\n  \"themes\": [\n    {\n      \"id\": 1,\n      \"title\": \"Short title\",\n      \"summary\": \"One sentence summary.\",\n      \"time_percent\": 20,\n      \"participants\": [\"Alice\", \"Bob\"],\n      \"tier\": \"PRIMARY\",\n      \"flags\": {\n        \"decision\": false,\n        \"action\": true,\n        \"question\": false,\n        \"dependency\": false\n      }\n    }\n  ]\n}"
        )
        messages2 = [
            SystemMessage(content=MASTER_THEME_PROMPT + "\n\nSTRICT: " + schema_hint),
            HumanMessage(content=f"Transcript:\n{transcript}"),
        ]
        raw = llm.invoke(messages2)
        raw_text = getattr(raw, "content", "") if raw is not None else ""
        # Strip code fences if present
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            # remove first fence line and optional language
            raw_text = "\n".join(raw_text.splitlines()[1:])
            if raw_text.endswith("```"):
                raw_text = "\n".join(raw_text.splitlines()[:-1])
        try:
            data = json.loads(raw_text)
            items = data.get("themes", []) if isinstance(data, dict) else []
        except Exception:
            items = []

        # Lenient normalization to Theme objects
        def _to_bool_flags(val) -> ThemeFlags:
            if isinstance(val, dict):
                return ThemeFlags(**{**ThemeFlags().model_dump(), **val})
            if isinstance(val, list):
                s = set([str(x).strip().lower() for x in val])
                return ThemeFlags(
                    decision=("decision" in s),
                    action=("action" in s or "actions" in s or "action items assigned" in s),
                    question=("question" in s or "questions" in s),
                    dependency=("dependency" in s or "dependencies" in s or "blocker" in s or "blockers" in s),
                )
            return ThemeFlags()

        def _norm_tier(t: Optional[str]) -> str:
            if not t:
                return "SECONDARY"
            s = str(t).strip().lower()
            if "primary" in s:
                return "PRIMARY"
            if "secondary" in s:
                return "SECONDARY"
            if "tangential" in s or "tangent" in s:
                return "TANGENTIAL"
            return "SECONDARY"

        normed: List[Theme] = []
        for idx, it in enumerate(items, start=1):
            try:
                pid = int(it.get("id") or idx)
            except Exception:
                pid = idx
            title = it.get("title") or it.get("name") or f"Theme {pid}"
            summary = it.get("summary") or it.get("description") or ""
            tp = it.get("time_percent")
            try:
                tpv = float(tp) if tp is not None else None
            except Exception:
                tpv = None
            participants = it.get("participants") or []
            if isinstance(participants, str):
                participants = [p.strip() for p in participants.split(",") if p.strip()]
            tier = _norm_tier(it.get("tier") or it.get("category"))
            flags = _to_bool_flags(it.get("flags") or it.get("indicators") or [])

            theme = Theme(
                id=pid,
                title=str(title)[:64],
                summary=str(summary)[:500] if summary else f"Discussion on {title}",
                time_percent=float(tpv) if tpv is not None else 0.0,
                participants=[str(x) for x in participants][:10],
                tier=tier,  # type: ignore
                flags=flags,
            )
            normed.append(theme)

        # If no time percents, distribute roughly
        if normed and all(t.time_percent == 0 for t in normed):
            even = round(100.0 / max(1, len(normed)), 2)
            for t in normed:
                t.time_percent = even

        themes = normed

    # Lightweight validation: ensure 5-10 items
    if len(themes) < 5 or len(themes) > 10:
        # Minimal retry with stronger instruction; production could add more controls
        strict_prompt = MASTER_THEME_PROMPT + "\n\nSTRICT: Return between 5 and 10 themes inclusive."
        messages[0] = SystemMessage(content=strict_prompt)
        try:
            result2 = structured.invoke(messages)
            if isinstance(result2, ThemeList):
                themes = result2.themes
            else:
                themes = [Theme(**t) for t in result2.get("themes", [])]
        except Exception:
            # Keep previous fallback themes if structured parse fails
            themes = themes

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

        # Try structured first; fall back to manual JSON parsing with normalization
        try:
            result = structured.invoke([sys, human])
            if not isinstance(result, ThemeNotes):
                result = ThemeNotes(**result)
        except Exception:
            # Fallback JSON instruction
            schema_hint = (
                "Return ONLY JSON matching this schema, no extra text.\n"
                "{\n  \"theme_id\": 1,\n  \"title\": \"...\",\n  \"notes_md\": \"...\",\n  \"word_count\": 350,\n  \"quotes\": [{\"text\": \"...\", \"speaker\": \"Alice\"}],\n  \"related_resources\": [{\"label\": \"Doc\", \"url\": \"https://...\"}]\n}"
            )
            sys_f = SystemMessage(content=THEME_EXPANSION_PROMPT + "\n\nSTRICT: " + schema_hint)
            raw = llm.invoke([sys_f, human])
            content = getattr(raw, "content", "") if raw is not None else "{}"
            if content.strip().startswith("```"):
                content = "\n".join(content.strip().splitlines()[1:-1])
            try:
                data_f = json.loads(content)
            except Exception:
                data_f = {}

            # Normalize quotes/resources if they are strings
            q = data_f.get("quotes")
            if isinstance(q, list):
                qn = []
                for item in q:
                    if isinstance(item, dict):
                        qn.append(item)
                    elif isinstance(item, str):
                        txt = item.strip().strip('"')
                        speaker = None
                        if " - " in txt:
                            parts = txt.rsplit(" - ", 1)
                            txt = parts[0].strip()
                            speaker = parts[1].strip()
                        qn.append({"text": txt, "speaker": speaker})
                data_f["quotes"] = qn

            rr = data_f.get("related_resources")
            if isinstance(rr, list):
                rrn = []
                for r in rr:
                    if isinstance(r, dict):
                        rrn.append(r)
                    elif isinstance(r, str):
                        rrn.append({"label": r})
                data_f["related_resources"] = rrn

            # Build ThemeNotes instance
            try:
                result = ThemeNotes(**data_f)
            except Exception:
                # As last resort, create minimal valid object
                notes_md = data_f.get("notes_md") or ""
                result = ThemeNotes(
                    theme_id=theme_obj.id,
                    title=theme_obj.title,
                    notes_md=str(notes_md)[:4000] or f"Notes for {theme_obj.title}",
                    word_count=_count_words(notes_md) if notes_md else 320,
                    quotes=[],
                    related_resources=[],
                )

        wc = result.word_count or _count_words(result.notes_md)
        # Retry structured once if out of range
        if wc < 300 or wc > 500:
            try:
                sys2 = SystemMessage(
                    content=THEME_EXPANSION_PROMPT
                    + "\n\nSTRICT: Ensure notes_md stays within 300-500 words."
                )
                result2 = structured.invoke([sys2, human])
                if not isinstance(result2, ThemeNotes):
                    result2 = ThemeNotes(**result2)
                result = result2
            except Exception:
                # If still failing, keep existing result and adjust count
                pass

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
    # Memory tracking fields
    action_id: Optional[str] = None  # Unique identifier for cross-meeting tracking
    meeting_id: Optional[str] = None  # Meeting where this action was first created
    created_date: Optional[str] = None  # ISO date when first created
    status: Optional[str] = "pending"  # pending, in_progress, completed, blocked
    mentions_count: int = 1  # How many meetings this has been mentioned in
    last_mentioned: Optional[str] = None  # ISO date of last mention
    escalation_level: int = 0  # 0=none, 1=escalated, 2=urgent


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


# =====================
# Action Item Memory Management
# =====================

class ActionItemMemory:
    """Persistent storage for action items across meetings."""
    
    def __init__(self, db_path: str = "action_items.db"):
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for action item storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS action_items (
                action_id TEXT PRIMARY KEY,
                what TEXT NOT NULL,
                owner TEXT,
                due TEXT,
                why TEXT,
                priority TEXT,
                depends_on TEXT,  -- JSON array
                meeting_id TEXT,
                created_date TEXT,
                status TEXT DEFAULT 'pending',
                mentions_count INTEGER DEFAULT 1,
                last_mentioned TEXT,
                escalation_level INTEGER DEFAULT 0,
                history TEXT  -- JSON array of meeting mentions
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS meetings (
                meeting_id TEXT PRIMARY KEY,
                date TEXT,
                attendees TEXT,  -- JSON array
                transcript_hash TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_action_item(self, action_item: ActionItem, meeting_id: str, meeting_date: str) -> str:
        """Store or update an action item with memory tracking."""
        if not action_item.action_id:
            action_item.action_id = f"{meeting_id}_{hash(action_item.what) % 10000}"
        
        # Check if this action item already exists
        existing = self.get_action_item_by_id(action_item.action_id)
        
        if existing:
            # Update existing item with new mention
            existing.mentions_count += 1
            existing.last_mentioned = meeting_date
            existing.escalation_level = self._calculate_escalation(existing)
            
            # Update history
            history = json.loads(existing.model_dump().get('history', '[]'))
            history.append({
                'meeting_id': meeting_id,
                'date': meeting_date,
                'status': action_item.status or existing.status
            })
            existing.history = json.dumps(history)
            
            self._update_db_item(existing)
            return existing.action_id
        else:
            # Create new item
            action_item.meeting_id = meeting_id
            action_item.created_date = meeting_date
            action_item.last_mentioned = meeting_date
            action_item.history = json.dumps([{
                'meeting_id': meeting_id,
                'date': meeting_date,
                'status': action_item.status or 'pending'
            }])
            
            self._insert_db_item(action_item)
            return action_item.action_id
    
    def get_action_item_by_id(self, action_id: str) -> Optional[ActionItem]:
        """Retrieve an action item by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM action_items WHERE action_id = ?', (action_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_action_item(row)
        return None
    
    def get_pending_action_items(self, owner: Optional[str] = None) -> List[ActionItem]:
        """Get all pending action items, optionally filtered by owner."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if owner:
            cursor.execute('''
                SELECT * FROM action_items 
                WHERE status IN ('pending', 'in_progress') AND owner = ?
                ORDER BY escalation_level DESC, mentions_count DESC
            ''', (owner,))
        else:
            cursor.execute('''
                SELECT * FROM action_items 
                WHERE status IN ('pending', 'in_progress')
                ORDER BY escalation_level DESC, mentions_count DESC
            ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_action_item(row) for row in rows]
    
    def get_escalated_items(self) -> List[ActionItem]:
        """Get action items that need escalation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM action_items 
            WHERE escalation_level > 0 AND status IN ('pending', 'in_progress')
            ORDER BY escalation_level DESC, mentions_count DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_action_item(row) for row in rows]
    
    def _calculate_escalation(self, action_item: ActionItem) -> int:
        """Calculate escalation level based on mentions and age."""
        escalation = 0
        
        # Escalate based on number of mentions
        if action_item.mentions_count >= 3:
            escalation = 2  # Urgent
        elif action_item.mentions_count >= 2:
            escalation = 1  # Escalated
        
        # Escalate based on overdue status
        if action_item.due and action_item.due != "TBD":
            try:
                # Simple date parsing for common formats
                due_date = self._parse_date(action_item.due)
                if due_date and due_date < datetime.now().date():
                    escalation = max(escalation, 2)  # Urgent if overdue
            except:
                pass  # Ignore date parsing errors
        
        return escalation
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats."""
        if not date_str or date_str == "TBD":
            return None
        
        # Try common formats
        formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        return None
    
    def _row_to_action_item(self, row) -> ActionItem:
        """Convert database row to ActionItem."""
        depends_on = json.loads(row[6]) if row[6] else []
        history = row[13] if row[13] else '[]'
        
        return ActionItem(
            what=row[1],
            owner=row[2],
            due=row[3],
            why=row[4],
            priority=row[5],
            depends_on=depends_on,
            action_id=row[0],
            meeting_id=row[7],
            created_date=row[8],
            status=row[9],
            mentions_count=row[10],
            last_mentioned=row[11],
            escalation_level=row[12]
        )
    
    def _insert_db_item(self, action_item: ActionItem):
        """Insert new action item into database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO action_items 
            (action_id, what, owner, due, why, priority, depends_on, meeting_id, 
             created_date, status, mentions_count, last_mentioned, escalation_level, history)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            action_item.action_id,
            action_item.what,
            action_item.owner,
            action_item.due,
            action_item.why,
            action_item.priority,
            json.dumps(action_item.depends_on),
            action_item.meeting_id,
            action_item.created_date,
            action_item.status,
            action_item.mentions_count,
            action_item.last_mentioned,
            action_item.escalation_level,
            action_item.history
        ))
        
        conn.commit()
        conn.close()
    
    def _update_db_item(self, action_item: ActionItem):
        """Update existing action item in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE action_items 
            SET what=?, owner=?, due=?, why=?, priority=?, depends_on=?, 
                status=?, mentions_count=?, last_mentioned=?, escalation_level=?, history=?
            WHERE action_id=?
        ''', (
            action_item.what,
            action_item.owner,
            action_item.due,
            action_item.why,
            action_item.priority,
            json.dumps(action_item.depends_on),
            action_item.status,
            action_item.mentions_count,
            action_item.last_mentioned,
            action_item.escalation_level,
            action_item.history,
            action_item.action_id
        ))
        
        conn.commit()
        conn.close()


# Global memory instance
action_memory = ActionItemMemory()


ACTION_EXTRACTION_PROMPT = (
    "You are a project management specialist extracting actionable outcomes from a meeting.\n\n"
    "Analyze the transcript and expanded notes to identify:\n"
    "1) EXPLICIT ACTION ITEMS (what, owner, due, why, dependencies)\n"
    "2) IMPLICIT ACTION ITEMS (convert questions and problems into tasks)\n"
    "3) DECISIONS MADE (statement, rationale, dissent, impact)\n"
    "4) OPEN QUESTIONS (question, context, suggested owner)\n\n"
    "MEMORY CONTEXT: You have access to previous action items and escalated items from past meetings.\n"
    "When extracting new action items, consider:\n"
    "- Are any of these items similar to previously discussed items?\n"
    "- Should you flag recurring items for escalation?\n"
    "- Are there dependencies on incomplete previous work?\n\n"
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


def load_memory_context(state: MeetingState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Load previous action items and generate memory insights."""
    meeting_id = state.get("meeting_id", "default_meeting")
    meeting_date = state.get("meeting_date", datetime.now().strftime("%Y-%m-%d"))
    metadata = state.get("metadata", {}) or {}
    attendees = metadata.get("attendees") if isinstance(metadata, dict) else []
    
    # Get previous action items for all attendees
    previous_items = []
    escalated_items = []
    
    if attendees:
        for attendee in attendees:
            attendee_items = action_memory.get_pending_action_items(attendee)
            previous_items.extend([item.model_dump() for item in attendee_items])
    
    # Also get all escalated items
    escalated = action_memory.get_escalated_items()
    escalated_items = [item.model_dump() for item in escalated]
    
    # Generate memory insights using LLM
    memory_insights = []
    if previous_items or escalated_items:
        temperature = 0.3
        if config and isinstance(config, dict):
            cfg = config.get("configurable", {}) or {}
            temperature = float(cfg.get("temperature", temperature))
            model = cfg.get("model")
            provider = cfg.get("provider")
        
        llm = _get_llm(temperature=temperature, model=model, provider=provider)
        
        insight_prompt = (
            "You are analyzing action items from previous meetings to provide insights for the current meeting.\n\n"
            "Generate 2-4 concise insights about:\n"
            "1. Recurring action items that have been mentioned multiple times\n"
            "2. Items that may be overdue or need escalation\n"
            "3. Dependencies or blockers that could affect current work\n"
            "4. Patterns in incomplete work that suggest systemic issues\n\n"
            "Format each insight as a single sentence starting with '•'"
        )
        
        items_text = ""
        if previous_items:
            items_text += "Previous Action Items:\n"
            for item in previous_items[:10]:  # Limit to avoid token limits
                items_text += f"- {item['what']} (Owner: {item['owner']}, Mentions: {item['mentions_count']}, Status: {item['status']})\n"
        
        if escalated_items:
            items_text += "\nEscalated Items:\n"
            for item in escalated_items:
                items_text += f"- {item['what']} (Owner: {item['owner']}, Escalation: {item['escalation_level']})\n"
        
        if items_text:
            messages = [
                SystemMessage(content=insight_prompt),
                HumanMessage(content=items_text)
            ]
            
            response = llm.invoke(messages)
            if response and hasattr(response, 'content'):
                insights = [line.strip() for line in response.content.split('\n') if line.strip().startswith('•')]
                memory_insights = insights[:4]  # Limit to 4 insights
    
    return {
        "previous_action_items": previous_items,
        "escalated_items": escalated_items,
        "memory_insights": memory_insights
    }


def extract_outcomes(state: MeetingState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    transcript = state.get("transcript", "")
    theme_notes = state.get("theme_notes", {}) or {}
    metadata = state.get("metadata", {}) or {}
    attendees = metadata.get("attendees") if isinstance(metadata, dict) else None
    meeting_id = state.get("meeting_id", "default_meeting")
    meeting_date = state.get("meeting_date", datetime.now().strftime("%Y-%m-%d"))
    
    # Get memory context
    previous_items = state.get("previous_action_items", [])
    escalated_items = state.get("escalated_items", [])
    memory_insights = state.get("memory_insights", [])

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
    
    # Build memory context for the prompt
    memory_context = ""
    if memory_insights:
        memory_context += "MEMORY INSIGHTS:\n" + "\n".join(memory_insights) + "\n\n"
    
    if previous_items:
        memory_context += "PREVIOUS ACTION ITEMS (for context):\n"
        for item in previous_items[:5]:  # Limit to avoid token limits
            memory_context += f"- {item['what']} (Owner: {item['owner']}, Mentions: {item['mentions_count']})\n"
        memory_context += "\n"
    
    if escalated_items:
        memory_context += "ESCALATED ITEMS (need attention):\n"
        for item in escalated_items:
            memory_context += f"- {item['what']} (Owner: {item['owner']}, Escalation Level: {item['escalation_level']})\n"
        memory_context += "\n"

    sys = SystemMessage(content=ACTION_EXTRACTION_PROMPT)
    human = HumanMessage(
        content=(
            memory_context +
            "Use both the original transcript and the expanded theme notes.\n\n"
            "Transcript (verbatim):\n" + transcript + "\n\n"
            "Expanded Theme Notes (markdown):\n" + compiled_md
        )
    )

    # Try structured; fall back to manual JSON parsing
    try:
        result = structured.invoke([sys, human])
        if not isinstance(result, Outcomes):
            result = Outcomes(**result)
    except Exception:
        schema_hint = (
            "Return ONLY JSON with keys: actions, decisions, open_questions.\n"
            "Each action: {what, owner, due, why, priority, depends_on[]}\n"
            "Each decision: {statement, rationale, dissent, impact}\n"
            "Each open_question: {question, context, suggested_owner}"
        )
        sys_f = SystemMessage(content=ACTION_EXTRACTION_PROMPT + "\n\nSTRICT: " + schema_hint)
        raw = llm.invoke([sys_f, human])
        text = getattr(raw, "content", "") if raw is not None else "{}"
        if text.strip().startswith("```"):
            text = "\n".join(text.strip().splitlines()[1:-1])
        try:
            data = json.loads(text)
        except Exception:
            data = {"actions": [], "decisions": [], "open_questions": []}

        def _norm_action(a: Any) -> ActionItem:
            if isinstance(a, ActionItem):
                return a
            if not isinstance(a, dict):
                a = {}
            deps = a.get("depends_on")
            if isinstance(deps, str):
                deps = [x.strip() for x in deps.split(",") if x.strip()]
            elif not isinstance(deps, list):
                deps = []
            return ActionItem(
                what=str(a.get("what", "")).strip() or "Unspecified action",
                owner=a.get("owner"),
                due=str(a.get("due")) if a.get("due") is not None else None,
                why=a.get("why"),
                priority=a.get("priority"),
                depends_on=[str(x) for x in deps],
            )

        def _norm_decision(d: Any) -> Decision:
            if isinstance(d, Decision):
                return d
            if not isinstance(d, dict):
                d = {"statement": str(d)}
            return Decision(
                statement=str(d.get("statement", "")).strip() or "Decision recorded",
                rationale=d.get("rationale"),
                dissent=d.get("dissent"),
                impact=d.get("impact"),
            )

        def _norm_q(q: Any) -> OpenQuestion:
            if isinstance(q, OpenQuestion):
                return q
            if not isinstance(q, dict):
                q = {"question": str(q)}
            return OpenQuestion(
                question=str(q.get("question", "")).strip() or "Unspecified question",
                context=q.get("context"),
                suggested_owner=q.get("suggested_owner"),
            )

        result = Outcomes(
            actions=[_norm_action(x) for x in data.get("actions", [])],
            decisions=[_norm_decision(x) for x in data.get("decisions", [])],
            open_questions=[_norm_q(x) for x in data.get("open_questions", [])],
        )

    # Validate counts; retry with stricter instruction if below targets
    if (
        len(result.actions) < min_actions
        or len(result.decisions) < min_decisions
        or len(result.open_questions) < min_questions
    ):
        try:
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
        except Exception:
            # Non-structured fallback to hit minimum counts
            ask = (
                "Return ONLY JSON with keys actions, decisions, open_questions. "
                f"Ensure counts >= {min_actions}, {min_decisions}, {min_questions} respectively."
            )
            raw = llm.invoke([SystemMessage(content=ACTION_EXTRACTION_PROMPT + "\n\nSTRICT: " + ask), human])
            text = getattr(raw, "content", "") if raw is not None else "{}"
            if text.strip().startswith("```"):
                text = "\n".join(text.strip().splitlines()[1:-1])
            try:
                data = json.loads(text)
            except Exception:
                data = {}
            # Merge non-empty lists
            if data.get("actions"):
                result.actions = [ActionItem(**a) if not isinstance(a, ActionItem) else a for a in data.get("actions", [])]
            if data.get("decisions"):
                result.decisions = [Decision(**d) if not isinstance(d, Decision) else d for d in data.get("decisions", [])]
            if data.get("open_questions"):
                result.open_questions = [OpenQuestion(**q) if not isinstance(q, OpenQuestion) else q for q in data.get("open_questions", [])]

    # Normalize owners with attendee list if provided
    actions: List[ActionItem] = []
    for a in result.actions:
        if not isinstance(a, ActionItem):
            a = ActionItem(**a)
        a.owner = _canonicalize_owner(a.owner, attendees)
        actions.append(a)

    # ---------- Matching to existing memory before storing ----------
    class ActionMapping(BaseModel):
        new_index: int
        link_action_id: Optional[str] = None  # existing action_id if match, else null
        reason: Optional[str] = None
        normalized_status: Optional[str] = None  # pending, in_progress, completed, blocked
        normalized_priority: Optional[str] = None  # IMMEDIATE/SHORT_TERM/LONG_TERM

    class ActionMappingList(BaseModel):
        mappings: List[ActionMapping]

    # Build candidate set from memory (pending/in_progress only)
    candidates = [a for a in action_memory.get_pending_action_items(None)]
    cand_payload = [
        {
            "action_id": c.action_id,
            "what": c.what,
            "owner": c.owner,
            "due": c.due,
            "status": c.status,
            "mentions_count": c.mentions_count,
            "escalation_level": c.escalation_level,
        }
        for c in candidates[:40]
    ]

    # Quick exact-match pre-link (owner-insensitive, text-equal)
    index_to_link: Dict[int, str] = {}
    norm = lambda s: (s or "").strip().lower()
    for idx, a in enumerate(actions):
        what = norm(a.what)
        owner = norm(a.owner)
        for c in candidates:
            if norm(c.what) == what and (not owner or norm(c.owner) == owner):
                index_to_link[idx] = c.action_id
                break

    # LLM-assisted linking for the rest
    remaining = [
        {"index": i, "what": a.what, "owner": a.owner, "due": a.due, "priority": a.priority}
        for i, a in enumerate(actions)
        if i not in index_to_link
    ]

    if remaining and cand_payload:
        temperature2 = 0.1
        model2 = provider2 = None
        if config and isinstance(config, dict):
            cfg = config.get("configurable", {}) or {}
            temperature2 = float(cfg.get("temperature", temperature2))
            model2 = cfg.get("model")
            provider2 = cfg.get("provider")
        llm2 = _get_llm(temperature=temperature2, model=model2, provider=provider2)

        sys2 = SystemMessage(
            content=(
                "You link newly extracted action items to an existing action ledger.\n"
                "For each new item, either return link_action_id of the matching existing item or null if new.\n"
                "Match semantically similar items (e.g., 'auth bug' ~ 'login problem'), using owner/due as tie-breakers.\n"
                "Normalize status to: pending, in_progress, completed, blocked.\n"
                "Normalize priority to: IMMEDIATE, SHORT_TERM, LONG_TERM when possible."
            )
        )
        human2 = HumanMessage(
            content=(
                "Existing (candidates):\n"
                + json.dumps(cand_payload, ensure_ascii=False)
                + "\n\nNew (to link):\n"
                + json.dumps(remaining, ensure_ascii=False)
            )
        )
        mappings_obj = llm2.with_structured_output(ActionMappingList).invoke([sys2, human2])
        if not isinstance(mappings_obj, ActionMappingList):
            mappings_obj = ActionMappingList(**mappings_obj)
        for m in mappings_obj.mappings:
            if m.link_action_id is not None:
                index_to_link[m.new_index] = m.link_action_id
                # Apply normalized hints back into action
                if 0 <= m.new_index < len(actions):
                    if m.normalized_status:
                        actions[m.new_index].status = m.normalized_status
                    if m.normalized_priority:
                        actions[m.new_index].priority = m.normalized_priority

    # Store action items in memory for cross-meeting tracking
    stored_actions = []
    for idx, action in enumerate(actions):
        try:
            # If linked, use existing action_id
            if idx in index_to_link:
                action.action_id = index_to_link[idx]
            action_id = action_memory.store_action_item(action, meeting_id, meeting_date)
            action.action_id = action_id
            stored_actions.append(action.model_dump())
        except Exception as e:
            # Fallback: store without memory if there's an error
            print(f"Warning: Could not store action item in memory: {e}")
            stored_actions.append(action.model_dump())

    # Optional approval interrupt before finalizing
    if require_approval:
        interrupt(
            "Please review ACTIONS/DECISIONS/OPEN QUESTIONS.\n"
            "Confirm owners, due dates, priorities, and dependencies."
        )

    # Return as simple dicts for state
    return {
        "actions": stored_actions,
        "decisions": [d.model_dump() if isinstance(d, BaseModel) else d for d in result.decisions],
        "open_questions": [q.model_dump() if isinstance(q, BaseModel) else q for q in result.open_questions],
    }


graph_builder.add_node("load_memory", load_memory_context)
graph_builder.add_node("actions_decisions", extract_outcomes)

graph_builder.add_edge("expand_themes", "load_memory")
graph_builder.add_edge("load_memory", "actions_decisions")

# =============================
# Step 4: Quality Control (QC)
# =============================

class QCFinding(BaseModel):
    issue: str
    fix: str
    evidence: Optional[str] = None
    # Accept any string from model, normalize later to 'minor'/'major' buckets
    severity: Optional[str] = None
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

    # Try structured; fall back to pass-through corrections
    try:
        result = structured.invoke([sys, human])
        if not isinstance(result, QCResult):
            result = QCResult(**result)
    except Exception:
        result = QCResult(
            corrections=QCOutput(
                themes=[t if isinstance(t, Theme) else Theme(**t) for t in themes_in],
                theme_notes=[ThemeNotes(**v) if not isinstance(v, ThemeNotes) else v for _, v in (theme_notes_in or {}).items()],
                actions=[ActionItem(**a) if not isinstance(a, ActionItem) else a for a in actions_in],
                decisions=[Decision(**d) if not isinstance(d, Decision) else d for d in decisions_in],
                open_questions=[OpenQuestion(**q) if not isinstance(q, OpenQuestion) else q for q in open_q_in],
                resources=_gather_resources_from_notes(theme_notes_in),
                metadata=metadata_in if isinstance(metadata_in, dict) else None,
            ),
            findings=[],
        )

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
    # Normalize finding severities to minor/major
    def _norm_sev(s: Optional[str]) -> Optional[str]:
        if not s:
            return s
        t = str(s).strip().lower()
        if t in {"low", "minor", "trivial", "info"}:
            return "minor"
        if t in {"high", "major", "critical"}:
            return "major"
        if t in {"medium", "moderate"}:
            return "minor"
        return t

    findings = []
    for f in result.findings:
        F = f if isinstance(f, QCFinding) else QCFinding(**f)
        d = F.model_dump()
        d["severity"] = _norm_sev(d.get("severity"))
        findings.append(d)

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
    memory_insights_md: str


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


def _mk_memory_insights_md(memory_insights: List[str]) -> str:
    lines = ["## Memory Insights\n"]
    if memory_insights:
        for insight in memory_insights:
            lines.append(f"- {insight}")
    else:
        lines.append("- No previous meeting context available")
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
    memory_insights = state.get("memory_insights", [])
    memory_insights_md = _mk_memory_insights_md(memory_insights)

    # 1) Executive summary
    es_struct = llm.with_structured_output(ExecSummary)
    es_sys = SystemMessage(content=FINAL_COMPILATION_PROMPT)
    # Include memory insights in executive summary
    memory_insights = state.get("memory_insights", [])
    insights_text = ""
    if memory_insights:
        insights_text = "\nMemory Insights:\n" + "\n".join(memory_insights) + "\n\n"
    
    es_human = HumanMessage(
        content=(
            "Craft an executive summary (<=150 words).\n\n"
            "Metadata: " + str(metadata) + "\n\n"
            + insights_text +
            "Decisions: \n" + "\n".join([d["statement"] if isinstance(d, dict) else d.statement for d in decisions]) + "\n\n"
            "Top Actions: \n" + "\n".join([a["what"] if isinstance(a, dict) else a.what for a in actions[:5]]) + "\n\n"
            "Transcript signal (context only):\n" + transcript[:4000]
        )
    )
    try:
        es = es_struct.invoke([es_sys, es_human])
        if not isinstance(es, ExecSummary):
            es = ExecSummary(**es)
    except Exception:
        # Fallback: assemble a concise summary
        md = metadata
        top_actions = ", ".join([(a["what"] if isinstance(a, dict) else a.what) for a in actions[:3]])
        decisions_txt = "; ".join([(d["statement"] if isinstance(d, dict) else d.statement) for d in decisions[:3]])
        text = (
            f"Meeting summary for {md.get('title','this meeting')}. "
            f"Key decisions: {decisions_txt}. "
            f"Top actions: {top_actions}."
        )
        es = ExecSummary(text=text, word_count=_count_words(text))
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
    try:
        body = body_struct.invoke([body_sys, body_human])
        if not isinstance(body, FinalBody):
            body = FinalBody(**body)
    except Exception:
        body = FinalBody(detailed_notes_md=ordered_notes_md, word_count=_count_words(ordered_notes_md))
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
            memory_insights_md=memory_insights_md,
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

# Compile with memory checkpointer for session persistence
checkpointer = MemorySaver()
app = graph_builder.compile(checkpointer=checkpointer)
