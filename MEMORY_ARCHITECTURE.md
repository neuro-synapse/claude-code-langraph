# Memory Architecture Diagram

## LangGraph Memory-Enhanced Meeting Processing Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MEETING PROCESSING WORKFLOW                       │
└─────────────────────────────────────────────────────────────────────────────┘

Meeting Input
     │
     ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│ Preprocess  │───▶│ Theme Extract│───▶│Expand Themes│───▶│Load Memory  │
│             │    │              │    │             │    │             │
│ • Validate  │    │ • Identify   │    │ • Detailed  │    │ • Previous  │
│   transcript│    │   themes     │    │   notes     │    │   action    │
│ • Normalize │    │ • Categorize │    │ • Quotes    │    │   items     │
│   input     │    │   by tier    │    │ • Resources │    │ • Escalated │
└─────────────┘    └──────────────┘    └─────────────┘    │   items     │
                                                          │ • Insights  │
                                                          └─────────────┘
                                                                 │
                                                                 ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│Actions &    │───▶│QC Enhance   │───▶│Final        │───▶│ Output      │
│Decisions    │    │             │    │Compile      │    │             │
│             │    │ • Quality   │    │             │    │ • Enhanced  │
│ • Extract   │    │   control   │    │ • Executive │    │   summary   │
│   actions   │    │ • Fix       │    │   summary   │    │ • Memory    │
│ • Store in  │    │   issues    │    │ • Detailed  │    │   insights  │
│   memory    │    │ • Normalize │    │   notes     │    │ • Action    │
│ • Track     │    │   data      │    │ • Appendices│    │   tracking  │
│   mentions  │    │             │    │ • Memory    │    │             │
└─────────────┘    └─────────────┘    │   insights  │    └─────────────┘
                                      └─────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           PERSISTENT MEMORY LAYER                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ SQLite      │    │ Action Item │    │ Escalation  │
│ Database    │    │ Tracking    │    │ Logic       │
│             │    │             │    │             │
│ • action_   │    │ • Mentions  │    │ • Based on  │
│   items     │    │   count     │    │   frequency │
│ • meetings  │    │ • History   │    │ • Due dates │
│ • metadata  │    │ • Status    │    │ • Blockers  │
│             │    │ • Owners    │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MEMORY FEATURES                                  │
└─────────────────────────────────────────────────────────────────────────────┘

• Cross-Meeting Tracking    • Automatic Escalation    • Context Awareness
• Persistent Storage        • Pattern Recognition     • Dependency Mapping
• Smart Deduplication       • Historical Insights     • Proactive Alerts

┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXAMPLE SCENARIO                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Week 1: "John commits to fixing the authentication bug by Friday"
        │
        ▼ Store in memory with action_id, meeting_id, created_date
        │
Week 2: "Someone mentions 'the auth issue' without context"
        │
        ▼ Recognize as same issue, increment mentions_count, update last_mentioned
        │
Week 3: "Discussion about 'that login problem from last month'"
        │
        ▼ Identify as recurring issue, escalate priority, generate insights:
        │
        └─► "Authentication bug has been discussed for 3 weeks without resolution"
            "This item is blocking Carol's frontend work mentioned in previous meetings"
            "Item has escalated to urgent priority due to repeated mentions"
```

## Key Components

### 1. Enhanced Action Item Model
```python
class ActionItem(BaseModel):
    # Original fields
    what: str
    owner: Optional[str] = None
    due: Optional[str] = None
    
    # Memory tracking fields
    action_id: Optional[str] = None
    meeting_id: Optional[str] = None
    mentions_count: int = 1
    escalation_level: int = 0
    history: str  # JSON array of meeting mentions
```

### 2. Memory Management Class
```python
class ActionItemMemory:
    def store_action_item(self, action_item, meeting_id, meeting_date)
    def get_pending_action_items(self, owner=None)
    def get_escalated_items(self)
    def _calculate_escalation(self, action_item)
```

### 3. Graph Integration
- New `load_memory` node loads historical context
- Enhanced `extract_outcomes` includes memory context in prompts
- Final compilation includes memory insights in summaries

### 4. Database Schema
```sql
CREATE TABLE action_items (
    action_id TEXT PRIMARY KEY,
    what TEXT NOT NULL,
    owner TEXT,
    mentions_count INTEGER DEFAULT 1,
    escalation_level INTEGER DEFAULT 0,
    history TEXT  -- JSON array of meeting mentions
);
```

## Benefits

✅ **Continuous Tracking**: Action items persist across meetings  
✅ **Automatic Escalation**: Recurring issues get appropriate attention  
✅ **Context Awareness**: Meeting summaries include historical perspective  
✅ **Pattern Recognition**: AI identifies systemic issues and blockers  
✅ **Accountability**: Clear tracking of commitments and follow-through  
✅ **Dependency Mapping**: Understanding of what's blocking other work


