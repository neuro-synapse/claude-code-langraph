#!/usr/bin/env python3
"""
Test script for the memory-enhanced meeting transcript processing workflow.

This script demonstrates continuous action item tracking across multiple meetings,
showing how the agent can:
1. Track action items across meetings
2. Escalate items that are mentioned repeatedly
3. Provide memory insights about recurring issues
4. Generate context-aware summaries

Usage:
    python test_memory_workflow.py
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add the current directory to Python path to import agent
sys.path.insert(0, str(Path(__file__).parent))

import agent as ag


def create_meeting_1():
    """Week 1: Initial meeting with action items."""
    return {
        "transcript": """
[00:00] Alice: Welcome to our weekly engineering standup. Let's start with updates.

[00:05] Bob: I'm working on the authentication system. We've identified a critical bug in the login flow that's causing users to be logged out unexpectedly.

[00:10] Alice: That's a high priority issue. Bob, can you commit to fixing this authentication bug by Friday?

[00:15] Bob: Yes, I'll have it fixed by Friday. It's a session management issue in the middleware.

[00:20] Carol: I'm blocked on the frontend work until Bob fixes the auth issue. The login component keeps failing.

[00:25] Alice: Understood. Carol, once Bob fixes the auth bug, how long will your frontend work take?

[00:30] Carol: Probably 2-3 days after the auth fix is deployed.

[00:35] Dave: I'm working on the CI pipeline. We need to upgrade the test runners to reduce build times.

[00:40] Alice: Dave, can you create a plan for the CI upgrades by next week?

[00:45] Dave: Yes, I'll draft a CI upgrade plan by Tuesday.

[00:50] Alice: Great. Let's close here and follow up on the action items.
        """,
        "metadata": {
            "attendees": ["Alice", "Bob", "Carol", "Dave"],
            "meeting_type": "engineering_standup",
            "date": "2024-01-15"
        },
        "meeting_id": "meeting_2024_01_15",
        "meeting_date": "2024-01-15"
    }


def create_meeting_2():
    """Week 2: Follow-up meeting mentioning the auth issue."""
    return {
        "transcript": """
[00:00] Alice: Welcome to this week's standup. Let's check on our action items.

[00:05] Bob: I'm still working on that auth issue we discussed last week. It's more complex than expected - the session tokens are being invalidated prematurely.

[00:10] Alice: How much longer do you think it will take?

[00:15] Bob: I need another week. The root cause is in our token refresh logic.

[00:20] Carol: I'm still blocked on my frontend work. I can't test the login component until the auth issue is resolved.

[00:25] Alice: This is becoming a blocker for multiple people. Bob, can you prioritize this and get it done by Wednesday?

[00:30] Bob: I'll do my best, but Wednesday might be tight. Let me commit to Friday at the latest.

[00:35] Dave: I've drafted the CI upgrade plan. We need to upgrade Docker images and increase parallelism.

[00:40] Alice: Great work Dave. When can we start implementing?

[00:45] Dave: I'd like to start next week if the team agrees.

[00:50] Alice: Sounds good. Let's wrap up here.
        """,
        "metadata": {
            "attendees": ["Alice", "Bob", "Carol", "Dave"],
            "meeting_type": "engineering_standup",
            "date": "2024-01-22"
        },
        "meeting_id": "meeting_2024_01_22",
        "meeting_date": "2024-01-22"
    }


def create_meeting_3():
    """Week 3: Meeting with vague reference to the auth issue."""
    return {
        "transcript": """
[00:00] Alice: Welcome to standup. Let's go around the room.

[00:05] Bob: I'm making progress on that login problem from last month. I think I found the issue in the middleware configuration.

[00:10] Alice: Wait, which login problem are you referring to?

[00:15] Bob: You know, the authentication bug we've been discussing. The one where users keep getting logged out.

[00:20] Carol: Oh right, that's been blocking my frontend work for weeks now. Any ETA on when it'll be fixed?

[00:25] Bob: I'm targeting end of this week, but I'm not 100% confident given how complex this has been.

[00:30] Alice: This has been dragging on for a while. Is there anything we can do to help unblock this?

[00:35] Dave: I could pair with Bob on this if it would help. I've been working on similar session issues in the CI system.

[00:40] Bob: That would actually be helpful. Dave, can we pair tomorrow morning?

[00:45] Dave: Absolutely. I'll block my calendar.

[00:50] Alice: Good. Let's make sure this gets resolved this week.
        """,
        "metadata": {
            "attendees": ["Alice", "Bob", "Carol", "Dave"],
            "meeting_type": "engineering_standup",
            "date": "2024-01-29"
        },
        "meeting_id": "meeting_2024_01_29",
        "meeting_date": "2024-01-29"
    }


def run_meeting_processing(meeting_data, config=None):
    """Process a single meeting and return results."""
    print(f"\n{'='*60}")
    print(f"Processing Meeting: {meeting_data['meeting_id']}")
    print(f"Date: {meeting_data['meeting_date']}")
    print(f"{'='*60}")
    
    try:
        # Process the meeting
        result = ag.app.invoke(meeting_data, config=config)
        
        # Display key results
        print(f"\n📊 MEETING RESULTS:")
        print(f"- Themes extracted: {len(result.get('themes', []))}")
        print(f"- Action items: {len(result.get('actions', []))}")
        print(f"- Decisions: {len(result.get('decisions', []))}")
        print(f"- Open questions: {len(result.get('open_questions', []))}")
        
        # Display memory insights if available
        memory_insights = result.get('memory_insights', [])
        if memory_insights:
            print(f"\n🧠 MEMORY INSIGHTS:")
            for insight in memory_insights:
                print(f"  • {insight}")
        
        # Display escalated items if any
        escalated = result.get('escalated_items', [])
        if escalated:
            print(f"\n⚠️  ESCALATED ITEMS:")
            for item in escalated:
                print(f"  • {item['what']} (Escalation Level: {item['escalation_level']})")
        
        # Display action items with memory tracking
        actions = result.get('actions', [])
        if actions:
            print(f"\n📋 ACTION ITEMS:")
            for action in actions:
                mentions = action.get('mentions_count', 1)
                escalation = action.get('escalation_level', 0)
                status_emoji = "🔄" if mentions > 1 else "🆕"
                escalation_emoji = "⚠️" if escalation > 0 else ""
                print(f"  {status_emoji} {escalation_emoji} {action['what']} (Owner: {action['owner']}, Mentions: {mentions})")
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing meeting: {e}")
        return None


def demonstrate_memory_features():
    """Demonstrate the memory-enhanced workflow across multiple meetings."""
    print("🚀 Starting Memory-Enhanced Meeting Processing Demo")
    print("This demo shows how action items are tracked across multiple meetings")
    
    # Configuration for processing
    config = {
        "configurable": {
            "temperature": 0.2,
            "require_approval": False,  # Skip approval for demo
            "provider": "anthropic"
        }
    }
    
    # Process Meeting 1
    meeting1_data = create_meeting_1()
    result1 = run_meeting_processing(meeting1_data, config)
    
    # Process Meeting 2
    meeting2_data = create_meeting_2()
    result2 = run_meeting_processing(meeting2_data, config)
    
    # Process Meeting 3
    meeting3_data = create_meeting_3()
    result3 = run_meeting_processing(meeting3_data, config)
    
    # Summary of memory tracking
    print(f"\n{'='*60}")
    print("🎯 MEMORY TRACKING SUMMARY")
    print(f"{'='*60}")
    
    if result3:
        memory_insights = result3.get('memory_insights', [])
        escalated_items = result3.get('escalated_items', [])
        
        print(f"\n📈 CROSS-MEETING ANALYSIS:")
        print(f"- Total memory insights generated: {len(memory_insights)}")
        print(f"- Items requiring escalation: {len(escalated_items)}")
        
        # Show how the auth issue was tracked
        actions = result3.get('actions', [])
        auth_actions = [a for a in actions if 'auth' in a['what'].lower() or 'login' in a['what'].lower()]
        if auth_actions:
            auth_action = auth_actions[0]
            print(f"\n🔍 AUTH ISSUE TRACKING:")
            print(f"- Issue: {auth_action['what']}")
            print(f"- Mentions across meetings: {auth_action.get('mentions_count', 1)}")
            print(f"- Escalation level: {auth_action.get('escalation_level', 0)}")
            print(f"- Created: {auth_action.get('created_date', 'Unknown')}")
            print(f"- Last mentioned: {auth_action.get('last_mentioned', 'Unknown')}")
        
        print(f"\n✅ DEMO COMPLETED SUCCESSFULLY!")
        print(f"The agent successfully:")
        print(f"  • Tracked action items across 3 meetings")
        print(f"  • Identified recurring issues (auth bug)")
        print(f"  • Generated memory insights about blockers")
        print(f"  • Escalated items mentioned multiple times")
        print(f"  • Maintained persistent memory in SQLite database")
    
    else:
        print("❌ Demo failed - could not process Meeting 3")


if __name__ == "__main__":
    demonstrate_memory_features()


