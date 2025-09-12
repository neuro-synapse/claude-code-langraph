#!/usr/bin/env python3
"""
Quick test to verify Google AI API connection works.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI

def test_api_connection():
    try:
        # Test basic connection
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)
        
        # Simple test query
        response = llm.invoke("Say 'Hello, LangGraph test successful!' and nothing else.")
        print(f"✅ API Connection Success!")
        print(f"Model Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ API Connection Failed: {e}")
        return False

if __name__ == "__main__":
    test_api_connection()