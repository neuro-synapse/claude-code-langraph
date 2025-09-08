PROMPT = """
Create a LangGraph-based text-to-SQL agent. It should generate sql, execute that sql against the chinook sqlite database, and then generate a response answer in natural language based on the result. 
To use chinook db, fetch it from `https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql` and create  in-memory db using langgraph. 
It might be useful to pass in the detailed info of the table schema to the prompt so the agent can convert user's natural request to correct sql. Whenever, the query is irrelevant, or cannot be answered using the sql db search, just say you don't know the answer and don't talk about anything. 
Your purpose is to only convert text request to sql and generate response in natural language.
"""