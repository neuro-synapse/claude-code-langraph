PROMPT = """
Create me a company researcher built using langgraph. It should be a multi-node graph. The user should be expected to provide the name of the company and the optional notes if they want. There should be a set maximum search queries that we should do per company and max search results. The LLM should generate the queries that should be searched using the Tavily API to fill the following structured object

  "title": "CompanyInfo",
  "description": "Basic information about a company",
  "type": "object",
  "properties": {
    "company_name": {"type": "string", "description": "Official name of the company"},
    "founding_year": {"type": "integer", "description": "Year the company was founded"},
    "founder_names": {"type": "array", "items": {"type": "string"}, "description": "Names of the founding team members"},
    "product_description": {"type": "string", "description": "Brief description of the company's main product or service"},
    "funding_summary": {"type": "string", "description": "Summary of the company's funding history”}
    "notable_customers": {"type": "string", "description": "Known customers that use company's product/service"}

  },
  "required": ["company_name"]
}

There should be a reflection step at the end. This step should determine if we have good and sufficient information for the the company. If we don’t have sufficient information we should execute web searches again to get information on the company. There should also be a cap on the number of allowed reflection steps. Keep track of the conversation in messages array and try to do web searchers in parallel to improve speed.
"""