UVX = uvx

.PHONY: smoke live

CLI_DEPS = langgraph langchain-core langchain-google-genai google-generativeai pydantic langchain-openai langchain-anthropic

smoke:
	$(UVX) --with langgraph --with langchain-core --with pydantic python run_smoke_test.py

live:
	# Requires GOOGLE_API_KEY set in environment
	$(UVX) \
	  --with $(CLI_DEPS) \
	  python run_live_test.py

.PHONY: cli
cli:
	# Example: make cli TRANSCRIPT=transcript.txt OUTPUT=out.md PROVIDER=google MODEL=gemini-2.5-pro
	@if [ -z "$(TRANSCRIPT)" ]; then echo "Usage: make cli TRANSCRIPT=path OUTPUT=out.md [PROVIDER=google MODEL=gemini-2.5-pro]" && exit 2; fi
	$(UVX) --with $(CLI_DEPS) python cli.py --transcript "$(TRANSCRIPT)" --output "$(OUTPUT)" $(if $(PROVIDER),--provider $(PROVIDER),) $(if $(MODEL),--model $(MODEL),)
