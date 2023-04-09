.PHONY: start
start:
	uvicorn chat_with_x.app.app_v2:app --reload --port 9000

.PHONY: format
format:
	black .
	isort .
