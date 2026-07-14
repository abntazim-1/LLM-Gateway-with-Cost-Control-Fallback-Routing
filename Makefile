.PHONY: run test loadtest demo dashboard clean

run:
	uvicorn gateway.main:app --reload --port 8080

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/

loadtest:
	locust -f loadtest/gateway_loadtest.py --host http://localhost:8080

demo:
	@echo "Running demo failover script..."
	# This would run a specialized script highlighting the circuit breaker
	cat docs/DEMO_SCRIPT.md

clean:
	rm -f ledger.db
	rm -rf __pycache__ gateway/__pycache__ dashboard/__pycache__
