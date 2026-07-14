from locust import HttpUser, task, between
import json

class GatewayUser(HttpUser):
    wait_time = between(0.1, 1.0)
    
    def on_start(self):
        self.headers = {
            "Authorization": "Bearer sk-test-tier-1",
            "Content-Type": "application/json"
        }
        self.payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "Hello, how are you today?"}
            ],
            "max_tokens": 50
        }

    @task
    def test_completion(self):
        with self.client.post(
            "/v1/chat/completions",
            headers=self.headers,
            json=self.payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                response.failure("Rate limited / Budget exceeded")
            else:
                response.failure(f"Failed with status: {response.status_code}")
