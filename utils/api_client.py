import requests
import json
import time

class SmartFocusAPIClient:
    def __init__(self, base_url, email, password):       
        self.base_url = f"{base_url}/api/v1"
        self.email = email
        self.password = password
        self.token = None
        self.session_active = False

    def login(self):
        url = f"{self.base_url}/auth/login/access-token"
        data = {
            "username": self.email,
            "password": self.password
        }
        try:
            response = requests.post(url, data=data)
            response.raise_for_status()
            self.token = response.json()["access_token"]
            print("[PI] Logged in successfully.")
            return True
        except Exception as e:
            print(f"[PI] Login failed: {e}")
            return False

    def _get_headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    def start_session(self):
        url = f"{self.base_url}/sessions/start"
        try:
            response = requests.post(url, headers=self._get_headers())
            response.raise_for_status()
            self.session_active = True
            print("[PI] Work session started.")
            return response.json()
        except Exception as e:
            print(f"[PI] Failed to start session: {e}")
            return None

    def send_event(self, event_type, score, details=None):
        url = f"{self.base_url}/events/ingest"
        if details and not isinstance(details, (str, bytes)):
            details = json.dumps(details)
        payload = {
            "event_type": event_type,
            "score": score,
            "details": details
        }
        try:
            response = requests.post(url, headers=self._get_headers(), json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[PI] Failed to send event: {e}")
            return None
