"""
data_gen.py - synthetic fraud session generator
"""
import random
import uuid
from typing import Any, Dict, List, Tuple

SIGNAL_EXTRACTORS = {
    "ip_velocity": lambda logs: {
        "unique_ips": len(set(l["ip"] for l in logs)),
        "requests": len(logs),
        "suspicious": len(set(l["ip"] for l in logs)) > 3
    },
    "device_fingerprint": lambda logs: {
        "devices": list(set(l.get("device", "unknown") for l in logs)),
        "suspicious": len(set(l.get("device", "unknown") for l in logs)) > 2
    },
    "login_frequency": lambda logs: {
        "logins": sum(1 for l in logs if l.get("event") == "login"),
        "suspicious": sum(1 for l in logs if l.get("event") == "login") > 5
    },
    "geo_anomaly": lambda logs: {
        "countries": list(set(l.get("country", "US") for l in logs)),
        "suspicious": len(set(l.get("country", "US") for l in logs)) > 1
    },
    "request_pattern": lambda logs: {
        "endpoints": [l.get("endpoint", "/") for l in logs[-5:]],
        "suspicious": any(l.get("endpoint", "").startswith("/admin") for l in logs)
    },
}

def generate_session(difficulty: str = "easy") -> Tuple[List[Dict[str, Any]], str]:
    session_id = str(uuid.uuid4())[:8]
    is_fraud = random.random() < (0.5 if difficulty == "easy" else 0.6 if difficulty == "medium" else 0.7)

    base_ip = f"192.168.{random.randint(1,255)}.{random.randint(1,255)}"
    countries = ["US"]
    devices = ["Chrome/Windows"]
    endpoints = ["/home", "/products", "/cart"]

    if is_fraud:
        if difficulty == "easy":
            ips = [base_ip] + [f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}" for _ in range(5)]
            countries = ["US", "RU", "CN"]
            devices = ["Chrome/Windows", "Bot/Linux", "Unknown"]
            endpoints = ["/login"] * 8 + ["/admin/users", "/admin/export"]
        elif difficulty == "medium":
            ips = [base_ip] + [f"172.16.{random.randint(0,255)}.{random.randint(0,255)}" for _ in range(3)]
            countries = ["US", "VPN"]
            devices = ["Chrome/Windows", "Chrome/Mac"]
            endpoints = ["/login"] * 4 + ["/account/transfer", "/account/settings"]
        else:
            ips = [base_ip, f"10.{random.randint(0,255)}.{random.randint(0,255)}.1"]
            countries = ["US"]
            devices = ["Chrome/Windows"]
            endpoints = ["/login", "/home", "/account/transfer"] * 3
    else:
        ips = [base_ip] * random.randint(3, 6)
        endpoints = ["/home", "/products", "/cart", "/checkout"] * 2

    logs = []
    events = ["login", "page_view", "api_call"]
    for i in range(random.randint(6, 12)):
        logs.append({
            "session_id": session_id,
            "timestamp": f"2024-01-01T10:{i:02d}:00Z",
            "ip": random.choice(ips),
            "device": random.choice(devices),
            "country": random.choice(countries),
            "event": random.choice(events),
            "endpoint": random.choice(endpoints),
        })

    return logs, "fraud" if is_fraud else "real"