"""
app.py - FastAPI server setup
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openenv.core.env_server.http_server import create_app
from fraud_detect_env_environment import FraudDetectEnvironment
from models import FraudDetectAction, FraudDetectObservation

# Create OpenEnv app
app = create_app(
    env=FraudDetectEnvironment,
    action_cls=FraudDetectAction,
    observation_cls=FraudDetectObservation,
    env_name="fraud_detect_env",
)

# Root endpoint for HuggingFace
@app.get("/")
def root():
    return {"message": "Fraud Detection Env Running"}