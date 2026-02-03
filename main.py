

import time
from enum import Enum
from transformers import pipeline
from PIL import Image   
import numpy as np


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


 
# Hugging Face Vision Agent

class HuggingFaceVisionAgent:
    def __init__(self, model_name="microsoft/resnet-50", offline_mode=False):
        """
        Initialize Hugging Face agent for action recognition in images.
        
        
                       - "microsoft/resnet-50" for general image classification
                       - "facebook/timesformer-base-finetuned-k400" for action recognition
                       - "MCG-NJU/videomae-base" for video action detection
        """
        print(f"Loading Hugging Face model: {model_name}...")
        try:
            if offline_mode:
                print(" Offline mode enabled - using cached models only")
                self.classifier = pipeline("image-classification", model=model_name, local_files_only=True)
            else:
                self.classifier = pipeline("image-classification", model=model_name)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print(" Falling back to simulated mode")
            print("   To fix: Check internet connection or use offline_mode=True with cached models")
            raise RuntimeError(f"Failed to load Hugging Face model: {e}")
        
        # Keywords that might indicate concerning actions
        self.risk_keywords = [
            "fighting", "hitting", "punching", "violence", "aggressive",
            "weapon", "attack", "assault", "pushing", "shoving", "threatening"
        ]
        
    def analyze_image(self, image_path_or_array):
        """        
        Args:
            image_path_or_array: Path to image file or PIL Image or numpy array
        Returns:
            dict with predictions, risk_score, and analysis
        """
        try:
            # Run inference
            predictions = self.classifier(image_path_or_array, top_k=5)
            
            # Calculate risk score based on detected actions
            risk_score = 0.0
            detected_concerns = []
            
            for pred in predictions:
                label = pred['label'].lower()
                confidence = pred['score']
                
                # Check if any risk keywords are in the label
                for keyword in self.risk_keywords:
                    if keyword in label:
                        risk_score = max(risk_score, confidence)
                        detected_concerns.append({
                            'action': label,
                            'confidence': confidence
                        })
            
            # Normalize risk score to 0-1 range
            # If no concerning actions detected, risk score remains 0
            
            return {
                'predictions': predictions,
                'risk_score': risk_score,
                'concerns': detected_concerns,
                'top_action': predictions[0]['label'] if predictions else None
            }
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return {
                'predictions': [],
                'risk_score': 0.0,
                'concerns': [],
                'top_action': None
            }
    
    def analyze_frame(self, frame):
        """
        Compatibility method for existing code structure.
        Analyzes a frame and returns just the risk score.
        """
        result = self.analyze_image(frame)
        return result['risk_score']



class DecisionAgent:
    def assess_risk(self, vision_score, speech_score):
        """
        Combine multi-modal signals.
        """
        combined_score = (vision_score + speech_score) / 2

        if combined_score < 0.3:
            return RiskLevel.LOW
        elif combined_score < 0.6:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH



class ActionAgent:
    def take_action(self, risk_level):
        if risk_level == RiskLevel.LOW:
            self.log_event("Low-risk behavior logged.")
        elif risk_level == RiskLevel.MEDIUM:
            self.alert_teacher()
        elif risk_level == RiskLevel.HIGH:
            self.trigger_emergency()

    def log_event(self, msg):
        print(f"[LOG] {msg}")

    def alert_teacher(self):
        print("[ALERT] Teacher notified about potential harassment.")

    def trigger_emergency(self):
        print("[EMERGENCY] Immediate action required!")
        print("[EMERGENCY] School authority & child safety team notified.")



class MemoryAgent:
    def __init__(self):
        self.history = []

    def store_event(self, vision, speech, risk):
        self.history.append({
            "vision_score": vision,
            "speech_score": speech,
            "risk": risk.value,
            "timestamp": time.time()
        })

    def show_summary(self):
        print("\n--- INCIDENT SUMMARY ---")
        for h in self.history:
            print(h)


class KidSafetyAgent:
    def __init__(self, offline_mode=False):
        """
        Initialize Kid Safety Agent.
        
        Args:
            offline_mode: If True, use only cached models (requires prior download)
        """
        try:
            self.vision_agent = HuggingFaceVisionAgent(offline_mode=offline_mode)
            print("✓ Using Hugging Face Vision Agent")
        except Exception as e:
            print(f" Failed to initialize Hugging Face agent: {e}")
            raise RuntimeError("Vision agent is required. Please check your configuration.")
        self.decision_agent = DecisionAgent()
        self.action_agent = ActionAgent()
        self.memory_agent = MemoryAgent()

    def monitor(self):
        print("Kid Safety Agent started...\n")

        for step in range(10):  # simulate 10 monitoring cycles code but will run indefinitely if used in real model
            print(f"\n[Cycle {step + 1}]")

            frame = "video_frame"
            audio = "audio_chunk"

            vision_score = self.vision_agent.analyze_frame(frame)
            speech_score = 0.0  # Speech analysis not implemented yet

            risk = self.decision_agent.assess_risk(
                vision_score, speech_score
            )

            print(f"Vision Risk: {vision_score:.2f}")
            print(f"Combined Risk Level: {risk.value}")

            self.memory_agent.store_event(
                vision_score, speech_score, risk
            )

            self.action_agent.take_action(risk)

            time.sleep(1)

        self.memory_agent.show_summary()



if __name__ == "__main__":
    # Options:
    # 1. offline_mode=False - Download model from internet (requires connection)
    # 2. offline_mode=True - Use cached model (must download first)
    
    agent = KidSafetyAgent(offline_mode=False)
    agent.monitor()
