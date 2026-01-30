# Hugging Face Agent for Action Recognition

This project includes a Hugging Face-powered vision agent that can identify actions in images.

## Installation

```bash
pip install transformers pillow torch torchvision
```

## Usage

### 1. Basic Usage with Simulated Agent (Fast)

```python
from main import KidSafetyAgent

# Use simulated agent (no model download needed)
agent = KidSafetyAgent(use_huggingface=False)
agent.monitor()
```

### 2. Using Real Hugging Face Model

```python
from main import KidSafetyAgent

# Use real Hugging Face model for vision analysis
agent = KidSafetyAgent(use_huggingface=True)
agent.monitor()
```

### 3. Standalone Hugging Face Vision Agent

```python
from main import HuggingFaceVisionAgent

# Initialize the agent
vision_agent = HuggingFaceVisionAgent(model_name="microsoft/resnet-50")

# Analyze an image
result = vision_agent.analyze_image("path/to/image.jpg")

print(f"Top Action: {result['top_action']}")
print(f"Risk Score: {result['risk_score']}")
print(f"All Predictions: {result['predictions']}")
```

### 4. Run Example Script

```bash
python example_with_images.py
```

## Available Models

You can use different pre-trained models:

- **`microsoft/resnet-50`** - General image classification (default)
- **`facebook/timesformer-base-finetuned-k400`** - Video action recognition
- **`google/vit-base-patch16-224`** - Vision Transformer
- **`openai/clip-vit-base-patch32`** - CLIP model for vision-language tasks

## Features

- **Action Recognition**: Identifies actions and behaviors in images
- **Risk Assessment**: Calculates risk scores based on detected actions
- **Multi-modal Integration**: Works with the existing speech and decision agents
- **Flexible Input**: Accepts image paths, PIL Images, or numpy arrays

## How It Works

The `HuggingFaceVisionAgent` class:

1. Loads a pre-trained model from Hugging Face Hub
2. Analyzes images to identify actions and objects
3. Checks for concerning keywords (fighting, violence, etc.)
4. Returns risk scores and detailed predictions
5. Integrates seamlessly with the existing monitoring system

## Example Output

```
Loading Hugging Face model: microsoft/resnet-50...
✓ Using Hugging Face Vision Agent

Top detected action: people playing basketball
Risk Score: 0.150

All predictions:
  - people playing basketball: 0.856
  - sports facility: 0.092
  - gymnasium: 0.034
```