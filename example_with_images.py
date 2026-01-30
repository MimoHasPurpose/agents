"""
Example: Using Hugging Face Vision Agent with real images
"""

from main import HuggingFaceVisionAgent
from PIL import Image
import requests
from io import BytesIO

def download_sample_image(url):
    """Download an image from URL"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def main():
    # Initialize the Hugging Face Vision Agent
    # You can try different models:
    # - "microsoft/resnet-50" - general image classification
    # - "facebook/timesformer-base-finetuned-k400" - video action recognition
    # - "google/vit-base-patch16-224" - Vision Transformer
    
    agent = HuggingFaceVisionAgent(model_name="microsoft/resnet-50")
    
    # Example 1: Analyze an image from URL
    print("\n" + "="*60)
    print("Example 1: Analyzing image from URL")
    print("="*60)
    
    sample_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    
    try:
        image = download_sample_image(sample_url)
        result = agent.analyze_image(image)
        
        print(f"\nTop detected action: {result['top_action']}")
        print(f"Risk Score: {result['risk_score']:.3f}")
        
        print("\nAll predictions:")
        for pred in result['predictions']:
            print(f"  - {pred['label']}: {pred['score']:.3f}")
        
        if result['concerns']:
            print("\n⚠️ Concerns detected:")
            for concern in result['concerns']:
                print(f"  - {concern['action']}: {concern['confidence']:.3f}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Analyze a local image file
    print("\n" + "="*60)
    print("Example 2: Analyzing local image")
    print("="*60)
    print("To use a local image, provide the file path:")
    print('  result = agent.analyze_image("path/to/your/image.jpg")')
    
    # Example 3: Using with the monitoring system
    print("\n" + "="*60)
    print("Example 3: Integration with monitoring system")
    print("="*60)
    print("""
To use with the monitoring system:

from main import KidSafetyAgent

# Create agent with Hugging Face vision
agent = KidSafetyAgent(use_huggingface=True)

# For real images, modify the monitor() method to pass actual images
# instead of the string "video_frame"
""")

if __name__ == "__main__":
    main()
