"""
Download and cache Hugging Face models for offline use.
Run this when you have internet connection, then use offline_mode=True later.
"""

from transformers import pipeline
import sys

def download_model(model_name):
    """Download a model to the Hugging Face cache"""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"{'='*60}")
    
    try:
        # This will download and cache the model
        classifier = pipeline("image-classification", model=model_name)
        print(f"✓ Successfully downloaded and cached: {model_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {model_name}")
        print(f"  Error: {e}")
        return False

def main():
    print("Hugging Face Model Downloader")
    print("="*60)
    print("This script will download models for offline use.")
    print("Make sure you have internet connection!\n")
    
    # List of recommended models
    models = [
        "microsoft/resnet-50",  # General image classification (default)
        "google/vit-base-patch16-224",  # Vision Transformer (alternative)
    ]
    
    print(f"Will download {len(models)} models...\n")
    
    results = {}
    for model in models:
        results[model] = download_model(model)
    
    # Summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    
    for model, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"{status}: {model}")
    
    successful = sum(results.values())
    print(f"\n{successful}/{len(models)} models downloaded successfully")
    
    if successful > 0:
        print("\nYou can now use offline mode:")
        print("   agent = KidSafetyAgent(use_huggingface=True, offline_mode=True)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user")
        sys.exit(1)
