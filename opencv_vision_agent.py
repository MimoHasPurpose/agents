"""
Offline Vision Agent using OpenCV and basic computer vision techniques.
No internet connection required!
"""

import cv2
import numpy as np
import random


class OpenCVVisionAgent:
    """
    Vision agent using OpenCV for action detection.
    Works completely offline without any model downloads.
    """
    
    def __init__(self):
        """Initialize OpenCV-based vision agent"""
        print("✓ Initializing OpenCV Vision Agent (offline mode)")
        
        # Motion detection parameters
        self.prev_frame = None
        self.motion_threshold = 0.3
        
    def detect_motion(self, frame):
        """
        Detect motion in frame using frame differencing.
        
        Args:
            frame: numpy array (image) or PIL Image
            
        Returns:
            float: Motion intensity (0-1)
        """
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Apply Gaussian blur
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # First frame
            if self.prev_frame is None:
                self.prev_frame = gray
                return 0.0
            
            # Compute frame difference
            frame_delta = cv2.absdiff(self.prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Calculate motion percentage
            motion_pixels = np.sum(thresh > 0)
            total_pixels = thresh.size
            motion_ratio = motion_pixels / total_pixels
            
            # Update previous frame
            self.prev_frame = gray
            
            return min(motion_ratio * 5, 1.0)  # Scale up and cap at 1.0
            
        except Exception as e:
            print(f"Error in motion detection: {e}")
            return 0.0
    
    def detect_edges(self, frame):
        """
        Detect edges in frame (can indicate sudden movements).
        
        Args:
            frame: numpy array (image)
            
        Returns:
            float: Edge density (0-1)
        """
        try:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 100, 200)
            
            # Calculate edge density
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.size
            edge_ratio = edge_pixels / total_pixels
            
            return min(edge_ratio * 10, 1.0)  # Scale up and cap at 1.0
            
        except Exception as e:
            print(f"Error in edge detection: {e}")
            return 0.0
    
    def analyze_image(self, image_path_or_array):
        """
        Analyze an image for potential concerning activity.
        
        Args:
            image_path_or_array: Path to image file or numpy array
            
        Returns:
            dict with analysis results
        """
        try:
            # Load image
            if isinstance(image_path_or_array, str):
                frame = cv2.imread(image_path_or_array)
            elif isinstance(image_path_or_array, np.ndarray):
                frame = image_path_or_array
            else:
                # PIL Image
                frame = np.array(image_path_or_array)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if frame is None:
                return {
                    'motion_score': 0.0,
                    'edge_score': 0.0,
                    'risk_score': 0.0,
                    'analysis': 'Unable to load image'
                }
            
            # Analyze frame
            motion_score = self.detect_motion(frame)
            edge_score = self.detect_edges(frame)
            
            # Combine scores for risk assessment
            # High motion + high edges might indicate aggressive movement
            risk_score = (motion_score * 0.6 + edge_score * 0.4)
            
            return {
                'motion_score': motion_score,
                'edge_score': edge_score,
                'risk_score': risk_score,
                'analysis': f'Motion: {motion_score:.2f}, Edges: {edge_score:.2f}'
            }
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return {
                'motion_score': 0.0,
                'edge_score': 0.0,
                'risk_score': 0.0,
                'analysis': f'Error: {e}'
            }
    
    def analyze_frame(self, frame):
        """
        Compatibility method for existing code structure.
        Analyzes a frame and returns just the risk score.
        """
        if isinstance(frame, str) and frame == "video_frame":
            # Simulated frame - return random low risk
            return random.uniform(0.1, 0.4)
        
        result = self.analyze_image(frame)
        return result['risk_score']


# Example usage
if __name__ == "__main__":
    print("OpenCV Vision Agent - Offline Mode\n")
    
    # Create agent
    agent = OpenCVVisionAgent()
    
    print("\nThis agent works completely offline!")
    print("It uses motion detection and edge detection to assess risk.")
    print("\nTo use with images:")
    print('  result = agent.analyze_image("path/to/image.jpg")')
    print("\nTo integrate with main.py:")
    print("  Replace VisionAgent with OpenCVVisionAgent in KidSafetyAgent.__init__")
