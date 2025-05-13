import cv2
import numpy as np
def density_heatmap(image_path, output_path=None):
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Threshold to find regions of interest
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Create a heatmap (density visualization)
    heatmap = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
    
    # Save or display
    if output_path:
        cv2.imwrite(output_path, overlay)
    cv2.imshow('Density Heatmap', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage
density_heatmap("sample.jpg", "heatmap.jpg")
