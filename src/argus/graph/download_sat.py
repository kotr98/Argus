import requests
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def download_satellite_image(west, south, east, north, width=512, height=512):
    #print("in download: ", width, height)
    """
    Downloads a satellite image for the specified bounding box using the ArcGIS REST API.

    Parameters:
        west (float): Western longitude.
        south (float): Southern latitude.
        east (float): Eastern longitude.
        north (float): Northern latitude.
        width (int): Width of the returned image in pixels.
        height (int): Height of the returned image in pixels.

    Returns:
        numpy.ndarray: The satellite image as a NumPy array of shape (height, width, 3).
    """
    url = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
    params = {
        "bbox": f"{west},{south},{east},{north}",
        "bboxSR": "4326",        # Spatial reference of the bbox
        "imageSR": "4326",       # Spatial reference for the output image
        "size": f"{width},{height}",
        "format": "jpg",         # Output format; JPEG returns an image with 3 channels (RGB)
        "f": "image"             # Response format (direct image)
    }
    
    response = requests.get(url, params=params, timeout=300)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    # Open the image and convert it to a numpy array
    img = Image.open(BytesIO(response.content))
    img_array = np.array(img)
    
    # Flip the array vertically to move the origin to the bottom-left
    img_array = np.flipud(img_array)
    img_array = cv2.resize(img_array, (int(width), int(height)))
    #print("sat: ", img_array.shape)
    
    return (img_array*0.5).astype(np.uint8)

# Example usage:
if __name__ == '__main__':
    # Define bounding box coordinates
    west = 11.639342
    south = 48.241096
    east = 11.687758
    north = 48.268999

    # Download the satellite image
    image_array = download_satellite_image(west, south, east, north, width=4061, height=2956)

    plt.imshow(image_array)
    plt.show()

    #print("Downloaded image shape:", image_array.shape)