import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def remove_background(image_path, progress_bar):
    img = np.array(Image.open(image_path))
    
    # Define color range for shiny area (adjust the values according to your image)
    lower_shiny = np.array([200, 200, 200], dtype=np.uint8)
    upper_shiny = np.array([255, 255, 255], dtype=np.uint8)
    
    # Create a mask for the shiny area
    shiny_mask = cv2.inRange(img, lower_shiny, upper_shiny)
    
    # Initialize the mask for grabCut algorithm
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # Set the probable foreground and probable background areas
    mask[shiny_mask == 0] = cv2.GC_PR_BGD
    mask[shiny_mask == 255] = cv2.GC_PR_FGD
    
    # Run grabCut algorithm
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, img.shape[1] - 50, img.shape[0] - 50)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create a mask where probable background and certain background are combined
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply the mask to the image
    img = img * mask[:, :, np.newaxis]
    
    # Replace black pixels with white pixels
    img[np.where((img == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    
    progress_bar.progress(100)  # Update progress to 100%
    return img

def main():
    st.title("Image Background Removal")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_path = 'temp_image.jpg'
        image.save(image_path)  # Save the image to a temporary file
        st.image(image, caption='Original Image')
        st.write("Removing background...")
        
        progress_bar = st.progress(0)  # Initialize progress bar
        
        removed_bg_img = remove_background(image_path, progress_bar)
        
        st.image(removed_bg_img, caption='Image with Background Removed')
        st.write("Background removed!")
        st.write("Note: The output image may contain some errors. I can further improve the results by tweaking the background removal parameters in this code. Kindly message me I will be waiting for you. I have built this in  Python Streamlit(Regards Afaq AHMAD)")
        
        # Update progress bar to complete
        progress_bar.progress(100)
        
        # Remove the temporary image file
        os.remove(image_path)

if __name__ == "__main__":
    main()
