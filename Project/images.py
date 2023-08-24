import pandas as pd
import random
import cv2
import os

test_set = pd.read_csv('Project/imageRecognition/mnist_test.csv', index_col=None)

lis = []

for _ in range(5):
    lis.append(random.randint(0, test_set.shape[0]))

# Specify the directory where you want to save the images
save_dir = 'Project/saved_images/'

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

for i in lis:
    image_pixel_values = test_set.iloc[i, 1:] 
    
    # Reshape the pixel values into a 28x28 array
    image_array = image_pixel_values.values.reshape(-1, 28, 28, 1)
    
    image_filename = f'image_{i}.png'
    image_filepath = os.path.join(save_dir, image_filename)
    cv2.imwrite(image_filepath, image_array[0] * 255)  # Multiply by 255 to convert back to pixel values
    
    print(f"Saved image {image_filename}")

print("Images saved successfully.")
