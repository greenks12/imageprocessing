import matplotlib.pyplot as plt
from PIL import Image
import os

image_directory = r"F:\Work\Project\runs\detect\train2" 

for filename in os.listdir(image_directory):
    if filename.lower().endswith((".jpg", ".png")):
        image_path = os.path.join(image_directory, filename)
        image = Image.open(image_path)

        plt.figure(figsize=(12, 12), dpi=150)
        plt.imshow(image)
        plt.title(f"Image: {filename}", fontsize=20, fontweight='bold', color='blue')
        plt.axis("off")
        plt.show()
