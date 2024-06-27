import os
from PIL import Image
import imageio.v2 as imageio

path="../merge/test/ss"
new_width=1024
new_height=1024
for filename in os.listdir(path):
    if ".txt" not in filename:
        image = imageio.imread(path+"/"+filename)  # read image
        img = Image.fromarray(image).resize((new_width,new_height))
        imageio.imwrite(path+"/"+filename, img)