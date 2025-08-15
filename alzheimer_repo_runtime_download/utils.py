
from PIL import Image
import numpy as np

def preprocess_pil_image(img, size=(128,128)):
    img = img.convert('RGB').resize(size)
    arr = np.array(img)/255.0
    arr = arr.reshape(1, size[0], size[1], 3)
    return arr
