"""
This module serves as the API provider for MNIST digit processing.
"""

import io
import json

import numpy as np
from keras.models import load_model
from PIL import Image
from PIL.ImageOps import fit, grayscale

MNIST_MODEL = load_model('var/data/mnist.h5')
print(MNIST_MODEL.summary())

def post_image(file):
    """
    Given a posted image, classify it using the pretrained model.

    This will take 'any size' image, and scale it down to 28x28 like our MNIST
    training data -- and convert to grayscale.

    Parameters
    ----------
    file:
        Bytestring contents of the uploaded file. This will be in an image file format.
    """
    #using Pillow -- python image processing -- to turn the poseted file into bytes
    image = Image.open(io.BytesIO(file.read()))
    image = grayscale(fit(image, (28, 28)))
    image_bytes = image.tobytes()
    #image needs to be a 'batch' though only of one, and with one channel -- grayscale
    image_array = np.reshape(np.frombuffer(image_bytes,  dtype=np.uint8), (1, 28, 28, 1))
    prediction = MNIST_MODEL.predict(image_array)
    #argmax to reverse the one hot encoding
    digit = np.argmax(prediction[0])
    #need to convert to int -- numpy.int64 isn't known to serialize
    return json.dumps({'digit': int(digit)})
