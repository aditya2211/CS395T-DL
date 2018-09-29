from keras.applications.xception import (
    Xception, preprocess_input, decode_predictions
)
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('image')

def main(args):

    # create model
    model = load_model(args.model)

    # load an input image
    img = image.load_img(args.image, target_size=(261, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # predict
    pred = model.predict(x)[0]
    print("Image name: %s" % (image))
    print(pred)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)