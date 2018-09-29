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
parser.add_argument('valid_image_path')
parser.add_argument('output_file')
def main(args):

    # create model
    model = load_model(args.model)
    f = open(args.output_file,'w')

    for image_name in os.listdir(args.valid_image_path):
    # load an input image
        img = image.load_img(os.path.join(args.valid_image_path,image_name), target_size=(261, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

	    # predict
        pred = model.predict(x)[0]
        f.write("%s\t%f\t%f\n" %(image_name, pred[0], pred[1]))

    f.close()
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
