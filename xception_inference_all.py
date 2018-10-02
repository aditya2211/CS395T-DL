from keras.applications.xception import (
    Xception, preprocess_input, decode_predictions
)
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os
import argparse
from util import *
import config
import pickle as pkl
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default = os.path.join(config.result_root, config.model_name))
parser.add_argument('--model', default =  "model_fine_final.h5")
parser.add_argument('--valid_image_path', default = "../geo/valid/")
parser.add_argument('--output_file', default = "output_new.txt")
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--convertToLL', action="store_true", default=False)

def main(args):
    
    model = load_model(os.path.join(args.model_path, args.model))
    if config.loss_type == "classification":
        grid = pkl.load(open(os.path.join(args.model_path,'grid.pkl')))
    f = open(args.output_file,'w')

    num_samples = len(os.listdir(args.valid_image_path))

    valid_image_paths = []
    image_names = []
    for image_name in os.listdir(args.valid_image_path):
        image_names.append(image_name)
        valid_image_paths.append(os.path.join(args.valid_image_path,image_name))

    
    for i in range(0, num_samples, args.batch_size):
        batch_inputs = list(map(
                lambda x: image.load_img(x, target_size=(261,150)),
                valid_image_paths[i:i+args.batch_size]
            ))
        batch_inputs = np.array(list(map(
                lambda x: image.img_to_array(x),
                batch_inputs
            )))

        batch_inputs = preprocess_input(batch_inputs)
        pred = model.predict(batch_inputs)

        print(pred)
        
        if config.loss_type == "regression":
            if args.convertToLL:
                pred = XYToCoordinate(pred)

        for j in range(args.batch_size):
            if config.loss_type == "regression":
                f.write("%s\t%f\t%f\n" %(image_names[i+j], pred[j][0], pred[j][1]))
            elif config.loss_type == "classification":
                lat_index = pred[j]/len(grid[1])
                lon_index = pred[j]%len(grid[1])
                f.write("%s\t%f\t%f\n" %(image_names[i+j], lat_index, lon_index))
    f.close()
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
