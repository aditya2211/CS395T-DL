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
from xception import *
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default = os.path.join(config.result_root, config.model_name))
parser.add_argument('--model', default =  "model_fine_final.h5")
parser.add_argument('--valid_image_path', default = "../../geo/valid/")
parser.add_argument('--output_file', default = "output_"+ config.model_name +"_29_20.txt")
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--convertToLL', action="store_true", default=False)

def main(args):
    
    
    if config.loss_type == "classification":
        if config.grid_type == "recursive":
            model_path_comp = os.path.join(args.model_path, "recursive" + str(config.max_x_distance) + "_" + str(config.max_y_distance) + "_" + str(config.max_labels))
            grid = pkl.load(open(os.path.join(model_path_comp, 'grid.pkl'), "rb"))
            gridmapping = pkl.load(open(os.path.join(model_path_comp, 'grid_mapping.pkl'), "rb"))
            model_path_comp = os.path.join(model_path_comp, args.model)
        else:
            model_path_comp = os.path.join(args.model_path,  args.model)
            grid = pkl.load(open(os.path.join(args.model_path, 'grid.pkl'), "rb"))
               
    f = open(args.output_file,'w')
    model = load_model(model_path_comp)
    num_samples = len(os.listdir(args.valid_image_path))

    valid_image_paths = []
    image_names = []
    for image_name in os.listdir(args.valid_image_path):
        image_names.append(image_name)
        valid_image_paths.append(os.path.join(args.valid_image_path,image_name))

    print("reached here")
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
        
        if config.loss_type == "regression":
            if args.convertToLL:
                pred = XYToCoordinate(pred)

        for j in range(args.batch_size):
            if config.loss_type == "regression":
                f.write("%s\t%f\t%f\n" %(image_names[i+j], pred[j][0], pred[j][1]))
            elif config.loss_type == "classification":
                if config.grid_type == "recursive":
                    print(np.argmax(pred, axis = 1))
                    continue
                    grid_path = grid_mapping[pred].split()
                else:
                    lat_index = int(np.argmax(pred[j])/len(grid[1]))
                    lon_index = np.argmax(pred[j])%len(grid[1])
                    f.write("%s\t%f\t%f\n" %(image_names[i+j], grid[0][lat_index], grid[1][lon_index]))
    f.close()
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
