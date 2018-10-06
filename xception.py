from keras.applications.xception import Xception, preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.losses import categorical_crossentropy
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils.np_utils import to_categorical
from time import time
import math
import numpy as np
import os
import argparse
import imghdr
import pickle as pkl
import datetime
import config 

current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default= config.dataset_root)
parser.add_argument('--result_root', default= config.result_root)
parser.add_argument('--model_name', default= config.model_name)
parser.add_argument('--logging_root', default = config.logging_root)
parser.add_argument('--epochs_pre', type=int, default = 5)
parser.add_argument('--epochs_fine', type=int, default = 10)
parser.add_argument('--batch_size_pre', type=int, default=32)
parser.add_argument('--batch_size_fine', type=int, default=16)
parser.add_argument('--lr_pre', type=float, default=1e-3)
parser.add_argument('--lr_fine', type=float, default=1e-4)
parser.add_argument('--snapshot_period_pre', type=int, default=1)
parser.add_argument('--snapshot_period_fine', type=int, default=1)

def generate_from_paths_and_labels(input_paths, labels, batch_size, input_size=(261,150)):
    labels = np.array(labels)
    num_samples = len(input_paths)
    while 1:
        perm = np.random.permutation(num_samples)
        input_paths = input_paths[perm]
        labels = labels[perm]
        for i in range(0, num_samples, batch_size):
            inputs = list(map(
                lambda x: image.load_img(x, target_size=input_size),
                input_paths[i:i+batch_size]
            ))
            inputs = np.array(list(map(
                lambda x: image.img_to_array(x),
                inputs
            )))
            inputs = preprocess_input(inputs)
            yield (inputs, labels[i:i+batch_size])

def create_grid(coordinates):
    lat_min = np.min([x[0] for x in coordinates])
    lat_max = np.max([x[0] for x in coordinates])
    lon_min = np.min([x[1] for x in coordinates])
    lon_max = np.max([x[1] for x in coordinates])

    x_coords = np.arange(lat_min, lat_max, 0.5)
    y_coords = np.arange(lon_min, lon_max, 0.5)
    
    return [x_coords, y_coords]

def gridify(labels, grid):

    grid_labels = []
    for l in labels:
        lat_index  = np.argmin((l[0]-grid[0])**2)
        long_index = np.argmin((l[1]-grid[1])**2)
        grid_labels.append(lat_index * len(grid[1]) + long_index)
    return grid_labels


def get_membership_point(p, x_mid, y_mid):
    if p[0] < x_mid:
        a = 0
    else:
        a = 1
    if p[1] < y_mid:
        b = 0
    else:
        b = 2

    return a+b

class grid:
    def __init__(self, x1, x2, y1, y2, labels, path):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.labels = labels
        self.children = []
        self.path = path

    def __repr__(self):
        string = "X_min_coordinate:%f\nX_max_coordinate:%f\nY_min_coordinate:%f\nY_min_coordinate:%f\n" % (self.x1, self.x2, self.y1, self.y2)
        string += "Path: %s\n" % (self.path)
        if len(self.labels) >= 2:
            string += "Points:[%f, %f],[%f,%f]" % (self.labels[0][0], self.labels[0][1], self.labels[1][0], self.labels[1][1])
        return string

    def create_children(self):
        x_mid = float(self.x1 + self.x2)/float(2)
        y_mid = float(self.y1 + self.y2)/float(2)
        #print "creating children"

        mem = {0: [], 1: [], 2: [], 3: []}
        for p in self.labels:
            membership = get_membership_point(p, x_mid, y_mid)
            mem[membership].append(p)

        g0 = grid(self.x1, x_mid, self.y1, y_mid, mem[0],  self.path + "_0")
        g1 = grid(x_mid, self.x2, self.y1, y_mid, mem[1],  self.path + "_1")
        g2 = grid(self.x1, x_mid, y_mid, self.y2, mem[2],  self.path + "_2")
        g3 = grid(x_mid, self.x2, y_mid, self.y2, mem[3],  self.path + "_3")

        self.children.append(g0)
        self.children.append(g1)
        self.children.append(g2)
        self.children.append(g3)
        
        return self.children

    def check_conditions(self, max_x, max_y, max_label):
        if len(self.labels) == 0:
            return True
        if abs(self.x1 - self.x2) > max_x:
            return False
        elif abs(self.y1 - self.y2) > max_y:
            return False
        elif len(self.labels) > max_label:
            return False
        else:
            return True



def grid_recursive(grid, max_x, max_y, max_label):
    if not grid.check_conditions(max_x, max_y, max_label):
        children = grid.create_children()
        for child in grid.children:
            grid_recursive(child, max_x, max_y, max_label)


def create_grid_recursive(points, max_x, max_y, max_label):
    lat_min = np.min([x[0] for x in points])
    lat_max = np.max([x[0] for x in points])
    lon_min = np.min([x[1] for x in points])
    lon_max = np.max([x[1] for x in points])

    grid_r = grid(lat_min, lat_max, lon_min, lon_max, points, "0")
    grid_recursive(grid_r, max_x, max_y, max_label)
    return grid_r

def gridify_recursive_point(grid, p):

    if len(grid.children) == 0:
        return(grid.path)

    else:
        x_mid = float(grid.x1 + grid.x2)/float(2)
        y_mid = float(grid.y1 + grid.y2)/float(2)

        m = get_membership_point(p, x_mid, y_mid)
        #print m
        return(gridify_recursive_point(grid.children[m], p))


def gridify_recursive(grid, labels):
    y_labels = []
    for p in labels:
        y_labels.append(gridify_recursive_point(grid, p))
    return y_labels

def main(args):

    # ====================================================
    # Preparation
    # ====================================================
    # parameters
    epochs = args.epochs_pre + args.epochs_fine
    args.dataset_root = os.path.expanduser(args.dataset_root)
    args.result_root = os.path.expanduser(args.result_root)

    # make input_paths and labels


    train_input_paths, train_labels, values = [], [], {}
    for label in open(os.path.join(args.dataset_root, 'geo_train.txt'), 'r'):
        gold_tags = label.rstrip().split('\t')
        values[gold_tags[0]] = [float(gold_tags[1]), float(gold_tags[2])]

    for image_name in os.listdir(os.path.join(args.dataset_root, 'train')):
        path = os.path.join(os.path.join(args.dataset_root, 'train'), image_name)
        if imghdr.what(path) == None:
            continue
        train_labels.append(values[image_name])
        train_input_paths.append(path)

    val_input_paths, val_labels, values = [], [], {}
    for label in open(os.path.join(args.dataset_root, 'geo_valid.txt'), 'r'):
        gold_tags = label.rstrip().split('\t')
        values[gold_tags[0]] = [float(gold_tags[1]), float(gold_tags[2])]

    for image_name in os.listdir(os.path.join(args.dataset_root, 'valid')):
        path = os.path.join(os.path.join(args.dataset_root, 'valid'), image_name)
        if imghdr.what(path) == None:
            # this is not an image file
            continue
        val_labels.append(values[image_name])
        val_input_paths.append(path)
    # convert to one-hot-vector format

    if config.loss_type == "classification":
        if config.grid_type == "recursive":
            max_labels = config.max_labels
            max_x_distance = config.max_x_distance
            max_y_distance = config.max_y_distance
            grid = create_grid_recursive(train_labels, max_x_distance, max_y_distance, max_labels)
            train_grid_labels_paths = gridify_recursive(grid, train_labels)
            val_grid_labels_paths = gridify_recursive(grid, val_labels)
            mapping = dict(enumerate(set(train_grid_labels_paths + val_grid_labels_paths)))
            inv_mapping = {v: k for k, v in mapping.items()}
            train_grid_labels = [inv_mapping[x] for x in train_grid_labels_paths]
            val_grid_labels = [inv_mapping[x] for x in val_grid_labels_paths]
        else:
            grid = create_grid(train_labels)
            train_grid_labels = gridify(train_labels, grid)
            val_grid_labels = gridify(val_labels, grid)

    print({x:train_grid_labels.count(x) for x in set(train_grid_labels)})
    # convert to numpy array
    train_input_paths = np.array(train_input_paths)
    val_input_paths = np.array(val_input_paths)
  
    print("Training on %d images and labels" % (len(train_input_paths)))
    print("Validation on %d images and labels" % (len(val_input_paths)))

    exit()
    # create a directory where results will be saved (if necessary)
    if os.path.exists(args.result_root) == False:
        os.makedirs(args.result_root)

    if os.path.exists(os.path.join(args.result_root, args.model_name)) == False:
        os.makedirs(os.path.join(args.result_root, args.model_name))

    if config.loss_type == "classification":
        if config.grid_type == "recursive":
            model_path = os.path.join(args.result_root, args.model_name, "recursive" + str(max_x_distance) + "_" + str(max_y_distance) + "_" + str(max_labels))
        else:
            model_path = os.path.join(args.result_root, args.model_name, str(len(grid[0])) + "_" + str(len(grid[1])))

        if os.path.exists(model_path) == False:
            os.makedirs(model_path)
        
        with open(os.path.join(model_path, 'grid.pkl'), 'wb') as gridpickle:
            pkl.dump(grid, gridpickle)

        with open(os.path.join(model_path, 'grid_mapping.pkl'), 'wb') as gridmappingpickle:
            pkl.dump(mapping, gridmappingpickle)
    # ====================================================
    # Build a custom Xception
    # ====================================================
    # instantiate pre-trained Xception model
    # the default input shape is (299, 299, 3)
    # NOTE: the top classifier is not included
    base_model = eval(args.model_name)(include_top=False, weights='imagenet', input_shape=(261,150,3))


    # create a custom top classifier
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    if config.loss_type == "regression":
        predictions = Dense(2)(x)
    elif config.loss_type == "classification":
        #predictions = Dense(len(grid[0])*len(grid[1]), activation='softmax')(x)
        predictions = Dense(len(mapping), activation='softmax')(x)
    
    model = Model(inputs=base_model.inputs, outputs=predictions)

    # ====================================================
    # Train only the top classifier
    # ====================================================
    # freeze the body layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile model
    if config.loss_type == "regression":
        model.compile(
            loss='mean_squared_error',
            optimizer=Adam(lr=args.lr_pre),
        )
        train_l = train_labels
        val_l = val_labels
    elif config.loss_type == "classification":
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(lr=args.lr_pre),
            metrics=['acc']
        )
        train_l = train_grid_labels
        val_l = val_grid_labels


    tensorboard = TensorBoard(log_dir = "{}/{}".format(args.logging_root,time()))
    # train
    hist_pre = model.fit_generator(
        generator=generate_from_paths_and_labels(
            input_paths=train_input_paths,
            labels=train_l,
            batch_size=args.batch_size_pre
        ),
        steps_per_epoch=math.ceil(len(train_input_paths) / args.batch_size_pre),
        epochs=args.epochs_pre,
        validation_data=generate_from_paths_and_labels(
            input_paths=val_input_paths,
            labels=val_l,
            batch_size=args.batch_size_pre
        ),
        validation_steps=math.ceil(len(val_input_paths) / args.batch_size_pre),
        verbose=1,
        callbacks=[tensorboard,
            ModelCheckpoint(
                filepath=os.path.join(model_path, 'model_pre_ep{epoch}_{val_loss:.3f}.h5'),
                period=args.snapshot_period_pre,
            ),
        ],
    )
    model.save(os.path.join(model_path, 'model_pre_final.h5'))

    # ====================================================
    # Train the whole model
    # ====================================================
    # set all the layers to be trainable
    for layer in model.layers:
        layer.trainable = True

    # recompile
    if config.loss_type == "regression":
        model.compile(
            loss='mean_squared_error',
            optimizer=Adam(lr=args.lr_pre),
        )
    elif config.loss_type == "classification":
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(lr=args.lr_pre),
            metrics=['acc']
        )

    # train
    hist_fine = model.fit_generator(
        generator=generate_from_paths_and_labels(
            input_paths=train_input_paths,
            labels=train_l,
            batch_size=args.batch_size_fine
        ),
        steps_per_epoch=math.ceil(len(train_input_paths) / args.batch_size_fine),
        epochs=args.epochs_fine,
        validation_data=generate_from_paths_and_labels(
            input_paths=val_input_paths,
            labels=val_l,
            batch_size=args.batch_size_fine
        ),
        validation_steps=math.ceil(len(val_input_paths) / args.batch_size_fine),
        verbose=1,
        callbacks=[tensorboard,
            ModelCheckpoint(
                filepath=os.path.join(model_path, 'model_fine_ep{epoch}_{val_loss:.3f}.h5'),
                period=args.snapshot_period_fine,
            ),
        ],
    )


    model.save(os.path.join(model_path, 'model_fine_final.h5'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
