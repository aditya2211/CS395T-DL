from keras.applications.xception import Xception, preprocess_input
from keras.applications.vgg19 import VGG19
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.losses import categorical_crossentropy
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import math
import numpy as np
import os
import imghdr
import pickle as pkl
import datetime
from config import *

current_directory = os.path.dirname(os.path.abspath(__file__))


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

    # convert to numpy array
    train_input_paths = np.array(train_input_paths)
    val_input_paths = np.array(val_input_paths)
  
    print("Training on %d images and labels" % (len(train_input_paths)))
    print("Validation on %d images and labels" % (len(val_input_paths)))

    # create a directory where results will be saved (if necessary)
    if os.path.exists(args.result_root) == False:
        os.makedirs(args.result_root)

    # ====================================================
    # Build a custom Xception
    # ====================================================
    # instantiate pre-trained Xception model
    # the default input shape is (299, 299, 3)
    # NOTE: the top classifier is not included
    if args.model_name == "Xception":
        base_model = Xception(include_top=False, weights='imagenet', input_shape=(261,150,3))
    elif args.model_name == "VGG19":
        base_model = VGG19(include_top=False, weights='imagenet', input_shape=(261,150,3))

    # create a custom top classifier
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2)(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)

    # ====================================================
    # Train only the top classifier
    # ====================================================
    # freeze the body layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile model
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(lr=args.lr_pre),
    )

    # train
    hist_pre = model.fit_generator(
        generator=generate_from_paths_and_labels(
            input_paths=train_input_paths,
            labels=train_labels,
            batch_size=args.batch_size_pre
        ),
        steps_per_epoch=math.ceil(len(train_input_paths) / args.batch_size_pre),
        epochs=args.epochs_pre,
        validation_data=generate_from_paths_and_labels(
            input_paths=val_input_paths,
            labels=val_labels,
            batch_size=args.batch_size_pre
        ),
        validation_steps=math.ceil(len(val_input_paths) / args.batch_size_pre),
        verbose=1,
        callbacks=[
            ModelCheckpoint(
                filepath=os.path.join(args.result_root, 'model_pre_ep{epoch}_{val_loss:.3f}.h5'),
                period=args.snapshot_period_pre,
            ),
        ],
    )
    model.save(os.path.join(args.result_root, 'model_pre_final.h5'))

    # ====================================================
    # Train the whole model
    # ====================================================
    # set all the layers to be trainable
    for layer in model.layers:
        layer.trainable = True

    # recompile
    model.compile(
        optimizer=Adam(lr=args.lr_fine),
        loss='mean_squared_error',)

    # train
    hist_fine = model.fit_generator(
        generator=generate_from_paths_and_labels(
            input_paths=train_input_paths,
            labels=train_labels,
            batch_size=args.batch_size_fine
        ),
        steps_per_epoch=math.ceil(len(train_input_paths) / args.batch_size_fine),
        epochs=args.epochs_fine,
        validation_data=generate_from_paths_and_labels(
            input_paths=val_input_paths,
            labels=val_labels,
            batch_size=args.batch_size_fine
        ),
        validation_steps=math.ceil(len(val_input_paths) / args.batch_size_fine),
        verbose=1,
        callbacks=[
            ModelCheckpoint(
                filepath=os.path.join(args.result_root, 'model_fine_ep{epoch}_agupta{val_loss:.3f}.h5'),
                period=args.snapshot_period_fine,
            ),
        ],
    )
    model.save(os.path.join(args.result_root, 'model_fine_final.h5'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
