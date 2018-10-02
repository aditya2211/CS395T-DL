from __future__ import print_function
from os import path
from math import sin, cos, atan2, sqrt, pi
import numpy as np
from util import *
def numToRadians(x):
  return x / 180.0 * pi

# Calculate distance (km) between Latitude/Longitude points
# Reference: http://www.movable-type.co.uk/scripts/latlong.html
EARTH_RADIUS = 6371
def dist(lat1, lon1, lat2, lon2):
  lat1 = numToRadians(lat1)
  lon1 = numToRadians(lon1)
  lat2 = numToRadians(lat2)
  lon2 = numToRadians(lon2)

  dlat = lat2 - lat1
  dlon = lon2 - lon1

  a = sin(dlat / 2.0) * sin(dlat / 2.0) + cos(lat1) * cos(lat2) * sin(dlon / 2.0) * sin(dlon / 2.0)
  c = 2 * atan2(sqrt(a), sqrt(1-a))

  d = EARTH_RADIUS * c
  return d

def read_file(file):

  f = open(file,'r')
  image_to_points = {}
  for line in f:
    tokens = line.strip().split('\t')
    image_to_points[tokens[0]] = [float(tokens[1]), float(tokens[2])]

  return image_to_points
def eval_streetview_results(pred_file, gold_labels):
  
  

  pred_points = read_file(pred_file)
  gold_points =  read_file(gold_labels)
  assert len(pred_points)==len(gold_points), "Number of predicted points does not equal number of gold points."
  total_count = len(gold_points)
  l1_dist = 0.0
  print( "Total validation data", total_count )
  count=0
  for image_name in gold_points.keys():
    pred_lat, pred_lon =  pred_points[image_name][0], pred_points[image_name][1]
    truth_lat, truth_lon = gold_points[image_name][0], gold_points[image_name][1]
    l1_dist += dist(pred_lat, pred_lon, truth_lat, truth_lon)
  l1_dist /= total_count
  print( "L1 distance", l1_dist )
  return l1_dist

def eval_baseline(train_file, gold_file):
  train_list = [n.strip().split('\t') for n in open(train_file,'r')]

    # Get all the labels
  coord = np.array([(float(y[1]), float(y[2])) for y in train_list])
  xy = coordinateToXY(coord)
  med = np.median(xy, axis=0, keepdims=True)
  med = np.squeeze(XYToCoordinate(med))
  print("Median of the train data is:{}".format(med))
  gold_points = read_file(gold_file)
  total_count = len(gold_points)
  l1_dist = 0.0
  count=0
  for image_name in gold_points.keys():
    truth_lat, truth_lon = gold_points[image_name][0], gold_points[image_name][1]
    l1_dist += dist(med[0], med[1], truth_lat, truth_lon)
  l1_dist /= total_count
  print( "L1 distance", l1_dist )

  return l1_dist

 
def eval_bound(gold_file):
  gold_points = read_file(gold_file)
  total_count = len(gold_points)
  l1_dist = 0.0
  count=0

  a = np.sqrt(2.0*.5*.5)
  for image_name in gold_points.keys():
    truth_lat, truth_lon = gold_points[image_name][0], gold_points[image_name][1]
    l1_dist += dist(truth_lat+a, truth_lon+a, truth_lat, truth_lon)
  l1_dist /= total_count
  print( "L1 distance", l1_dist )

  return l1_dist

 



if __name__ == "__main__":
  import importlib
  from argparse import ArgumentParser
  parser = ArgumentParser("Evaluate a model on the validation set")
  parser.add_argument("pred_file") 
  parser.add_argument("gold_file")
  parser.add_argument("train_file")
  args = parser.parse_args()
  eval_bound(args.gold_file)
  eval_baseline(args.train_file, args.gold_file)
  eval_streetview_results(args.pred_file, args.gold_file)





