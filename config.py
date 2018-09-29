import argparse
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default= '/scratch/cluster/tanya/geo/')
parser.add_argument('--result_root', default='/scratch/cluster/tanya/geo/geo_xception_results')
parser.add_argument('--model_name', default= 'VGG19')
parser.add_argument('--epochs_pre', type=int, default=5)
parser.add_argument('--epochs_fine', type=int, default=5)
parser.add_argument('--batch_size_pre', type=int, default=32)
parser.add_argument('--batch_size_fine', type=int, default=16)
parser.add_argument('--lr_pre', type=float, default=1e-3)
parser.add_argument('--lr_fine', type=float, default=1e-4)
parser.add_argument('--snapshot_period_pre', type=int, default=1)
parser.add_argument('--snapshot_period_fine', type=int, default=1)