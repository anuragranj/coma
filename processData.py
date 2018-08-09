import argparse
import os
from facemesh import *
parser = argparse.ArgumentParser(description='Preprocessing data for Convolutional Mesh Autoencoders',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', dest='data', type=str, required=True,
                    help='path to the data directory')
parser.add_argument('--save_path', dest='save_path', type=str, default='data',
                    help='path where processed data will be saved')

def main():
    args = parser.parse_args()
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("Preprocessing Slice Time Data")
    generateSlicedTimeDataSet(args.data, save_path)

    print("Preprocessing Expression Cross Validation")
    generateExpressionDataSet(args.data, save_path)

if __name__ == '__main__':
    main()
