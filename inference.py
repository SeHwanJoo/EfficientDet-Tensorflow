import cv2
import json
import numpy as np
import os
import time
import glob
import tensorflow as tf
import argparse
import sys

from datetime import date
from model import efficientdet
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes



def check_args(parsed_args):
    if parsed_args.gpu and parsed_args.batch_size < len(parsed_args.gpu.split(',')):
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             len(parsed_args.gpu.split(
                                                                                                 ','))))

    return parsed_args


def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    parser.add_argument('--model-path', help='Inference from a model path.')
    parser.add_argument('--weighted-bifpn', help='Use weighted BiFPN', action='store_true', default=False)

    parser.add_argument('--phi', help='Hyper parameter phi', default=0, type=int, choices=(0, 1, 2, 3, 4, 5, 6))
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).', type=int, default=4)
    parser.add_argument('--gpu', help='score threshold', default=0.5, type=float)

    print(vars(parser.parse_args(args)))
    return check_args(parser.parse_args(args))


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    # optionally choose specific GPU
    if args.gpu:
        # GPU settings
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # use args.gpu
            try:
                print('start with GPU {}'.format(args.gpu))
                tf.config.experimental.set_visible_devices(gpus[int(args.gpu)], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[int(args.gpu)], True)
            except RuntimeError as e:
                print(e)

    model_path = args.model_path
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[args.phi]
    # pascal classes
    classes = {value['id'] - 1: value['name'] for value in json.load(open('pascal.json', 'r')).values()}
    num_classes = 20
    colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
    _, model = efficientdet(phi=args.phi,
                            weighted_bifpn=args.weighted_bifpn,
                            num_classes=num_classes,
                            score_threshold=args.score_threshold)
    model.load_weights(model_path, by_name=True)
    sum = 0
    i = 0
    path = os.path.join('detection', model_path.split('/')[-1][:-3])
    os.makedirs(path, exist_ok=True)
    for image_path in glob.glob(args.pascal_path):
        image = cv2.imread(image_path)
        src_image = image.copy()
        # BGR -> RGB
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale = preprocess_image(image, image_size=image_size)
        image = [np.expand_dims(image, axis=0)]
        # run network
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(image)
        sum += time.time() - start
        print(time.time() - start)
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        if i == 0:
            sum = 0
        boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

        # select indices which have a score above the threshold
        indices = np.where(scores[:] > args.score_threshold)[0]

        # select those detections
        boxes = boxes[indices]
        labels = labels[indices]

        draw_boxes(src_image, boxes, scores, labels, colors, classes)
        filename = os.path.join(path, image_path.split('/')[-1])
        cv2.imwrite(filename, src_image)
        i += 1

    print('FPS : {}'.format(i / sum))


if __name__ == '__main__':
    main()
