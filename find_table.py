import numpy as np
import img2pdf  # to export table images as pdf, which will be later recognized
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import copy
import base64

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")

# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = os.path.join('data', 'frozen_inference_graph.pb')
PATH_TO_CKPT = '../TableTrainTest/data/frozen_inference_graph_momentum.pb'

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')
PATH_TO_LABELS = '../TableTrainTest/data/object-detection.pbtxt'

NUM_CLASSES = 1

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

# load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,	use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def from_png_to_bmp(png_path):
	# convert a .png image file to a .bmp image file using PIL

	file_in = png_path
	img = Image.open(file_in)

	file_out = png_path + '.bmp'
	print
	len(img.split())  # test
	if len(img.split()) == 4:
		# prevent IOError: cannot write mode RGBA as BMP
		r, g, b, a = img.split()
		img = Image.merge("RGB", (r, g, b))
		img.save(file_out)
	else:
		img.save(file_out)


def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	# print(im_width)
	return np.array(image.getdata()).reshape(
		(im_height, im_width, 3)).astype(np.uint8)


# detection
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = 'data'
PATH_TO_TEST_IMAGES_DIR = '../TableTrainTest/data/'
# TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.png'.format(i)) for i in range(1, 2)]
# TEST_IMAGE_PATHS = os.path.join('data', 'image1.png')
TEST_IMAGE_PATHS = ['../TableTrainTest/data/image1.png']
from_png_to_bmp(TEST_IMAGE_PATHS[0])
TEST_IMAGE_PATHS = ['../TableTrainTest/data/image1.png.bmp']

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		for image_path in TEST_IMAGE_PATHS:
			image = Image.open(image_path)
			image.convert()
			# the array based representation of the image will be used later in order to prepare the
			# result image with boxes and labels on it.
			image_np = load_image_into_numpy_array(image)
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(image_np, axis=0)
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			# Each box represents a part of the image where a particular object was detected.
			boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			# Each score represent how level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			scores = detection_graph.get_tensor_by_name('detection_scores:0')
			classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			# Actual detection.
			(boxes, scores, classes, num_detections) = sess.run(
				[boxes, scores, classes, num_detections],
				feed_dict={image_tensor: image_np_expanded})

			print('boxes: ', boxes[0][0])
			print('scores', scores)
			print('classes', classes)
			print('num_detections', num_detections)

			# print(boxes[0][0][0], boxes[0][0][1], boxes[0][0][2], boxes[0][0][3])
			image1 = copy.deepcopy(image)
			image2 = copy.deepcopy(image)
			vis_util.visualize_boxes_and_labels_on_image_array(
				image_np,
				boxes[0],
				classes[0],
				scores[0],
				category_index,
				agnostic_mode=True,
				skip_labels=True
			)
			vis_util.draw_bounding_box_on_image(
				image,
				boxes[0][0][0],
				boxes[0][0][1],
				boxes[0][0][2],
				boxes[0][0][3]
			)
			vis_util.draw_bounding_boxes_on_image(
				image1,
				boxes[0]
			)
			image.save('../TableTrainTest/data/test1_box_momentum.bmp')
			image1.save('../TableTrainTest/data/test1_boxes_momentum.bmp')


