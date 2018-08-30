import numpy as np
import os
import shutil
import errno
import glob
import tensorflow as tf
from PIL import Image


# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util


def reshape_image_into_numpy_array(pil_image):
	"""

	:param pil_image: a pillow image
	:return: a reshaped numpy image ready for inference
	"""
	(im_width, im_height) = pil_image.size
	# print(im_width)
	zeros = np.zeros((im_height, im_width, 2))
	# return np.array(pil_image.getdata()).reshape(
	# 	(im_height, im_width, 1)).astype(np.uint8)
	np_array = np.array(pil_image.getdata()).reshape((im_height, im_width, 1)).astype(np.uint8)
	np_array = np.concatenate((np_array, np_array, np_array), axis=2)
	return np_array


def do_inference_with_graph(pil_image, inference_graph_path):
	"""
	It takes a pillow image and looks for tables inside

	:param pil_image: Pillow image
	:param inference_graph_path:
	:return: (boxes, scores), two lists with all the boxes and their likelihood scores
	"""

	detection_graph = tf.Graph()

	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(inference_graph_path, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			pil_image.convert(mode='RGB')
			# the array based representation of the image will be used later in order to prepare the
			# result image with boxes and labels on it.
			image_np = reshape_image_into_numpy_array(pil_image)
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

			return boxes[0], scores[0]


def check_if_intersected(coord_a, coord_b):
	"""
	Check if the rectangular b is not intersecated with a
	:param coord_a: dict with {y_min, x_min, y_max, x_max}
	:param coord_b: same as coord_a
	:return: true if inside, false if outside
	"""
	return coord_a['x_max'] > coord_b['x_min'] and coord_a['x_min'] < coord_b['x_max'] and coord_a['y_max'] > coord_b['y_min'] and coord_a['y_min'] < coord_b['x_max']


def keep_best_boxes(boxes, scores, max_num_boxes=5, min_score=0.8):
	"""
	return a list of the max_num_boxes not overlapping boxes found in inference
	boxes are: box[0]=ymin, box[1]=xmin, box[2]=ymax, box[3]=xmax

	:param boxes: list of boxes found in inference
	:param scores: likelihood of the boxes
	:param max_num_boxes: max num of boxes to be saved
	:param min_score: min box score to check
	:return: list of the best not overlapping boxes
	"""

	kept_scores = []
	kept_boxes = []  # always keep the firs box, which is the best one.
	num_boxes = 0
	i = 0
	if scores[0] > min_score:
		kept_boxes.append(boxes[0])
		kept_scores.append(scores[0])
		num_boxes += 1
		i += 1
		for b in boxes[1:]:
			if num_boxes < max_num_boxes and scores[i] > min_score:
				flag = True
				coord_b = {
					'y_min': b[0],
					'x_min': b[1],
					'y_max': b[2],
					'x_max': b[3]
				}
				for kb in kept_boxes:
					# checks if box score is high enough
					coord_kb = {
						'y_min': kb[0],
						'x_min': kb[1],
						'y_max': kb[2],
						'x_max': kb[3]
					}
					flag = check_if_intersected(
						coord_a=coord_b,
						coord_b=coord_kb
					)

				if flag:
					kept_boxes.append(b)
					num_boxes += 1
					kept_scores.append(scores[i])
				i += 1
			else:
				break

		# print(str(i) + '\tbox(es) found\nBest score(s):\t', *kept_scores, sep='\n')
		print(str(i) + '\tbox(es) found')
		for box in kept_boxes:
			print('Box\t')
			print(box)
		for score in kept_scores:
			print('Score:\t')
			print(score)

	else:
		kept_boxes = []
		print('No boxes found\nBest score:\t' + str(scores[0]))

	return kept_boxes


def crop_image(pil_image, boxes):
	cropped_images = []
	for box in boxes:
		cropped_images.append(pil_image.crop(tuple(box)))
	return cropped_images


def crop_wide(pil_image, boxes):
	cropped_tables = []
	segments = [0]  # adding position 0 to simplify anti-crop text later
	height_of_crops = 0
	if not boxes == []:
		(im_width, im_height) = pil_image.size

		for box in boxes:
			cropped_tables.append(pil_image.crop(tuple((0, int(box[0]), im_width, int(box[2])))))
			segments.append(int(box[0]))
			segments.append(int(box[2]))
			height_of_crops += (int(box[2]) - int(box[0]))

		# sorts all segments for their
		segments.append(im_height)  # adding last position to simplyfy anti-crop text later
		segments.sort()

		# create new image with new dimension
		new_image = Image.new('L', (im_width, im_height - height_of_crops))
		start_position = 0
		# cutting image in anti-boxes position
		for i in range(len(segments)):  # segments will always be even
			if not i % 2 and i < len(segments) - 1:  # takes only even positions
				if i != 0:
					start_position += segments[i-1] - segments[i-2]
				new_image.paste(pil_image.crop(tuple((0, segments[i], im_width, segments[i+1]))), (0, start_position))
		cropped_text = new_image
	else:
		cropped_text = pil_image

	return cropped_tables, cropped_text


def extract_tables_and_text(image_path, inference_graph_path):
	"""
	Extracts tables and text from image_path using inference_graph_path

	:param image_path:
	:param inference_graph_path:
	:return: (cropped_tables, cropped_text), list of table pillow images and a text image
	"""
	pil_image = Image.open(image_path)
	(im_width, im_height) = pil_image.size
	boxes, scores = do_inference_with_graph(pil_image, inference_graph_path)
	best_boxes = keep_best_boxes(
		boxes=boxes,
		scores=scores,
		max_num_boxes=5,
		min_score=0.4
	)

	# create coordinates based on image dimension
	for box in best_boxes:
		box[0] = int(box[0]*im_height)
		box[2] = int(box[2]*im_height)
		box[1] = int(box[1]*im_width)
		box[3] = int(box[3]*im_width)

	(cropped_tables, cropped_text) = crop_wide(pil_image, best_boxes)
	return cropped_tables, cropped_text


def clear_and_create_folders(file_name):
	"""
	Clear any existing table/file_name and text/file_name folder for creating new images

	:param file_name:
	:return: None
	"""
	if not os.path.isdir('tables/'):
		# creates folder for table images per page
		try:
			os.mkdir('tables')
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise

	# creates folder for text images per page
	if not os.path.isdir('text/'):
		try:
			os.mkdir('text')
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise

	if os.path.isdir('tables/' + str(file_name)):
		shutil.rmtree('tables/' + str(file_name), ignore_errors=True)
	if os.path.isdir('text/' + str(file_name)):
		shutil.rmtree('text/' + str(file_name), ignore_errors=True)

	try:
		os.mkdir('tables/' + str(file_name))
	except OSError as exc:  # Guard against race condition
		if exc.errno != errno.EEXIST:
			raise

	try:
		os.mkdir('text/' + str(file_name))
	except OSError as exc:  # Guard against race condition
		if exc.errno != errno.EEXIST:
			raise


def write_crops(file_name, cropped_tables, cropped_text):
	"""
	Writes table and text images under table and text folder

	:param file_name:
	:param cropped_tables:
	:param cropped_text:
	:return: None
	"""

	i = 0
	for ct in cropped_tables:
		new_file_path = 'tables/' + str(file_name) + '/table_' + str(i) + '.jpeg'
		ct.save(new_file_path)
		i += 1

	new_file_path = 'text/' + str(file_name) + '/text' + '.jpeg'
	cropped_text.save(new_file_path)


def main():
	for file in glob.iglob(PATH_TO_IMAGES + '/**/*.jpeg', recursive=True):
		# if file.endswith(".jpeg"):
		path_to_image = os.path.join(file)
		file_name = os.path.splitext(path_to_image)[0] \
			.split("\\")[-1]
		print('Now processing: ' + str(file_name))
		clear_and_create_folders(file_name=file_name)
		cropped_tables, cropped_text = extract_tables_and_text(path_to_image, PATH_TO_CKPT)
		write_crops(
			file_name=file_name,
			cropped_tables=cropped_tables,
			cropped_text=cropped_text
		)


PATH_TO_CKPT = '../TableTrainNet/data/frozen_inference_graph_momentum.pb'
PATH_TO_IMAGES = './PDFs/'
main()

