from PIL import Image
from alyn import deskew
import os
import errno
import numpy as np
import shutil
import cv2
from pdf2image import convert_from_path
import logging
from logger import return_handler
from costants import extraction_dpi

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_name = 'pipeline-' + str(0) + '.log'
logger.addHandler(return_handler(file_name))
# logger.info('Hello baby')

# import argparse
#
# parser = argparse.ArgumentParser(description='Path to input pdf file')
# parser.add_argument('--pdf_path', dest='pdf_path', help='the complete path to your pdf file')
# args = parser.parse_args()


def clear_and_create_create_temp_folders(file_name, temp_path='temp'):
	logger.info('Clear and create temp file for images from pdf')
	try:
		os.mkdir(temp_path)
		logger.info('Folder created successfully')
	except OSError as exc:  # Guard against race condition
		if exc.errno != errno.EEXIST:
			logger.warning('Parent folder was not created correctly. Probably already present')
			raise

	if os.path.isdir(os.path.join(temp_path, str(file_name))):
		logger.info('Deleting not empty temp folder')
		shutil.rmtree(os.path.join(temp_path, str(file_name)), ignore_errors=True)
	try:
		os.mkdir(os.path.join(temp_path, str(file_name)))
		logger.info('Subfolder of parent created successfully')
	except OSError as exc:  # Guard against race condition
		if exc.errno != errno.EEXIST:
			logger.warning('Child folder was not created correctly. Probably already present')
			raise


def write_image_on_temp_file(file_name, image_np, counter=0, temp_path='temp'):
	logger.info('Writing temp images on disk...')
	image_filename = os.path.join(str(file_name), '_page_' + str(counter) + '.jpeg')
	cv2.imwrite(
		filename=os.path.join(temp_path, image_filename),
		img=image_np
	)
	logger.info('Image_ ' + str(counter) + 'wrote on disk')


def from_pdf_to_pil_list_images(file_path):
	# print('Generating images from PDF...')
	logger.info('Generating images from PDF...')
	# all_pages = wImage(filename=file_path, resolution=300)
	all_pages = convert_from_path(
		pdf_path=file_path,
		dpi=extraction_dpi,
		fmt='jpeg',
		thread_count=2
	)
	logger.info('All pages converted from pdf')
	bw_pil_list = []
	logger.info('Converting images in greyscale...')
	for page in all_pages:
		pg = page.convert(
			mode='L'
		)
		bw_pil_list.append(pg)
		logger.info('Page converted in greyscale and appended to returning list')
	return bw_pil_list


def beautify_pages(bw_pil_list, create_temp_folder=False, temp_path='temp', file_name=None):
	"""
	Do some modifications to the pil list to make recognition work better
	:param bw_pil_list:
	:param create_temp_folder:
	:param temp_path:
	:param file_name:
	:return:
	"""
	logger.info('Making pages looks better for recognition...')
	# run beautifier over image blobs
	pil_beautified_images = []
	counter = 0
	for image in bw_pil_list:
		counter = counter + 1
		image_np = np.asarray(image)
		logger.info('Beautifying page ' + str(counter))
		beautified_image_np = beautify_image(image_np)
		pil_beautified_images.append(Image.fromarray(beautified_image_np))

		if create_temp_folder:
			logger.info('Creating temp files...')
			write_image_on_temp_file(file_name, beautified_image_np, counter, temp_path)
			logger.info('Temp files created.')

	return pil_beautified_images


def beautify_image(np_array_image):
	"""
	Do some modifications to images
	:param np_array_image:
	:return: np_array_image
	"""
	logger.info('Beautifying images...')
	# threshold = cv2.threshold(np_array_image, 150, 255, cv2.THRESH_BINARY)[1]
	# threshold = cv2.medianBlur(threshold, 3)
	#
	# threshold = cv2.adaptiveThreshold(np_array_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 8)
	# deskewing images:
	logger.info('Doing deskew...')
	sd = deskew.Deskew(
		input_numpy=np_array_image,
		output_numpy=True
	)
	de_skewed_image_np = sd.run()
	logger.info('Deskew done.')
	to_return = de_skewed_image_np
	logger.info('Image beautified.')
	return to_return


def generate_pil_images_from_pdf(file_path, create_temp_folder=False, temp_path='temp'):
	"""
	Takes a pdf file and convert it to jpeg bw images. create_temp_folder decide to write images to temp_path path.
	:param file_path: /path/to/pdf.pdf
	:param create_temp_folder: True/False. For development use. Using pillow images in a pipeline it is not needed
	:param temp_path: /path/to/tempfiles. It is not deleted automatically
	:return: pillow images list of betterified images of pdf
	"""
	file_name = os.path.splitext(file_path)[0] \
		.split("\\")[-1]
	if create_temp_folder:
		logger.info('Creating temp folder...')
		clear_and_create_create_temp_folders(file_name)
		logger.info('Temp folder created')
	bw_pil_list = from_pdf_to_pil_list_images(file_path)
	# bar = pyprind.ProgPercent(len(bw_pil_list), track_time=True, title='Processing images...', stream=sys.stdout)
	# bar.update()
	bw_beautified_pil_list = beautify_pages(
		bw_pil_list=bw_pil_list,
		create_temp_folder=create_temp_folder,
		temp_path=temp_path,
		file_name=file_name
	)
	logging.info('Extraction of pages from pdf completed')
	return bw_beautified_pil_list



# file_path='C:\\Users\\giova\\Documents\\PycharmProjects\\Polizze\\glossario.pdf'
#
# generate_pil_images_from_pdf(
# 	file_path=file_path,
# 	create_temp_folder=True,
# 	temp_path='temp'
# )


