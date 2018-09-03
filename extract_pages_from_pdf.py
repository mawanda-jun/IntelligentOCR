from PIL import Image
from alyn import deskew
import os
import errno
import numpy as np
import shutil
import cv2
import logging
from logger import return_handler
from costants import extraction_dpi, temp_page_from_pdf
from subprocess import Popen, PIPE, STDOUT
import copy

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
	"""
	Create a page generator from pdf to make it load less RAM as it takes one page at a once
	:param file_path:
	:return: image generator from pdf
	"""
	flag = True
	first_page = 1
	last_page = 1
	logger.info("Creating page generator from " + str(file_path) + "...")
	if not os.path.exists(temp_page_from_pdf):
		try:
			os.mkdir(temp_page_from_pdf)
		except OSError as exc:  # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise

	while flag:

		args = [
			"pdftoppm",
			"-l",
			str(first_page),
			"-f",
			str(last_page),
			"-r",
			str(extraction_dpi),
			"-gray",
			file_path,
			os.path.join(temp_page_from_pdf, "temp")
		]

		# args.append(item for item in config_list)

		proc = Popen(
			args,
			stdin=PIPE,
			stdout=PIPE,
			stderr=STDOUT,
			# cwd=os.path.join(temp_page_from_pdf)
		)
		output, outerr = proc.communicate()

		if proc.returncode == 0:
			# Everything went well
			logger.info("page: {}"
							.format(first_page) + 'successfully extracted')
			if first_page < 9:
				fp = os.path.join(temp_page_from_pdf, 'temp-{}.pgm'.format(first_page))
			else:
				fp = os.path.join(temp_page_from_pdf, 'temp-{}.pgm'.format(first_page))
			# print(fp)
			img = Image.open(fp)
			img = copy.deepcopy(img)
			if os.path.exists(fp):
				os.remove(fp)
			img.convert(mode='L')
			yield img
			first_page += 1
			last_page += 1

		else:
			if outerr is None:
				logger.info('Probably reached end of file. The following error messages can be ignored.')
			logger.error("Error while extracting pdf page at {}".format(first_page))
			logger.error("Tesseract Output: {}".format(output))
			logger.error("Tesseract Error: {}".format(outerr))
			flag = False



	# load all pages in lowest res to count the pages
	# with tempfile.TemporaryDirectory() as path:
	# images = convert_from_path(file_path, dpi=1)
	# num_pages = len(images)
	# delete var to free ram
	# del images
	# read one page at a once.
	# for n in range(50):
	# 	img = convert_from_path(
	# 		file_path,
	# 		dpi=extraction_dpi,
	# 		first_page=n,
	# 		last_page=n,
	# 		thread_count=1
	# 		output_folder=path
		# )
		# yield img[0]
	# for img in images:
	# 	logger.info('Generator created')
	# 	yield img


def beautify_pages(page_generator, create_temp_folder=False, temp_path='temp', file_name=None):
	# print('Generating images from PDF...')
	# logger.info('Generating images from PDF...')
	# all_pages = wImage(filename=file_path, resolution=300)
	# all_pages = page_generator(file_path)
	# logger.info('All pages converted from pdf')
	# bw_pil_list = []
	logger.info('Converting images in greyscale...')
	counter = 0
	for page in page_generator:
		page = page.convert(
			mode='L'
		)
		# bw_pil_list.append(pg)
		logger.info('Page converted to greyscale')
		# load image as np for beautifying
		logger.info('Beautifying image...')
		image_np = np.asarray(page)
		beautified_image_np = beautify_image(image_np)
		logger.info('Image beautified')
		if create_temp_folder:
			logger.info('Creating temp files...')
			write_image_on_temp_file(file_name, beautified_image_np, counter, temp_path)
			logger.info('Temp files created.')

	# return b/w pil generator
		yield page


# def beautify_pages(bw_pil_list, create_temp_folder=False, temp_path='temp', file_name=None):
# 	"""
# 	Do some modifications to the pil list to make recognition work better
# 	:param bw_pil_list:
# 	:param create_temp_folder:
# 	:param temp_path:
# 	:param file_name:
# 	:return:
# 	"""
# 	logger.info('Making pages looks better for recognition...')
# 	# run beautifier over image blobs
# 	pil_beautified_images = []
# 	counter = 0
# 	for image in bw_pil_list:
# 		counter = counter + 1
# 		image_np = np.asarray(image)
# 		logger.info('Beautifying page ' + str(counter))
# 		beautified_image_np = beautify_image(image_np)
# 		pil_beautified_images.append(Image.fromarray(beautified_image_np))
#
# 		if create_temp_folder:
# 			logger.info('Creating temp files...')
# 			write_image_on_temp_file(file_name, beautified_image_np, counter, temp_path)
# 			logger.info('Temp files created.')
#
# 	return pil_beautified_images


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
	pil_gen = from_pdf_to_pil_list_images(file_path)
	# bar = pyprind.ProgPercent(len(pil_gen), track_time=True, title='Processing images...', stream=sys.stdout)
	# bar.update()
	bw_beautified_pil_gen = beautify_pages(
		page_generator=pil_gen,
		create_temp_folder=create_temp_folder,
		temp_path=temp_path,
		file_name=file_name
	)
	logging.info('Extraction of pages from pdf completed')
	# print('Fin qui tutto bene')
	return bw_beautified_pil_gen



# file_path='C:\\Users\\giova\\Documents\\PycharmProjects\\Polizze\\glossario.pdf'
#
# generate_pil_images_from_pdf(
# 	file_path=file_path,
# 	create_temp_folder=True,
# 	temp_path='temp'
# )


