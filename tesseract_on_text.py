import os

from PIL import Image as Image
import glob
import errno
import pytesseract
import numpy as np
import cv2


def do_ocr_to_text(pil_image, file_name, output_folder='text'):
	# Define config parameters.
	# '-l eng'  for using the English language
	# '--oem 1' for using LSTM OCR Engine
	# '--psm 12' for sparse text with OSD. 3 is default and it's not working bad.
	config = '-l ita --oem 1 --psm 12'
	txt = pytesseract.image_to_string(pil_image, config=config)

	# check if destination folder already exists
	if not os.path.exists(os.path.dirname(output_folder)):
		try:
			os.mkdir(output_folder)
		except OSError as exc:  # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise

	# write the output file
	with open(os.path.join(output_folder, file_name + '.txt'), 'w', encoding='utf-8') as result:
		result.write(txt)
	# print(line)
