import os
from costants import TEXT_FOLDER
from PIL import Image as Image
import glob
import errno
import pytesseract
import numpy as np
import cv2


def do_ocr_to_text(pil_image, file_name, output_folder=TEXT_FOLDER):
	# Define config parameters.
	# '-l eng'  for using the English language
	# '--oem 1' for using LSTM OCR Engine
	# --psm 4  Assume a single column of text	of variable	sizes.
	# '--psm 12' for sparse text with OSD. 3 is default and it's not working bad.
	config = '-l ita --oem 1 --psm 4'
	txt = pytesseract.image_to_string(pil_image, config=config)
	txt += '\n\n\n'

	# check if destination folder already exists
	if not os.path.exists(os.path.dirname(output_folder)):
		try:
			os.makedirs(output_folder)
		except OSError as exc:  # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise

	# write the output file
	with open(os.path.join(output_folder, file_name + '.txt'), 'a', encoding='utf-8') as result:
		result.write(txt)
	# print(line)
