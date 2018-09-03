from subprocess import Popen, PIPE, STDOUT
import os
import glob
from logger import return_handler
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_name = 'pipeline-' + str(0) + '.log'
logger.addHandler(return_handler(file_name))

"""
Takes care of decrypting/verifying the downloaded
p7m file into a readable XML file. Depends on the downloader stage
"""


def do_tesseract_on_tables(file_name, temp_table_path):
	"""

	:param file_name:
	:param temp_table_path:
	:return:
	"""
	for file in glob.iglob(os.path.join(temp_table_path, file_name) + '/**/*.jpeg', recursive=True):
		input_file_name = os.path.splitext(file)[0] \
			.split("\\")[-1]
		input_file = file
		# Define config parameters.
		# '-l eng'  for using the English language
		# '--oem 1' for using LSTM OCR Engine
		# --oem 2 for using Legacy + LSTM engines
		# '--psm 12' for sparse text with OSD. 3 is default and it's not working bad.

		config = '-l ita --oem 2 --psm 12 pdf'
		config_list = config.split(' ')

		args = [
			"tesseract",
			input_file,
			input_file_name,
			'out',
			*config_list,
			'pdf'
		]

		# args.append(item for item in config_list)

		print(args)
		proc = Popen(
			args,
			stdin=PIPE,
			stdout=PIPE,
			stderr=STDOUT,
			cwd=os.path.join(temp_table_path, file_name)
		)
		output, outerr = proc.communicate()

		if proc.returncode == 0:
			# Everything went well
			logger.info("pdf was succesfully extracted from: {}"
							.format(input_file_name))

		else:
			logger.error("Error while extracting pdf from {}".format(input_file))
			logger.error("Tesseract Output: {}".format(output))
			logger.error("Tesseract Error: {}".format(outerr))


if __name__ == '__main__':
	do_tesseract_on_tables('AGBA_app_11_', 'table')

