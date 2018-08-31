import logging
import os
import time


def return_handler():
	# create a file handler
	file_name = 'pipeline-' + str(0) + '.log'
	handler = logging.FileHandler(file_name)
	handler.setLevel(logging.INFO)

	# create a logging format
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)
	return handler

# add the handlers to the logger

