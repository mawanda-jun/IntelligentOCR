import logging
import os
import time


def return_handler(file_name):
	# create a file handler
	handler = logging.FileHandler(file_name)
	handler.setLevel(logging.INFO)

	# create a logging format
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)
	return handler

# add the handlers to the logger

