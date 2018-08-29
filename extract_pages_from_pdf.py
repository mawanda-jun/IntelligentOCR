from wand.image import Image as wImage
from wand.image import Color as wColor
from PIL import Image as Image
from alyn import deskew
import io
import os
import errno
import numpy as np
import shutil
import cv2
import pyprind
import sys
import glob
from threading import Thread

# import argparse
#
# parser = argparse.ArgumentParser(description='Path to input pdf file')
# parser.add_argument('--pdf_path', dest='pdf_path', help='the complete path to your pdf file')
# args = parser.parse_args()


class MyThread (Thread):
	def __init__(self, name, path):
		Thread.__init__(self)
		self.name = name
		self.path = path

	def add_path(self, path):
		if self.path[0] == '':
			self.path.append(path)
		else:
			self.path = [path]

	def run(self):
		for file_path in self.path:
			extract_pages(file_path)


def extract_pages(file_path):
	print('Now processing: ' + file_path)
	file_name = os.path.splitext(file_path)[0] \
		.split("\\")[-1]
	if os.path.isdir('PDFs/' + str(file_name)):
		shutil.rmtree('PDFs/' + str(file_name), ignore_errors=True)
	try:
		os.mkdir('PDFs/' + str(file_name))
	except OSError as exc:  # Guard against race condition
		if exc.errno != errno.EEXIST:
			raise

	# folder_path = file_path.replace(file_name + '.pdf', '') \
	# 	.replace('..\\pdf\\', '', 1) \
	# 	.replace('\\', '/')

	req_image = []

	# wand converts all the separate pages into separate image blobs
	print('Generating images from PDF...')
	all_pages = wImage(filename=file_path, resolution=300)

	# print('Processing images...')
	bar = pyprind.ProgPercent(len(all_pages.sequence) * 2, track_time=True, title='Processing images...', stream=sys.stdout)

	for i, page in enumerate(all_pages.sequence):
		with wImage(page) as img:
			img.background_color = wColor('white')
			img.alpha_channel = 'remove'
			img.type = 'grayscale'
			# img.format = 'jpeg'
			img.convert('jpeg')
			# img.save(filename='PDFs/polizza/prova' + str(i) + '.jpeg')
			req_image.append(img.make_blob('jpeg'))
			bar.update()

	# append all the blobs into req_image
	# for img in image_jpeg.sequence:
	# 	img_page = wImage(image=img)
	# 	img_page.type = 'grayscale'
	# 	req_image.append(img_page.make_blob('jpeg'))

	image_pages = []

	counter = 0
	# run beautifier over image blobs
	for img in req_image:
		with Image.open(io.BytesIO(img)) as image:
			# image.convert('LA')

			counter = counter + 1
			image_np = np.asarray(image)
			beautified_image_np = beautify_image(image_np)

			# image_pages.append(beautified_image_np)
			image_filename = str(file_name) + '_page_' + str(counter) + '.jpeg'
			cv2.imwrite(
				filename=os.path.join('PDFs', file_name, image_filename),
				img=beautified_image_np
			)
			bar.update()

			# txt = pytesseract.image_to_string(beautified_image, config=config)
			# final_text.append(txt)

	# counter = 0
	# for image_np in image_pages:
	# 	counter = counter + 1
	# 	#  write the output file
		# image_filename = str(file_name) + 'page' + str(counter) + '.png'
		# cv2.imwrite(
		# 	filename=os.path.join('PDFs', file_name, image_filename),
		# 	img=image_np
		# )

	# print(final_text)
	# with open(write_path, 'w', encoding='utf-8') as result:
	# 	for line in final_text:
	# 		result.write(line)
	# print(line)


def beautify_image(array_image):
	# threshold = cv2.threshold(array_image, 150, 255, cv2.THRESH_BINARY)[1]
	# threshold = cv2.medianBlur(threshold, 3)
	#
	# threshold = cv2.adaptiveThreshold(array_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 8)

	sd = deskew.Deskew(
		input_numpy=array_image,
		output_numpy=True
	)
	de_skewed_image_np = sd.run()
	to_return = de_skewed_image_np
	return to_return


# extract_pages(os.path.join('PDFs', 'polizza.pdf'))
# os.mkdir('PDFs/' + 'ciao')
# extract_pages(args.pdf_path)

pathList1 = []
pathList2 = []
counter = 0
# for file in os.listdir("pdf/"):
for file in glob.iglob("..\\Polizze\\" + '/**/*.pdf', recursive=True):
	# if file.endswith(".pdf"):
	pdfPath = os.path.join(file)
	if counter % 2 == 0:
		pathList1.append(pdfPath)
	else:
		pathList2.append(pdfPath)
	counter = counter+1


thread1 = MyThread("Thread1", pathList1)
thread2 = MyThread("Thread2", pathList2)

thread1.start()
thread2.start()

thread1.join()
thread2.join()

print("extraction completed")

