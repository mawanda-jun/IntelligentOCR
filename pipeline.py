import os
from extract_pages_from_pdf import generate_pil_images_from_pdf
from find_table import extract_tables_and_text, write_crops, clear_and_create_temp_folders
from memory_profiler import profile
import glob
import logging
from logger import return_handler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.addHandler(return_handler())


@profile
def pipeline(pdf_path, inference_graph_path):
		# if file.endswith(".jpeg"):
		path_to_pdf = os.path.join(pdf_path)
		file_name = os.path.splitext(path_to_pdf)[0] \
			.split("\\")[-1]
		logger.info('Now elaborating: ' + str(file_name))
		bw_pil_list = generate_pil_images_from_pdf(
			file_path=path_to_pdf,
			create_temp_folder=False,
			temp_path='temp'
		)
		cropped_text = []
		cropped_tables = []
		for pil_image in bw_pil_list:
			c_tables, c_text = extract_tables_and_text(
				pil_image=pil_image,
				inference_graph_path=inference_graph_path
			)
			cropped_tables.extend(c_tables)
			cropped_text.append(c_text)
		logger.info('Extraction of tables and text completed')
		if not cropped_tables == [] or cropped_text is not None:
			clear_and_create_temp_folders(file_name)
			write_crops(
				file_name=file_name,
				cropped_tables=cropped_tables,
				cropped_text=cropped_text,
				temp_table_path='table',
				temp_text_path='text'
			)
		logger.info('Writing tables and text on disk completed')



INFERENCE_GRAPH = 'C:\\Users\\giova\\Documents\\PycharmProjects\\TableTrainNet\\data\\frozen_inference_graph_momentum.pb'

alupress_polizza_completa = 'C:\\Users\\giova\\Documents\\PycharmProjects\\Polizze\\polizza Alupress\\ALUPRESS POLIZZA COMPLETA rev. MM 2018.pdf'
agbra_polizza_incendio = 'C:\\Users\\giova\\Documents\\PycharmProjects\\Polizze\\Polizza AGBA\\AGBA pol INCENDIO.pdf'

PATH = agbra_polizza_incendio

pipeline(
	pdf_path=PATH,
	inference_graph_path=INFERENCE_GRAPH
)


for file in glob.iglob("..\\Polizze\\" + '/**/*.pdf', recursive=True):
	# if file.endswith(".pdf"):
	PATH = os.path.join(file)
	pipeline(
		pdf_path=PATH,
		inference_graph_path=INFERENCE_GRAPH
	)
