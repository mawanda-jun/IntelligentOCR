THREADS = 8
EXTRACTION_DPI = 300
TEMP_IMG_FOLDER_FROM_PDF = 'pdf_temp'
inference_graph_momentum = \
	'C:\\Users\\giova\\Documents\\PycharmProjects\\TableTrainNet\\data\\frozen_inference_graph_momentum.pb'

inference_graph_adam_3 = \
	'C:\\Users\\giova\\Documents\\PycharmProjects\\TableTrainNet\\' \
	'model__rcnn_inception_adam_3\\frozen\\frozen_inference_graph.pb'

# alupress_polizza_completa =
# 'C:\\Users\\giova\\Documents\\PycharmProjects\\Polizze\\polizza Alupress\\ALUPRESS POLIZZA COMPLETA rev. MM 2018.pdf'
# agbra_polizza_incendio = 'C:\\Users\\giova\\Documents\\PycharmProjects\\Polizze\\Polizza AGBA\\AGBA pol INCENDIO.pdf'

INFERENCE_GRAPH = inference_graph_adam_3
TEST_PDF_PATH = 'C:\\Users\\giova\\Documents\\PycharmProjects\\Polizze\\glossario.pdf'

TABLE_TEMP_FOLDER = 'table'
TEXT_TEMP_FOLDER = 'text'

MAX_NUM_BOXES = 10
MIN_SCORE = 0.4