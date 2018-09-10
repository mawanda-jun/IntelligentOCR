import os


THREADS = 8
EXTRACTION_DPI = 300
TEMP_IMG_FOLDER_FROM_PDF = 'pdf_temp'
PATH_TO_EXTRACTED_IMAGES = None  # set to None if you don't want any
inference_graph_momentum = \
    os.path.join(os.path.join(
        'C:/Users/giova/Documents/PycharmProjects/TableTrainNet/trained_models/model__rcnn_inception_adam_1/frozen/frozen_inference_graph_momentum.pb'))

inference_graph_adam_3 = \
    os.path.join('C:/Users/giova/Documents/PycharmProjects/TableTrainNet/trained_models/'
                 'model__rcnn_inception_adam_3/frozen/frozen_inference_graph.pb')

# alupress_polizza_completa =
# os.path.join('C:/Users/giova/Documents/PycharmProjects/Polizze/polizza Alupress/ALUPRESS POLIZZA COMPLETA rev. MM 2018.pdf')
# agbra_polizza_incendio = os.path.join('C:/Users/giova/Documents/PycharmProjects/Polizze/Polizza AGBA/AGBA pol INCENDIO.pdf')

INFERENCE_GRAPH = inference_graph_adam_3
TEST_PDF_PATH = os.path.join('C:/Users/giova/Documents/PycharmProjects/Polizze/glossario.pdf')
TEST_TABLE_PATH = os.path.join('table/glossario/table_pag_0_0.jpeg')

TABLE_FOLDER = 'table'
TEXT_FOLDER = 'text'

MAX_NUM_BOXES = 10
MIN_SCORE = 0.4
