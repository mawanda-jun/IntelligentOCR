from extract_pages_from_pdf import generate_pil_images_from_pdf
from find_table import extract_tables_and_text, write_crops, create_temp_folders
from tesseract_tabula_on_tables import do_tesseract_on_tables
from tesseract_on_text import do_ocr_to_text
from personal_errors import InputError, OutputError
from costants import \
    INFERENCE_GRAPH, \
    TEST_PDF_PATH, \
    TEXT_FOLDER, \
    TABLE_FOLDER, \
    TEMP_IMG_FOLDER_FROM_PDF
from io import StringIO
import os
from memory_profiler import profile
import logging
from logger import TimeHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.addHandler(TimeHandler().handler)

fp = StringIO()


@profile(stream=fp)
def pipeline(pdf_path, inference_graph_path, thread_name=None):
    # if file.endswith(".jpeg"):
    path_to_pdf = os.path.join(pdf_path)
    if not os.path.isfile(path_to_pdf):
        raise InputError('{} not found'.format(path_to_pdf))
    # replace empty spaces with underscore to avoid problems while creating folders
    pdf_name = os.path.basename(path_to_pdf).split(os.extsep)[0].replace(" ", "_")
    logger.info('Now elaborating: {}'.format(pdf_name))
    bw_pil_gen = generate_pil_images_from_pdf(
        file_path=path_to_pdf,
        temp_path=TEMP_IMG_FOLDER_FROM_PDF,
        thread_name=thread_name
    )
    # cropped_text = []
    # cropped_tables = []
    page_number = 0
    # create temp folders
    create_temp_folders(pdf_name)
    for pil_image in bw_pil_gen:
        c_tables, c_text = extract_tables_and_text(
            pil_image=pil_image,
            inference_graph_path=inference_graph_path
        )
        # yield (c_tables, c_text)
        # cropped_tables.extend(c_tables)
        # cropped_text.append(c_text)
        logger.info('Extraction of tables and text completed')

        if not c_tables == []:
            table_paths, text_path = write_crops(
                file_name=pdf_name,
                cropped_tables=c_tables,
                cropped_text=None,
                temp_table_path=TABLE_FOLDER,
                temp_text_path=None,
                page_number=page_number
            )
            for table_path in table_paths:
                do_tesseract_on_tables(table_path, TABLE_FOLDER)
            # counter_table = 0
            table_name = 'table_pag_{pag_num}'.format(pag_num=page_number)
            if page_number == 0:
                if os.path.isfile(os.path.join(TABLE_FOLDER, pdf_name, table_name + '.txt')):
                    os.remove(os.path.join(TABLE_FOLDER, pdf_name, table_name + '.txt'))
            for c_t in c_tables:
                do_ocr_to_text(
                    pil_image=c_t,
                    file_name=table_name,
                    output_folder=os.path.join(TABLE_FOLDER, pdf_name)
                )
        logger.info('Writing tables on disk completed')

        if c_text is not None:
            table_paths, text_path = write_crops(
                file_name=pdf_name,
                cropped_tables=None,
                cropped_text=c_text,
                temp_table_path=None,
                temp_text_path=TEXT_FOLDER,
                page_number=page_number
            )
            text_name = 'text'
            if page_number == 0:
                if os.path.isfile(os.path.join(TEXT_FOLDER, pdf_name, text_name + '.txt')):
                    os.remove(os.path.join(TEXT_FOLDER, pdf_name, text_name + '.txt'))
            do_ocr_to_text(
                pil_image=c_text,
                file_name=text_name,
                output_folder=os.path.join(TEXT_FOLDER, pdf_name)
            )
            page_number += 1


if __name__ == '__main__':
    pipeline(
        pdf_path=TEST_PDF_PATH,
        inference_graph_path=INFERENCE_GRAPH
    )
    logger.info(fp.getvalue())
    fp.close()
