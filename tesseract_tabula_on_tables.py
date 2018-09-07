from subprocess import Popen, PIPE, STDOUT
import os
from costants import TABLE_FOLDER
import glob
import tabula
from logger import TimeHandler
import logging

from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.addHandler(TimeHandler().handler)


def do_tesseract_on_tables(table_path, temp_table_path=TABLE_FOLDER):
    """

    :param table_path:
    :param temp_table_path:
    :return:
    """
    # for file in glob.iglob(os.path.join(temp_table_path, table_path) + '/**/*.jpeg', recursive=True):
    input_file_name = os.path.splitext(table_path)[0] \
        .split("\\")[-1]
    pdf_name = os.path.splitext(table_path)[0] \
        .split("\\")[-2]
    input_file = table_path
    # Define config parameters.
    # '-l eng'  for using the English language
    # '--oem 1' for using LSTM OCR Engine
    # --oem 2 for using Legacy + LSTM engines NOT AVAILABLE IN ITALIAN
    # '--psm 12' for sparse text with OSD. 3 is default and it's not working bad.

    config = '-l ita --oem 1 --psm 12 pdf'
    config_list = config.split(' ')

    img = Image.open(input_file)
    print(img.size)

    args = [
        "tesseract",
        input_file,  # actual file to be analyzed
        input_file_name,  # output file name
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
        cwd=os.path.join(temp_table_path, pdf_name)
    )
    output, outerr = proc.communicate()

    if proc.returncode == 0:
        # Everything went well
        logger.info("pdf was succesfully extracted from: {}".format(input_file_name))
        logger.info('Tesseract output: {}'.format(output))
        tabula_input_path = os.path.join(temp_table_path, pdf_name, str(input_file_name) + '.pdf')
        tabula_output_path = os.path.join(temp_table_path, pdf_name, str(input_file_name) + '.csv')
        tabula.convert_into(
            tabula_input_path,
            output_path=tabula_output_path,
            output_format='csv',
            pages='all'
        )

    else:
        logger.error("Error while extracting pdf from {}".format(input_file))
        logger.error("Tesseract Output: {}".format(output))
        logger.error("Tesseract Error: {}".format(outerr))


if __name__ == '__main__':
    do_tesseract_on_tables('table\\glossario\\table_pag_0_0.jpeg', TABLE_FOLDER)
