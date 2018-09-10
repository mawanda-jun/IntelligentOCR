from subprocess import Popen, PIPE, STDOUT
import os
from costants import \
    TABLE_FOLDER, \
    TEST_TABLE_PATH
import tabula
from logger import TimeHandler
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.addHandler(TimeHandler().handler)


def do_tesseract_on_tables(table_path, destination_pdf_path=TABLE_FOLDER):
    """

    :param table_path:
    :param destination_pdf_path:
    :return:
    """
    # takes only file name without extension
    input_file_name = os.path.basename(table_path).split(os.extsep)[0]
    # take the name of the folder in which the images are stored, that is the name of the original pdf
    pdf_name = os.path.dirname(table_path).split(os.altsep)[-1]
    input_file = table_path
    # Define config parameters.
    # '-l eng'  for using the English language
    # '--oem 1' for using LSTM OCR Engine
    # --oem 2 for using Legacy + LSTM engines NOT AVAILABLE IN ITALIAN
    # '--psm 12' for sparse text with OSD. 3 is default and it's not working bad.

    config = '-l ita --oem 1 --psm 12 pdf'
    config_list = config.split(' ')  # make a list of parameters

    args = [
        "tesseract",
        input_file,  # actual file to be analyzed
        input_file_name,  # output file name
        *config_list,  # extract all parameters inside this array
        'pdf'
    ]

    proc = Popen(
        args,
        stdin=PIPE,
        stdout=PIPE,
        stderr=STDOUT,
        cwd=os.path.join(destination_pdf_path, pdf_name)
    )
    # do the actual job on CL
    output, outerr = proc.communicate()

    if proc.returncode == 0:
        # Everything went well
        logger.info("pdf was successfully extracted from: {}".format(input_file_name))
        logger.info('Tesseract output: {}'.format(output))

        # now extracting tables with tabula
        tabula_input_path = os.path.join(destination_pdf_path, pdf_name, '{}.pdf'.format(input_file_name))
        tabula_output_path = os.path.join(destination_pdf_path, pdf_name, '{}.csv'.format(input_file_name))
        tabula.convert_into(
            tabula_input_path,
            output_path=tabula_output_path,
            output_format='csv',
            pages='all'
        )

    else:
        # something went wrong
        logger.error("Error while extracting pdf from {}".format(input_file))
        logger.error("Tesseract Output: {}".format(output))
        logger.error("Tesseract Error: {}".format(outerr))


if __name__ == '__main__':
    do_tesseract_on_tables(TEST_TABLE_PATH, TABLE_FOLDER)
