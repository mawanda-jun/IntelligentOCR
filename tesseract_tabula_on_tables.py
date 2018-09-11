from subprocess import Popen, PIPE, STDOUT
import os
from costants import \
    TABLE_FOLDER, \
    TEST_TABLE_PATH
from personal_errors import InputError, OutputError
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
    pdf_name = os.path.dirname(table_path).split(os.path.sep)[-1]
    # checking if file exists
    if not os.path.isfile(table_path):
        raise InputError('{} not found'.format(table_path))
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
    start_folder = os.path.join(destination_pdf_path, pdf_name)
    proc = Popen(
        args,
        stdin=PIPE,
        stdout=PIPE,
        stderr=STDOUT,
        cwd=start_folder
    )
    # do the actual job on CL
    output, outerr = proc.communicate()

    if proc.returncode == 0:
        # Everything went well
        logger.info("pdf was successfully extracted from: {}".format(input_file_name))
        logger.info('Tesseract output: {}'.format(output))

        # now extracting tables with tabula
        tabula_input_path = os.path.join(destination_pdf_path, pdf_name, '{}.pdf'.format(input_file_name))
        if not os.path.isfile(tabula_input_path):
            raise InputError('{} was not found. Maybe was not created?'.format(tabula_input_path))
        tabula_output_path = os.path.join(destination_pdf_path, pdf_name, '{}.csv'.format(input_file_name))
        try:
            tabula.convert_into(
                tabula_input_path,
                output_path=tabula_output_path,
                output_format='csv',
                pages='all'
            )
        except Exception as e:
            raise OutputError('Tabula is not performing well...\n{}'.format(e))

    else:
        # something went wrong
        logger.error("Error while extracting pdf from {}".format(input_file))
        logger.error("Tesseract Output: {}".format(output))
        raise OutputError('Tesseract is not performing well...\n{}'
                          .format("Tesseract Error: {}".format(outerr)))


if __name__ == '__main__':
    do_tesseract_on_tables(TEST_TABLE_PATH, TABLE_FOLDER)
