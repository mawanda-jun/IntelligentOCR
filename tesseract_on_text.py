import os
from costants import TEXT_FOLDER
from personal_errors import APIError
import errno
import pytesseract


def do_ocr_to_text(pil_image, file_name, output_folder=TEXT_FOLDER):
    """
    Takes an image and do ocr on it.
    :param pil_image: pillow image of text
    :param file_name: name of pdf original file
    :param output_folder: /path/to/output_file
    :return:
    """
    # Define config parameters.
    # '-l eng'  for using the English language
    # '--oem 1' for using LSTM OCR Engine
    # --psm 4  Assume a single column of text	of variable	sizes.
    # '--psm 12' for sparse text with OSD. 3 is default and it's not working bad.
    config = '-l ita --oem 1 --psm 4'
    try:
        txt = pytesseract.image_to_string(pil_image, config=config)
        txt += '\n\n\n'

        # check if destination folder already exists
        if not os.path.exists(os.path.dirname(output_folder)):
            try:
                os.makedirs(output_folder)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # write the output file, appending to the existing one
        with open(os.path.join(output_folder, '{}.txt'.format(file_name)), 'a', encoding='utf-8') as result:
            result.write(txt)
    except Exception as e:
        raise APIError('PyTesseract is not performing well...\n{}'
                       .format(e))
# print(line)
