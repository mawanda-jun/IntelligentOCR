"""
The first step of the pipeline lead us to generate good images from pdf to do inference and OCR.
To avoid memory leak - as the user can upload very large pdf files - I've decided to use tue utility
pdftoppm and access one page at a once.
Then the pages are beautified - this part can be better, since the only thing I do here is deskewing pages
In particular, for deskewing object a personalized version of alyn has been created and must be installed
from wheels/alyn-xxx.whl: now it is possible to load, deskew and retrieve a numpy image without writing it
on disk.
If needed the user can write resulting images on disk.
"""
from PIL import Image
from alyn import deskew
import os
import errno
import numpy as np
from costants import \
    EXTRACTION_DPI, \
    TEMP_IMG_FOLDER_FROM_PDF, \
    PATH_TO_EXTRACTED_IMAGES, \
    TEST_PDF_PATH
from personal_errors import InputError, OutputError, APIError
from subprocess import Popen, PIPE, STDOUT
import copy

import logging
from logger import TimeHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TimeHandler().handler)


def clear_and_create_temp_folders(path_to_folder=PATH_TO_EXTRACTED_IMAGES):
    """
    Create a folder with file name to store images extracted from pdf. If path exists it is deleted and then re-created
    :param path_to_folder: path/to/folder in which to store images.
    :return void
    """
    logger.info('Clear and create temp file for images from pdf')
    try:
        os.makedirs(path_to_folder)
        logger.info('Folder created successfully')
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            message = '{path}\nwas not created correctly.' \
                .format(path=path_to_folder)
            raise InputError(
                message=message
            )
        else:
            logger.info('Folder exists')


def write_image_on_disk(file_name, pil_image, page=0, path=PATH_TO_EXTRACTED_IMAGES):
    """
    Writes image on disk
    :param file_name: name of original file
    :param pil_image: numpy array greyscale image
    :param page: page counter from upward function.
    :param path: path/to/folder where to write images
    :return:
    """
    logger.info('Writing temp images on disk...')
    path_to_image = os.path.join(path, '{fn}_page_{c}.jpeg'.format(fn=file_name, c=page))
    try:
        pil_image.save(path_to_image, dpi=(EXTRACTION_DPI, EXTRACTION_DPI))
        logger.info('Image_{} wrote on disk'.format(page))
    except IOError or ValueError as e:
        raise OutputError(
            message='Cannot write image on disk: \n{}'.format(e)
        )


def from_pdf_to_pil_generator(file_path, temp_folder=TEMP_IMG_FOLDER_FROM_PDF, thread_name=None):
    """
    Create a page generator from pdf to make it load less RAM as it takes one page at a once. It read a page at once from
    pdf, then acquire it in RAM and offer as generator.
    It temporarly write the image in temp_folder, then it delete it automatically
    :param file_path: path/to/file.pdf
    :param thread_name: name of the thread in case of batch process
    :param temp_folder: path/to/folder to store temp image before acquiring it in RAM
    :return: PIL generator. Return None if nothing is found
    """

    if not os.path.isfile(file_path):
        raise InputError(
            message='{} not found'.format(file_path)
        )
    else:
        page = 1
        # logger.info("Creating page generator from {path}...".format(path=file_path))
        if not os.path.isdir(temp_folder):
            try:
                os.makedirs(temp_folder)
                logger.info('Temp folder for extraction written on disk')
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise OutputError(
                        message=exc
                    )
                else:
                    logger.info('{} already exists. No need to create it'.format(temp_folder))
        # Extract one page at a once. The iterator goes from first page to last until it reaches the end. In that case a
        # StopIteraton is raised.
        # Uses pdftoppm
        while True:

            args = [
                "pdftoppm",
                "-l",
                str(page),
                "-f",
                str(page),
                "-r",
                str(EXTRACTION_DPI),
                "-gray",
                file_path,
                os.path.join(temp_folder, "temp-{}".format(thread_name))
            ]

            # args.append(item for item in config_list)

            proc = Popen(
                args,
                stdin=PIPE,
                stdout=PIPE,
                stderr=STDOUT,
                # cwd=os.path.join(temp_folder)
            )
            output, outerr = proc.communicate()

            if proc.returncode == 0:
                # Everything went well
                logger.info("Page {} successfully extracted".format(page))
                # checking if the number of pages goes up to 999 pages. In the case that the number of pages is > 10,
                # the temp file number of the first page will be 01 instead of 1. If num_pages > 100, then 001 instead of 1.
                # here we check if temp file exists, if not we check the 01 one and so on.
                fp = os.path.join(temp_folder, 'temp-{tn}-{n}.pgm'.format(n=page, tn=thread_name))
                if page < 10:
                    if not os.path.isfile(fp):
                        fp = os.path.join(temp_folder,
                                          'temp-{tn}-0{n}.pgm'.format(n=page, tn=thread_name))
                        if not os.path.isfile(fp):
                            fp = os.path.join(temp_folder,
                                              'temp-{tn}-00{n}.pgm'.format(n=page, tn=thread_name))

                elif 11 <= page <= 100:
                    if not os.path.isfile(fp):
                        fp = os.path.join(temp_folder,
                                          'temp-{tn}-0{n}.pgm'.format(n=page, tn=thread_name))

                try:
                    img = Image.open(fp)
                    # explicit copy of image so we can delete it from disk safely
                    img = copy.deepcopy(img)
                    if os.path.exists(fp):
                        os.remove(fp)
                    # convert image to greyscale mode
                    img.convert(mode='L')
                    page += 1
                    # return it as a generator
                    yield img
                    # return img
                except FileNotFoundError as e:
                    raise InputError(
                        message=e
                    )

            # case mostly used for stopping iteration when EOF
            else:
                if outerr is None:
                    logger.warning('pdftoppm output: {}'.format(output))
                    logger.warning('Probably reached end of file.')
                    raise StopIteration
                else:
                    logger.error('Something went wrong...')
                    logger.error('pdftoppm output: {}'.format(output))
                    raise InputError(
                        message='pdftoppm error: {}'.format(outerr)
                    )


def beautify_pages(page_generator, file_name, extraction_path=PATH_TO_EXTRACTED_IMAGES):
    """
    Function to beautify pages for inference.
    :param page_generator: list of pillow images
    :return: beautified list of pages
    """
    counter = 0
    for page in page_generator:
        # if page was not converted to greyscale yet
        page_grey = page.convert(
            mode='L'
        )
        logger.info('Page converted to greyscale')
        # load image as np for beautifying
        logger.info('Beautifying pages...')
        # I decided to make another function to beautify a single page at a once avoiding correlation
        image_np = np.asarray(page_grey)
        beautified_np = beautify_image(image_np)
        page_grey = Image.fromarray(beautified_np).convert('L')
        if extraction_path is not None:
            destination_folder = os.path.join(extraction_path, file_name)
            logger.info('Creating folder: {}'.format(destination_folder))
            clear_and_create_temp_folders(path_to_folder=destination_folder)
            logger.info('Temp folder created')
            # create a deep copy of generator since the for loops consume generators
            # copy_of_pil_gen = copy.deepcopy(bw_beautified_pil_gen)
            logger.info('Writing images on disk')
            write_image_on_disk(file_name, copy.deepcopy(page_grey), counter, path=destination_folder)
            counter += 1
        logger.info('Pages beautified')
        # page = page_grey

        # return b/w pil generator
        yield page_grey


def beautify_image(np_array_image):
    """
    Do some modifications to images. This is the right place to put background noise removal, for example.
    Here we only de-skew images to help OCR and table recognition later
    :param np_array_image: input numpy array image
    :return: a beautified numpy array image
    """
    logger.info('Beautifying images...')

    logger.info('Doing deskew...')
    try:
        sd = deskew.Deskew(
            input_numpy=np_array_image,
            output_numpy=True
        )
        de_skewed_image_np = sd.run()
        logger.info('Deskew done.')

        to_return = de_skewed_image_np
        logger.info('Image beautified.')
        return to_return
    except Exception as e:
        # deskew is not so well implemented so I'm catching every exception
        raise APIError(
            message='Deskew is not performing well. Please check API\n{}'.format(e)
        )


def generate_pil_images_from_pdf(file_path, temp_path=TEMP_IMG_FOLDER_FROM_PDF, thread_name='',
                                 extraction_path=PATH_TO_EXTRACTED_IMAGES):
    """
    Takes a pdf file and offer it as a generator of pillow 8-bit greyscale single channel images.
    :param file_path: /path/to/pdf.pdf
    :param temp_path: /path/to/tempfiles.
    :param thread_name: name of referring thread
    :param extraction_path: default is None, path/to/folder to save the result of beautified images on disk
    :return: dict with: 'status': True if everything went good, False instead. Messages/data are inside 'data'
    """

    file_name = os.path.basename(file_path).split('.')[0]
    # clear temp path to store the extracted pages
    # effectively extract pages
    pil_gen = from_pdf_to_pil_generator(file_path, thread_name=thread_name, temp_folder=temp_path)
    # beautify pages before do inference on them. Possibility to write result on disk
    # with yield we cannot check if the status of the return is False or True,
    # so we have to manage it inside beautify_pages
    bw_beautified_pil_gen = beautify_pages(page_generator=pil_gen, file_name=file_name, extraction_path=extraction_path)
    # logger.info('Extraction of pages from pdf completed')

    return bw_beautified_pil_gen


if __name__ == '__main__':
    generator = generate_pil_images_from_pdf(
        file_path=TEST_PDF_PATH,
        temp_path=TEMP_IMG_FOLDER_FROM_PDF,
        extraction_path=PATH_TO_EXTRACTED_IMAGES
    )
    for image in generator:
        print(image)
