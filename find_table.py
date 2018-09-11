import numpy as np
import os
from costants import \
    EXTRACTION_DPI, \
    TEXT_FOLDER, \
    TABLE_FOLDER, \
    MAX_NUM_BOXES, \
    MIN_SCORE
from personal_errors import InputError, OutputError
import errno
import tensorflow as tf
from PIL import Image
from alyn import deskew
import logging
from logger import TimeHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.addHandler(TimeHandler().handler)


def reshape_image_into_numpy_array(pil_image):
    """
    The neural network needs a numpy RGB 3-channels image (because of the pre-trained network)
    So we need to convert a pillow image into a numpy int8 height*width*3 array
    We cannot use zero instead of the two additional layers because the NN uses every channel to make predictions,
    so if we fill the array with zeros the scores become 1/3.

    :param pil_image: a pillow image
    :return: a reshaped numpy image ready for inference
    """
    logger.info('Converting pillow image in numpy 3-dimension array...')
    (im_width, im_height) = pil_image.size
    np_array = np.array(pil_image.getdata()).reshape((im_height, im_width, 1)).astype(np.uint8)
    logger.info('Pillow image converted in heigth*width*1 numpy image')
    np_array = np.concatenate((np_array, np_array, np_array), axis=2)
    logger.info('Numpy 3-dimension array created')
    return np_array


def do_inference_with_graph(pil_image, inference_graph_path):
    """
    It takes a pillow image and looks for tables inside

    :param pil_image: Pillow image
    :param inference_graph_path:
    :return: (boxes, scores), two lists with all the boxes and their likelihood scores
    """
    logger.info('Reading inference graph...')
    detection_graph = tf.Graph()
    # checking if inference graph exists
    if not os.path.isfile(inference_graph_path):
        raise InputError('Inference graph at\n{}\nnot found'.format(inference_graph_path))

    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = reshape_image_into_numpy_array(pil_image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            logger.info('Running inference...')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            logger.info('Inference run, boxes and scores have been found')
            return boxes[0], scores[0]


def check_if_intersected(coord_a, coord_b):
    """
    Check if the rectangular b is not intersected with a
    :param coord_a: dict with {y_min, x_min, y_max, x_max}
    :param coord_b: same as coord_a
    :return: true if intersected, false instead
    """
    logger.info('Returning if the two boxes are intersected...')
    return \
        coord_a['x_max'] > coord_b['x_min'] and \
        coord_a['x_min'] < coord_b['x_max'] and \
        coord_a['y_max'] > coord_b['y_min'] and \
        coord_a['y_min'] < coord_b['x_max']


def check_if_vertically_overlapped(box_a, box_b):
    """
    Return if box_b is intersected vertically with coord_a boxes.
    :param box_a:
    :param box_b:
    :return: true if intersected, false instead
    """
    return \
        box_a['y_min'] < box_b['y_min'] < box_a['y_max'] or \
        box_a['y_min'] < box_b['y_max'] < box_a['y_max'] or \
        (box_a['y_min'] >= box_b['y_min'] and box_a['y_max'] <= box_b['y_max']) or \
        (box_a['y_min'] <= box_b['y_min'] and box_a['y_max'] >= box_b['y_max'])


def merge_vertically_overlapping_boxes(boxes):
    """
    Returns a list of boxes that has been merged together if vertically overlapping.
    This is recursive - aka no overlapping box is left behind even if only the last one is detected.
    This algorithm works in our case because the boxes that are inside boxes are already ordered by score. So the
    first one is scored higher than the second and so on. This is a pre-condition.
    :param boxes: list of boxes possibly overlapping
    :return: list of merged boxes
    """
    # first box is always inside
    merged_boxes = [boxes[0]]
    i = 0
    overlapping = False
    for box in boxes[1:]:
        i += 1
        # extraction of coordinates for better reading
        coord_box = {
            'y_min': box[0],
            'x_min': box[1],
            'y_max': box[2],
            'x_max': box[3]
        }
        for m_box in merged_boxes:
            # extraction of coordinates for better reading
            coord_m_box = {
                'y_min': m_box[0],
                'x_min': m_box[1],
                'y_max': m_box[2],
                'x_max': m_box[3]
            }

            if check_if_vertically_overlapped(coord_m_box, coord_box):
                overlapping = True
                # merge of the two overlapping boxes
                if m_box[0] > box[0]:
                    m_box[0] = box[0]
                if m_box[2] < box[2]:
                    m_box[2] = box[2]
        if not overlapping:
            # if not overlapping we append the box. Exit condition for recursive call
            merged_boxes.append(box)
    if overlapping:
        # recursive call. It converges because the exit condition consumes the generator.
        return merge_vertically_overlapping_boxes(merged_boxes)
    else:
        return merged_boxes


def keep_best_not_overlapping_boxes(boxes, scores, max_num_boxes=5, min_score=0.8):
    """
    Return the boxes with higher score with no overlapping. Merging NOT applied:
    boxes are: box[0]=ymin, box[1]=xmin, box[2]=ymax, box[3]=xmax

    :param boxes: list of boxes found in inference
    :param scores: likelihood of the boxes
    :param max_num_boxes: max num of boxes to be saved
    :param min_score: min box score to check
    :return: list of the best not overlapping boxes
    """
    logger.info('Detecting best matching boxes...')
    kept_scores = []
    kept_boxes = []  # always keep the firs box, which is the best one.
    num_boxes = 0
    i = 0
    if scores[0] > min_score:
        kept_boxes.append(boxes[0])
        kept_scores.append(scores[0])
        num_boxes += 1
        i += 1
        for b in boxes[1:]:
            if num_boxes < max_num_boxes and scores[i] > min_score:
                intersected = False
                coord_b = {
                    'y_min': b[0],
                    'x_min': b[1],
                    'y_max': b[2],
                    'x_max': b[3]
                }
                for kb in kept_boxes:
                    coord_kb = {
                        'y_min': kb[0],
                        'x_min': kb[1],
                        'y_max': kb[2],
                        'x_max': kb[3]
                    }
                    intersected = check_if_intersected(
                        coord_a=coord_b,
                        coord_b=coord_kb
                    )
                if not intersected:
                    kept_boxes.append(b)
                    num_boxes += 1
                    kept_scores.append(scores[i])

                i += 1
            else:
                break
        # no merge
        # kept_boxes = merge_vertically_overlapping_boxes(kept_boxes)
    else:
        kept_boxes = []

    return kept_boxes, kept_scores


def keep_best_boxes_merged(boxes, scores, max_num_boxes=5, min_score=0.8):
    """
    Return the boxes with higher score with no overlapping. Merging applied:
    boxes are: box[0]=ymin, box[1]=xmin, box[2]=ymax, box[3]=xmax

    :param boxes: list of boxes found in inference
    :param scores: likelihood of the boxes
    :param max_num_boxes: max num of boxes to be saved
    :param min_score: min box score to check
    :return: list of the best not overlapping boxes
    """
    logger.info('Detecting best matching boxes...')
    kept_scores = []
    kept_boxes = []  # always keep the firs box, which is the best one.
    num_boxes = 0
    i = 0
    if scores[0] > min_score:
        kept_boxes.append(boxes[0])
        kept_scores.append(scores[0])
        num_boxes += 1
        i += 1
        for b in boxes[1:]:
            # add boxes to the ones to be merged
            if num_boxes < max_num_boxes and scores[i] > min_score:

                kept_boxes.append(b)
                num_boxes += 1
                kept_scores.append(scores[i])

                i += 1
            else:
                break

        kept_boxes = merge_vertically_overlapping_boxes(kept_boxes)
    else:
        kept_boxes = []

    return kept_boxes, kept_scores


def crop_wide(pil_image, boxes):
    """
    Crop tables from images. To simplify cropping (and to reduce by half the risk of mistake as we consider only two bounds)
    we cut the image widely from the upper bound to the lower. Then creates a image for table and stores into a list
    and parses every remaining text box into one image.
    If no boxes are found only the text image is returned and is equal to pil_image
    :param pil_image: an image in which some table have been found.
    :param boxes: bounding boxes for tables
    :return: pillow list of cropped tables images, pillow image of text.
    """
    cropped_tables = []
    segments = [0]  # adding position 0 to simplify anti-crop text later
    height_of_crops = 0
    logger.info('Checking if there are some boxes recorded...')
    if not boxes == []:
        (im_width, im_height) = pil_image.size

        logger.info('Boxes have been found. Cropping tables...')
        for box in boxes:
            cropped_tables.append(pil_image.crop(tuple((0, int(box[0]), im_width, int(box[2])))))
            segments.append(int(box[0]))
            segments.append(int(box[2]))
            height_of_crops += (int(box[2]) - int(box[0]))
        logger.info('Tables cropped')
        # sorts all segments to simplify anti-crop text later
        segments.append(im_height)  # adding last position to simplify anti-crop text later
        segments.sort()

        # create new image with new dimension
        new_image = Image.new('L', (im_width, im_height - height_of_crops))
        start_position = 0
        logger.info('Creating image from cropped text slices...')
        # cutting image in anti-boxes position
        for i in range(len(segments)):  # segments will always be even
            if not i % 2 and i < len(segments) - 1:  # takes only even positions
                if i != 0:
                    start_position += segments[i - 1] - segments[i - 2]
                new_image.paste(pil_image.crop(tuple((0, segments[i], im_width, segments[i + 1]))), (0, start_position))
        cropped_text = new_image
        logger.info('Created text image')

    else:
        logger.info('No boxes found')
        cropped_text = pil_image

    return cropped_tables, cropped_text


def extract_tables_and_text(pil_image, inference_graph_path):
    """
    Extracts tables and text from image_path using inference_graph_path

    :param pil_image:
    :param inference_graph_path:
    :return: (cropped_tables, cropped_text), list of table pillow images and a text image
    """
    (im_width, im_height) = pil_image.size
    boxes, scores = do_inference_with_graph(pil_image, inference_graph_path)
    best_boxes, best_scores = keep_best_boxes_merged(
        boxes=boxes,
        scores=scores,
        max_num_boxes=MAX_NUM_BOXES,
        min_score=MIN_SCORE
    )
    logger.info("Best boxes are: ")
    for box in best_boxes:
        logger.info(box)
    logger.info("With scores:")
    for score in best_scores:
        logger.info(score)

    # create coordinates based on image dimension
    for box in best_boxes:
        box[0] = int(box[0] * im_height)
        box[2] = int(box[2] * im_height)
        box[1] = int(box[1] * im_width)
        box[3] = int(box[3] * im_width)

    (cropped_tables, cropped_text) = crop_wide(pil_image, best_boxes)
    return cropped_tables, cropped_text


def create_temp_folders(file_name, temp_table_folder=TABLE_FOLDER, temp_text_folder=TEXT_FOLDER):
    """
    Clear any existing table/file_name and text/file_name folder for creating new images

    :param file_name:
    :param temp_table_folder:
    :param temp_text_folder:
    :return: None
    """
    logger.info('Clear and create temp file for images from pdf')
    if not os.path.isdir(temp_table_folder):
        # creates folder for table images per page
        try:
            os.makedirs(temp_table_folder)
            logger.info('{} created successfully'.format(temp_table_folder))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise OutputError('{} was not created correctly.'
                                  .format(temp_table_folder))
            else:
                logger.info('{} already present'.format(temp_table_folder))

    # creates folder for text images per page
    logger.info(temp_text_folder + ' folder created successfully')
    if not os.path.isdir(temp_text_folder):
        try:
            os.makedirs(temp_text_folder)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise OutputError('{} was not created correctly.'
                                  .format(temp_text_folder))
            else:
                logger.info('{} already present'.format(temp_text_folder))

    if os.path.isdir(os.path.join(temp_table_folder, str(file_name))):
        logger.info('Clearing table temp folder from existing files...')
        # shutil.rmtree(os.path.join(temp_table_folder, str(file_name)), ignore_errors=True)
        logger.info('Clear done')
    if os.path.isdir(os.path.join(temp_text_folder, str(file_name))):
        logger.info('Clearing text temp folder from existing files...')
        # shutil.rmtree(os.path.join(temp_text_folder, str(file_name)), ignore_errors=True)
        logger.info('Clear done')

    try:
        logger.info('Creating {}...'.format(temp_table_folder))
        os.makedirs(os.path.join(temp_table_folder, str(file_name)))
        logger.info(temp_table_folder + ' created')
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise OutputError('{} was not created.'.format(temp_table_folder))
        else:
            logger.info('{} already present'.format(temp_table_folder))

    try:
        logger.info('Creating {}...'.format(temp_text_folder))
        os.makedirs(os.path.join(temp_text_folder, str(file_name)))
        logger.info(temp_text_folder + ' created')
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise OutputError('{} was not created'.format(temp_text_folder))
        else:
            logger.info('{} already present'.format(temp_text_folder))


def write_crops(file_name, cropped_tables=None, cropped_text=None, temp_table_path=TABLE_FOLDER,
                temp_text_path=TEXT_FOLDER, page_number=None):
    """
    Writes table and text images under table and text folder

    :param file_name:
    :param cropped_tables: list of pillow images
    :param cropped_text: list of pillow images
    :param temp_table_path:
    :param temp_text_path:
    :return: None
    """
    i = 0
    logger.info('Writing cropped tables...')
    table_paths = []
    text_path = None
    if cropped_tables is not None:
        for ct in cropped_tables:
            new_file_path = \
                os.path.join(temp_table_path, str(file_name),
                             'table_pag_{pag_num}_{c}.jpeg'.format(pag_num=page_number, c=i))
            ct = ct.convert('L')
            logger.info('Deskewing table...')
            sd = deskew.Deskew(
                input_numpy=np.asarray(ct),
                output_numpy=True
            )
            de_skewed_image_np = sd.run()
            logger.info('Deskew done')
            ct = Image.fromarray(de_skewed_image_np)
            ct = ct.convert(mode='L')
            try:
                ct.save(new_file_path, dpi=(EXTRACTION_DPI, EXTRACTION_DPI))
                logger.info('Image_{} wrote on disk'.format(new_file_path))
            except IOError or ValueError as e:
                raise OutputError('Cannot write image on disk: \n{}'.format(e))
            i += 1
            table_paths.append(new_file_path)
        logger.info('Writing cropped tables done.')
    else:
        logger.info('No tables to write on disk')

    if cropped_text is not None:
        logger.info('Writing cropped text...')
        # for cl in cropped_text:
        new_file_path = os.path.join(temp_text_path, str(file_name), 'text_pag_{}.jpeg'.format(page_number))
        # ct_l = cl.convert('L')
        try:
            cropped_text.save(new_file_path, dpi=(EXTRACTION_DPI, EXTRACTION_DPI))
            logger.info('Image_{} wrote on disk'.format(new_file_path))
        except IOError or ValueError as e:
            raise OutputError('Cannot write image on disk: \n{}'.format(e))
        # i += 1
        logger.info('Writing cropped text done.')
        text_path = new_file_path
    return table_paths, text_path

# def find_table(file_name, pil_image, create_temp_files=False, temp_table_path=TABLE_FOLDER, temp_text_path=TEXT_FOLDER):
#     """
#     useful only for batch. The function extract_tables_and_text does everything
#     :param file_name:
#     :param pil_image:
#     :param create_temp_files:
#     :param temp_table_path:
#     :param temp_text_path:
#     :return:
#     """
#     cropped_tables, cropped_text = extract_tables_and_text(pil_image=pil_image, inference_graph_path=PATH_TO_CKPT)
#     if create_temp_files:
#         create_temp_folders(file_name=file_name)
#         write_crops(
#             file_name=file_name,
#             cropped_tables=cropped_tables,
#             cropped_text=cropped_text
#         )
