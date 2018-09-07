# IntelligentOCR
## An intelligent OCR that separates tables and text from documents producing csv files and text.
This project was developed to distinguish between tables and text inside insurance policies documents. For the first it produces `csv` tables for every table found, for the latter a `txt` file with text from outside the tables.

## General overview
The project uses a personalized neural network that has been trained in
[TableTrainNet](https://github.com/mawanda-jun/TableTrainNet)
, to distinguish between tables and text,
[Tesseract](https://github.com/tesseract-ocr/tesseract)
to do ocr in documents and 
[Tabula](https://github.com/tabulapdf/tabula-java)
to extract tables from recognized tables.


### Required libraries
Before we go on make sure you have everything installed to be able to use the project:
* [Tensorflow](https://www.tensorflow.org/)
* Python 3
* Pillow
* pandas
* numpy
* [pdftoppm](https://www.xpdfreader.com/pdftoppm-man.html)

In addition, a personalized version of
[alyn](https://github.com/mawanda-jun/Alyn)
has been made, so you can install it from repository or from folder `wheel/alyn-xxx.whl`

## Project pipeline
The project is made up of different parts that acts together as a pipeline. As a matter of fact a `pipeline.py` file has been made an it contains all the scripts that transforms a `TEST_PDF_PATH` pdf into `path/to/TABLE_FOLDER/file.csv`s  files and `path/to/TEXT_FOLDER/text.txt` text. So if you do not understand something take a look at it.

#### Take confidence with costants
The entire project can be manipulated changing only the `costants.py` file. More instructions are coming.

#### Read pdf and extract images from it
The reading of a pdf can be heavy: they can be a very large set of images so it is not clever to load it directly into memory. For this reason I decided to use 
[pdftoppm](https://www.xpdfreader.com/pdftoppm-man.html)
, access a page at a once to make a generator of pages and to "beautify" them before any further process.

`python extract_pages_from_pdf.py` will return a generator of pillow images.
To use it please modifying `costants.py`:
* `TEST_PDF_PATH`: /path/to/file.pdf you want to extract
* `TEMP_IMG_FOLDER_FROM_PDF`: /path/to/folder where to temporarly store the ppm extracted images. The script takes care to load `pdftoppm` one page at a once, but it has to store a ~30MB file that is immediately deleted;
* `PATH_TO_EXTRACTED_IMAGES`: /path/to/folder where to write the images extracted from pages. It is useful if some beautifying scripts are added and the user wants to see the result. If set to `None` it will not produce any output;
* `EXTRACTION_DPI`: int for quality output of images. Useful only if `PATH_TO_EXTRACTED_IMAGES` is not set to `None`.

A generator of pages is returned if everything went OK, instead if something went wrong `None`
is the result. A `.log` file is always produced so you can see what went wrong.

#### Find tables and text inside page
