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
* [Tesseract](https://github.com/tesseract-ocr/tesseract)
* pytesseract
* PIL (pillow)
* pandas
* numpy
* pdftoppm
* [Tabula](https://github.com/tabulapdf/tabula-java) and its wrapper tabula-py

In addition, a personalized version of
[alyn](https://github.com/mawanda-jun/Alyn)
has been made, so you can install it from repository or from folder `wheel/alyn-xxx.whl`

## Project pipeline
The project is made up of different parts that acts together as a pipeline. As a matter of fact a `pipeline.py` 
file has been made an it contains all the scripts that transforms a 
`TEST_PDF_PATH` pdf into `path/to/TABLE_FOLDER/file.csv`s  files and 
`path/to/TEXT_FOLDER/text.txt` text using a `INFERENCE_GRAPH`. So if you just want to have the job done, modify those 
costants and run the project with `python pipeline.py`. 

If you want to learn more about the motivation behind every project decision please take a look at `FAQ.md`
file.

Lastly, if you want to take a look at the code...
_Go down The Rabbit Hole!_

#### Take confidence with costants
The entire project can be manipulated changing only the `costants.py` file. More instructions are coming.

#### Read pdf and extract images from it
The reading of a pdf can be heavy: they can be a very large set of images so it is not clever to load it directly into
memory. For this reason I decided to use `pdftoppm`, access a page at a once to make a generator of pages and to 
"beautify" them before any further process.

`python extract_pages_from_pdf.py` will return a generator of pillow images.
To use it please modifying `costants.py`:
* `TEST_PDF_PATH`: /path/to/file.pdf you want to extract
* `TEMP_IMG_FOLDER_FROM_PDF`: /path/to/folder where to temporarly store the ppm extracted images. The script takes 
care to load `pdftoppm` one page at a once, but it has to store a ~30MB file that is immediately deleted;
* `PATH_TO_EXTRACTED_IMAGES`: /path/to/folder where to write the images extracted from pages. It is useful if 
some beautifying scripts are added and the user wants to see the result. If set to `None` it will not produce any output;
* `EXTRACTION_DPI`: int for quality output of images. Useful only if `PATH_TO_EXTRACTED_IMAGES` is not set to `None`.

A generator of pages is returned if everything went OK, instead if something went wrong `None`
is the result. A `.log` file is always produced so you can see what went wrong.

#### Find tables and text inside page
This part takes as input a generator of Pillow images and a inference graph and returns two values:
* List of tables, which are Pillow images cropped from original pages
* A Pillow image which is the merge of what was not cropped

The inference is done in four parts:
1. First of all we have to find all the boxes with which the neural network says where the tables are and with which score;
2. Analyze the scores to understand which are the best one;
3. Interpret and merge the correct boxes;
4. Crop original images and separate them into the two groups mentioned above.

This part is made with `find_table.py`, in particular from `extract_tables_and_text` function
and it uses those costants:
* `MAX_NUM_BOXES`: max number of boxes to be considered before merge;
* `MIN_SCORE`: minimum score of boxes to be considered before merge.

#### Extract tables from table images
Since I needed to reconstruct the tables structure I found that tabula was good to make the job done.
Unfortunately the python wrapper takes a pdf searchable file and outputs a csv file for every table found.
For this reason I need to create a searchable pdf file before proceeding in getting table structure.

Now that we have the cropped images of tables, we can process them to get the structure and the data.

We proceed to make OCR with Tesseract on image and to export a searchable pdf. Unfortunately
its wrapper has no options to export file as pdf, so I needed to use the CL commands instead.
We can manage this part with:
* `TABLE_FOLDER`: /path/where/to/save/pdf/and/csv/files;
* `TEST_TABLE_PATH`: /path/to/file.jpg from which to take the image to process.

This will create a pdf file with the table and text recognized from it and a csv file with
the table informations.

#### Extract text from text images
This part is the simplest one, since it simple take a pillow image and transform it to text.

Use `do_ocr_to_text` from `tesseract_on_text.py` and customize costant:
* `TEXT_FOLDER`: /path/to/folder in which to save text extracted from file.




---
---
<sup>1</sup>. This is not yet implemented due to some problems with numpy arrays. Even if the training was done
with this method, the inference seems not to understand anything from modified images - 2018/09/10
 