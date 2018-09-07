# FAQ for IntelligentOCR project. (Why the hell did you do what you did?)

## Extraction of pdf into memory
### Why calling command-line scripts instead of using wrappers?
This script has been thought inside a server with quad-core processor and 4GB of RAM
full of running processes. Then my focus has been to RAM saving. 
This is why I preferred to use the CL script so I could process a page at a 
once - and manage them as a generator of pillow images instead of a list.

### Do images need to be beautiful?
Yes, they do. In fact, every OCR program has this pre-processing part. 
However, using neural networks and deep learning to find tables led me to 
personally manage this part. For this reason I introduced a de-skewing of all pages 
before feeding the NN.

More steps are coming: the next one is de-noising of dirty backgrounds - maybe
introducing 
[these](https://www.kaggle.com/c/denoising-dirty-documents/kernels)
projects.

