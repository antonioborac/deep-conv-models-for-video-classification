This repository contains code on which my master thesis is based. The title of the thesis is Convolutional models for video classification.

'slow-fast' directory contains Jupiter Notebooks and all needed python files to run experiments on 'slow-fast' architecture.
'pooling' directory contains Jupiter Notebooks and all needed python files to run experiments on 'feature-pooling' architecture.
'data' directory contains all code that is used for data preparation and for data generation.

To be able to reproduce the experiments, the data needs to be prepared. Dataset and dataset splits need to be downloaded from https://www.crcv.ucf.edu/data/UCF101.php.
The data should be extracted in directory named 'Videos' and the splits should be in a directory named 'ucfTrainTestlist'. 
The next step is to prepare and start the script 'data/prepare-data.ipynb'. These paths need to be adjusted: data_directory - the directory which contains extracted data
and splits and the current directory in which the code is run. The code should split test and train data in two different folders which is important for the data generator.

The next step is to run appropriate Jupiter Notebooks to run the experiments. All notebooks have the same pattern. First, the code for data preparation is executed.
After that part, the setup differs for further training of already trained models and for models that are trained from scratch.