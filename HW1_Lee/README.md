# CS 484 - Predict Sentiment w/ kNN (HW1)

# Install Requirements
Make sure you are using Python 3.6 to run this project.

`sudo pip3 install -r /requirements.txt`

# Training and Test Files
Ensure `train.data` and `test.data` files are placed in `/data/train` and `/data/test` directories before running the program.

# Running the Program
In the `/src` directory, the only file you need to run is `sentimentclassifier.py`.

`python3 sentimentclassifier.py`

# Results
The sentiment predictions will be written to `predicitions.data` located in `/data/predictions`.

# Saving and Loading Files `.pkl` `.model`
A checkpoint is made for each stage of the pipeline:
Pre-processing -> Feature extraction/representation -> Cross-validation -> kNN classification

Checkpoints mean that data is pickled (saved) as a .pkl or .model file. In `sentimentclassifier.py`, the condition
`loadFile = True` if the file has already been saved and the `.pkl` file just needs to be loaded. If `loadFile = False`, then the `.data` file will be read and saved as a `.pkl` file.

For the Doc2Vec model, if `retrain=True` in `sentimentclassifier.py`, then the Doc2Vec model is retrained and saved as a `.model` file (may take several minutes). If `retrain=False`, then the Doc2Vec model is loaded through the `doc2vec.model` file.

# Cross-validation and Optimizing k
To run a k-fold cross-validation, call the methods from `crossvalidation.py` in `sentimentclassifier.py`  so that training data can be fed as input.

# `/data` Directory and Subdirectories
Do not delete the`/data` directory as it will contain all of the files needed to run the sentiment classifier. Place your training data in the `/data/train` directory and test data in the `/data/test/` directory.
