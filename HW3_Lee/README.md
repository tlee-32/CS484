# CS 484 - Clustering Iris & Text Data HW3

# Install Requirements
Make sure you are using Python 3.6 to run this project.

`sudo pip3 install -r /requirements.txt`

# K-means clustering implementation
The implementation of k-means clustering algorithm is located in `/classifier/kmeans.py`

# Data Files
Ensure data files are placed in the `/data` directory before running the program.

# Running the Program
In the `/src` directory, the only file you need to run is `main.py`.

`python3 main.py`

# Results
The sentiment predictions will be written to `predicitions.data` located in `/data`.

# Saving and Loading Files `.pkl`
For faster run time, the pre-processed data is pickled (saved) in .pkl files and later loaded for re-use.
In `main.py`, `loadFile = True` if the file has already been saved and the `.pkl` file just needs to be loaded. If `loadFile = False`, then the `.data` file will be read and saved as a `.pkl` file.

# Cluster validation

# `/data` Directory and Subdirectories
Do not delete the`/data` directory as it will contain all of the files needed to run the sentiment classifier. Place your training data in the `/data/train` directory. 
