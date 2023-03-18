# CS432_Project
Final project for CS432 Computer Vision for Education and Social Science Research
Authors: 

## Authors: Micael Tchapmi

## A. Running the code

## Clone Repository
    git clone 

## Data Preprocessing
    Download data from Kaggle [here](https://www.kaggle.com/datasets/aryaminus/electronic-components) and place into a data/images directory in the working directory. Then run:

    python preprocess_data.py

### Create Virtual Environment
    $PYTHON_BIN = path/to/python/bin
    virtualenv -p $PYTHON_BIN venv
    source activate venv/bin/activate
   
### Install requirements
    pip install -r requirements.txt

## Run clip inference
    python CLIP.py

### Training/Testing a network
    1. Set parameters in run.sh such as model name, batch size, etc
    2. Run: ./run.sh
    3. Results are saved in './results' directory
    
### Plotting loss after training
    python tools/plot_loss.py path/to/results/trainlog.txt

### Visualizing Predictions on Images
    1. Run evaluation by setting train=0 in run.sh
    2. Predictions are stored in path/to/results/modelname/Images