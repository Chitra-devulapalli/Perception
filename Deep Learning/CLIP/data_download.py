import os 
os.environ['KAGGLE_USERNAME'] = "XXXX"
os.environ['KAGGLE_KEY'] = "XXXX"

import zipfile

# Set up the environment for Kaggle API credentials
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')

# Download the dataset
os.system('kaggle datasets download -d adityajn105/flickr8k')

# Unzip the dataset
with zipfile.ZipFile('flickr8k.zip', 'r') as zip_ref:
    zip_ref.extractall('8k_dataset')

dataset = "8k_dataset"