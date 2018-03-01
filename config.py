import os

# Home directory for the project
# PROJECT_HOME = os.getcwd()
PROJECT_HOME = '.'

# A folder in which all experiments are stored in subfolders
EXPERIMENT_DIR = os.path.join(PROJECT_HOME, '../rnn-experiment-storage')

# A folder in which all time experiments are stored in subfolders
TIME_EXPERIMENT_DIR = os.path.join(PROJECT_HOME, '../rnn-experiment-storage/time')

# A folder with a subfolder for each dataset in which 3 files are found: [train;valid;text].txt
DATA_DIR = os.path.join(PROJECT_HOME, 'datasets')

# A folder containing .json file with alphabet specifications - see the Alphabet class
ALPHABETS_DIR = os.path.join(PROJECT_HOME, 'alphabets')

# All experiment folders should have file with this name inside
EXPERIMENT_SPEC_FILENAME = 'spec.json'
RESUME_EXPERIMENT_SPEC_FILENAME = 'resume_spec.json'


# For seeding RNGs while training
RANDOM_SEED = 15243

# 2GB memory limit - for loading text files
DEFAULT_MEMORY_LIMIT_BYTES = int(2e9)