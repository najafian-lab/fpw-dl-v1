""" Docker utility functions that help map functions/utilities in the scripts more easily """
from colorama import Fore, Back, Style
import os
import sys

# get bool of if this is a docker enviroment
CURRENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
IN_DOCKER = os.environ.get('IN_DOCKER', False)
if IN_DOCKER is not None:
    if isinstance(IN_DOCKER, str):
        IN_DOCKER = IN_DOCKER.lower().strip()
        IN_DOCKER = (IN_DOCKER == '1') or (IN_DOCKER.startswith('t'))
    else:
        IN_DOCKER = bool(IN_DOCKER)

# specify the input and output directories
DATASET_DIR = os.environ.get('DATASET_DIR', '/dataset' if IN_DOCKER else os.path.join(CURRENT_DIR, 'dataset'))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/output' if IN_DOCKER else os.path.join(CURRENT_DIR, 'output'))

# ensure the directories are specified
def ensure_dir(path, _create=False, _exit=True):
    """ Ensures that the specified directory is required and mounted unless specified otherwise """
    if not os.path.isdir(path):
        print(f'{Fore.YELLOW}The folder {path} does not exist {Fore.RESET}')

        if IN_DOCKER:
            print(f'{Fore.BLUE}Please use the docker option -v <absolute path to your dataset directory>:{path} to mount a dataset folder to the docker image! {Fore.RESET}')

        if _create:
            print(f'{Fore.GREEN}Creating directory {path}! {Fore.RESET}')
            os.makedirs(path, exist_ok=True)
        elif _exit:
            sys.exit(1)


def ensure_dataset(path=DATASET_DIR):
    """ Ensure that the dataset directory is existant/cannot be created """
    ensure_dir(path, _create=False, _exit=True)


def ensure_output(path=OUTPUT_DIR):
    """ Ensure that the output folder is existant and the output CANNOT be created in docker as that's pointless """
    ensure_dir(path, _create=(not IN_DOCKER), _exit=(IN_DOCKER))


def ensure_dataset_and_output(inpath=DATASET_DIR, outpath=OUTPUT_DIR):
    """ Ensure the dataset and output directories are properly defined """
    ensure_dataset(inpath)
    ensure_output(outpath)
