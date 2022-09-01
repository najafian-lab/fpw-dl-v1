""" Simple script that downloads the dataset from google drive (or any place really) """
import os
import traceback
from urllib import request
import zipfile

import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from docker import ensure_dataset, IN_DOCKER

# change this to any zip
BASE_URL = 'https://docs.google.com/uc?export=download'
DOWNLOAD_ID = '1bAQLG-5c1JxkwHm8ttPqh-I7JjfSEJYG'
DATASET_FILE = 'dataset.zip'

# make sure dataset folder has been provided
ensure_dataset()


def get_confirm_token(response):
    """ Determine if large file confirmation. Specifically for google """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return 'GET', value
    
    # if nothing was found in the cookies let's see if it's html and try to get the code from the form
    html = False
    if 'content-type' in response.headers:  # usually case insensitive but let's be sure
        html = 'text/html' in response.headers['content-type'].lower()
    elif 'Content-Type' in response.headers:
        html = 'text/html' in response.headers['Content-Type'].lower()

    # check the html
    if html:
        sp = BeautifulSoup(response.text, 'lxml')
        for item in sp.find_all(id='downloadForm'):
            if 'action' in item.attrs:  # get the action url if available
                return 'POST', item.attrs['action']

    return 'GET', None


# as described here https://sumit-ghosh.com/articles/python-download-progress-bar/
# and https://github.com/nsadawi/Download-Large-File-From-Google-Drive-Using-Python/blob/master/Download-Large-File-from-Google-Drive.ipynb
def download(id, filename):
    session = requests.Session()
    response = session.get(BASE_URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    # if it's a large file request (usually over 16Gb)
    if token[1] is not None:
        print('Found large file token...')

        if token[0] == 'GET':
            params = {'id': id, 'confirm': token}
            response = session.get(BASE_URL, params=params, stream=True)
        else:
            # fake browser form submission
            headers = {'User-Agent': 'Mozilla/5.0'}
            payload = {'submit': True}
            response = session.post(
                token[1],  # the action url
                headers=headers,
                data=payload,
                stream=True
            )
    else:
        print('No large file token found')

    if response.status_code != 200:
        print('Got a non 200 status code!')
        exit(1)

    # save the file
    with open(filename, 'wb') as f:
        total = response.headers.get('content-length')
        print(f'Got content length of {total}')

        if total is None:
            f.write(response.content)
        else:
            total = int(total)
            chunk_size = int(max(int(total/1000), 1024*1024))
            for data in tqdm(response.iter_content(chunk_size=chunk_size), desc='Downloading', total=int(total / chunk_size), unit='chunk', colour='green'):
                f.write(data)


if __name__ == '__main__':
    if os.path.isfile(DATASET_FILE):
        print(f'Dataset file {DATASET_FILE} already exists! Skipping download')
    else:
        print('Downloading dataset...')
        try:
            download(DOWNLOAD_ID, DATASET_FILE)
        except Exception as err:
            print(f'Failed to download {str(err)}')
            traceback.print_exc()
            print('Exiting...')
            exit(1)
        print('Finished downloading')

    print('Unzipping dataset...')
    try:
        with zipfile.ZipFile(DATASET_FILE, 'r') as zip_ref:
            # in the docker instance we want to extract to /dataset (which means current directory will be root)
            if IN_DOCKER:
                c_dir = '/'
            else:
                c_dir = os.path.dirname(os.path.realpath(__file__))
            for item in tqdm(zip_ref.infolist(), desc='Extracting'):
                try:
                    zip_ref.extract(item, c_dir)
                except zipfile.error as e:
                    print(f'Failed to extract member {str(e)}')
    except Exception as err:
        print(f'Failed to extract zip {str(err)}')
        traceback.print_exc()
        print('Exiting...')
        exit(1)
    print('Done!')
