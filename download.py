""" Simple script that downloads the dataset from google drive (or any place really) """
import os
import traceback
import zipfile

import requests
from tqdm import tqdm

# change this to any zip
BASE_URL = 'https://docs.google.com/uc?export=download'
DOWNLOAD_ID = '1bAQLG-5c1JxkwHm8ttPqh-I7JjfSEJYG'
DATASET_FILE = 'dataset.zip'


def get_confirm_token(response):
    """ Determine if large file confirmation. Specifically for google """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


# as described here https://sumit-ghosh.com/articles/python-download-progress-bar/
# and https://github.com/nsadawi/Download-Large-File-From-Google-Drive-Using-Python/blob/master/Download-Large-File-from-Google-Drive.ipynb
def download(id, filename):
    session = requests.Session()
    response = session.get(BASE_URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    # if it's a large file request (usually over 16Gb)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(BASE_URL, params=params, stream=True)

    # save the file
    with open(filename, 'wb') as f:
        total = response.headers.get('content-length')

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
