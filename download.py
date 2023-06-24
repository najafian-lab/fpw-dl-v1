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
# DOWNLOAD_ID = '1bAQLG-5c1JxkwHm8ttPqh-I7JjfSEJYG'

OUTPUT_FOLDER = 'dataset'  # folder to place it in
DOWNLOAD_PARTS = [   # pairs of google drive ids and zip names
('1SuKZzpAWTMxpLerDY9nTYCXvUvjAPqtI', 'fabry.zip'),
('1ImfTLW6vq9knE8ZasBMdouh7Yzd9je6y', 'normal.zip'),
('1hsUTb0-yJmAwYc8usNoMaCNG5cOLF9ns', 'eval.zip'),
('1BXWVqaQTN-qBZhdO46UW1AbqzCLj1hs3', 'dkd.zip'),
('1oN81ruWD1gYCzx9E0edHzv_tVi4MxZCa', 'fabry-f.zip'),
('14YWpgZL_bphFC-0Yaz-R6FPqMuBLE-VA', 'mcdfsgs.zip')
]
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"} 

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

    print('Is-Html', html)

    # check the html
    if html:
        sp = BeautifulSoup(response.text, 'lxml')
        for item in list(sp.find_all(id='downloadForm')) + list(sp.find_all(id='download-form')) + list(sp.find_all(id='downloadform')) + list(sp.find_all(id='download_form')):  # they seem to change it from time to time
            if 'action' in item.attrs:  # get the action url if available
                return 'POST', item.attrs['action']

        quota_str = str(sp).lower()
        if 'quota' in quota_str and ('exceeded' in quota_str or 'reached' in quota_str):
            print('Google Drive Quota limit reached. Probably too many people are downloading this file right now...')
            exit(1)

    return 'GET', None


# as described here https://sumit-ghosh.com/articles/python-download-progress-bar/
# and https://github.com/nsadawi/Download-Large-File-From-Google-Drive-Using-Python/blob/master/Download-Large-File-from-Google-Drive.ipynb
def download(id, filename):
    session = requests.Session()
    response = session.get(BASE_URL, params={'id': id}, stream=True, headers=HEADERS)
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
        print('Got a non 200 status code! Please try again later')
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
    print('Downloading all', len(DOWNLOAD_PARTS), 'dataset parts. Please be patient as some are larger than others')
    for ind, (gid, name) in tqdm(enumerate(DOWNLOAD_PARTS), desc='Overall Parts', total=len(DOWNLOAD_PARTS), colour='blue', unit='part'):
        print()
        try_again = True
        throw_error_next = False
        while try_again:
            if os.path.isfile(name):
                print(f'Dataset file {name} already exists! Skipping download. To force redownload delete the {name}.zip file')
            else:
                print(f'Downloading dataset {name}... Part {ind + 1}/{len(DOWNLOAD_PARTS)}')
                
                try:
                    download(gid, name)
                except Exception as err:
                    if not throw_error_next:
                        print(f'Failed to download {str(err)}... trying again')
                        throw_error_next = True  # throw error next time
                        continue
                    else:
                        print(f'Failed to download {str(err)}')
                        traceback.print_exc()
                        # try_again = False
                        print('Exiting...')
                        exit(1)
                print('Finished downloading')
            
            print(f'Unzipping dataset {name}... Part {ind + 1}/{len(DOWNLOAD_PARTS)}')
            try:
                with zipfile.ZipFile(name, 'r') as zip_ref:
                    # in the docker instance we want to extract to /dataset (which means current directory will be root)
                    if IN_DOCKER:
                        c_dir = f'/{OUTPUT_FOLDER}'
                    else:
                        c_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), OUTPUT_FOLDER)
                    for item in tqdm(zip_ref.infolist(), desc='Extracting'):
                        try:
                            zip_ref.extract(item, c_dir)
                        except zipfile.error as e:
                            print(f'Failed to extract member {str(e)}')
                try_again = False  # exit
            except Exception as err:
                if not throw_error_next:
                    print(f'Failed to download {str(err)}... Deleting zip and trying again')
                    os.remove(name)
                    try_again = True
                    throw_error_next = True
                else:
                    print(f'Failed to extract zip {str(err)}')
                    traceback.print_exc()
                    # try_again = False  # exit
                    print('Exiting...')
                    exit(1)
        
        print('\n------------------------------\n\n')
    print('Done!')
