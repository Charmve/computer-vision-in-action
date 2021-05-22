import os
from zipfile import ZipFile

with ZipFile('dataset.zip') as myzip:
    myzip.extractall()
    print('Extracting Complete.')

os.rename('hymenoptera_data', 'dataset')

os.remove('dataset/train/ants/imageNotFound.gif')
