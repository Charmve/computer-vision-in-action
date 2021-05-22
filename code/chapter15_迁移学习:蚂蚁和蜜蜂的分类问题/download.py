from urllib import request


DATA_URL = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'


def _progress(block_num, block_size, total_size):
    '''回调函数
       @block_num: 已经下载的数据块
       @block_size: 数据块的大小
       @total_size: 远程文件的大小
    '''
    print('>> Downloading %s %.1f%%' % ('dataset.zip', float(
        block_num * block_size) / float(total_size) * 100.0), end='\r')


filepath, _ = request.urlretrieve(DATA_URL, 'dataset.zip', _progress)
print()
