#!/usr/bin/env python3

from setuptools import setup, find_packages
import L0CV

requirements = [
    'jupyter',
    'numpy',
    'pytorch'
    'matplotlib',
    'requests',
    'pandas'
]

# requirements = open('requirements.txt').readlines() # 更全依赖文件

print(L0CV.__version__)

setup(
    name='L0CV',
    version=L0CV.__version__,
    python_requires='>=3.5',
    author='ZHANG WEI (Charmve)',
    author_email='yidazhang1@gmail.com',
    url='https://github.com/Charmve/computer-vision-in-action',
    description='Computer Vision in Action',
    license='GPL2.0',
    packages=find_packages(),
    zip_safe=True,
    install_requires=requirements,
)
