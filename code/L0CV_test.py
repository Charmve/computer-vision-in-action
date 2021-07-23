# Computer Vison in Action (L0CV)
#
# Copyright 2017 Fraunhofer FKIE
#
# L0CV is free software: you can redistribute it and/or modify
# it under the terms of the Apache 2.0 License as published
# by the Free Software Foundation, either version 2.0 of the License, or
# (at your option) any later version.
#
# L0CV is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License for more details.
#
# You should have received a copy of the Apache License
# along with IVA in the COPYING and COPYING.LESSER files.
# If not, see <https://www.apache.org/licenses/LICENSE-2.0>.


import torch
import torchvision
import sys

import L0CV  ## 还有bug #18


asciichart = '\n   ______ __________________    __\n   ___  / __  __ \_  ____/_ |  / /\n   __  /  _  / / /  /    __ | / / \n   _  /___/ /_/ // /___  __ |/ /  \n   /_____/\____/ \____/  _____/   \n\n    Computer Vision in Action\n                          By Charmve\n'


def main():
    printAsciiChart() # logo chat

    sys.path.append("..") 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 均已测试
    
    temp = "%-15s %-15s %15s" 
    print(temp % ("device", "torch version", "L0CV version"))
    print(temp % (device, torch.__version__, L0CV.__version__))
    
    return 0

def printAsciiChart():
    print(asciichart)
    
if __name__ == '__main__':
    sys.exit(main())

