import numpy as np
import cv2
import os
from xml.dom import minidom
import sys
import csv
import json

# if len(sys.argv != 2):
#     print("Oops")
#     exit(-1)

name = sys.argv[1]

doc = minidom.parse(name);
words = doc.getElementsByTagName('Table')

data_all = [];

for i in words:
    data_table = {}
    word_text = i.firstChild.nodeValue.strip()
    left = i.getAttribute('x0')
    top = i.getAttribute('y0')
    right = i.getAttribute('x1')
    bottom = i.getAttribute('y1')
    data_table['left'] = int(left)
    data_table['top'] = int(top)
    data_table['right'] = int(right)
    data_table['bottom'] = int(bottom)
    data_all.append(data_table)


new_path = os.path.splitext(name)[0]+'.json';

with open(new_path, 'w') as outfile:
    json.dump(data_all, outfile)