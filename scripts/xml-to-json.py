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

print("Hello", name)

doc = minidom.parse(name);
words = doc.getElementsByTagName('word')

data_all = [];

for i in words:
    data_word = {}
    word_text = i.firstChild.nodeValue.strip()
    left = i.getAttribute('left')
    top = i.getAttribute('top')
    right = i.getAttribute('right')
    bottom = i.getAttribute('bottom')
    print(left, top, right, bottom, word_text)
    data_word['word'] = word_text
    data_word['left'] = int(left)
    data_word['top'] = int(top)
    data_word['right'] = int(right)
    data_word['bottom'] = int(bottom)
    data_all.append(data_word)


new_path = os.path.splitext(name)[0]+'.json';

with open(new_path, 'w') as outfile:
    json.dump(data_all, outfile)