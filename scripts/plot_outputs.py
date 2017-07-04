import os
import json
import cv2
import numpy as np

out_dir = '/home/srq/Playground/unlv-train/test/outs'
images_dir = '/home/srq/Datasets/unlv'
post_bin = '/home/srq/Projects/TablesMachineLearning/cmake-build-debug/TablesMachineLearningPost'

for i in os.listdir(out_dir):
    if not i.endswith(".json"):
        continue

    id = os.path.splitext(i)[0]
    print(id)

    out_json_path = out_dir+'/'+i
    image_path = images_dir+'/'+id+'.png'
    out_image_path_gt = out_dir+'/'+id+'_gt.png'
    out_image_path_out = out_dir+'/'+id+'_out.png'
    out_image_path_post = out_dir+'/'+id+'_out_post.png'

    os.system(post_bin + ' ' + out_json_path + ' ' + image_path + ' ' + out_image_path_gt + ' ' + out_image_path_out + ' ' + out_image_path_post)