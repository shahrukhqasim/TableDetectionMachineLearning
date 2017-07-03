import os

path = '/home/srq/Playground/unlv'

for i in os.listdir(path):
    id = os.path.splitext(i)[0]
    print(id)
    image_path = path + '/' + id + '.png'
    gt_path = path + '/unlv_xml_gt/' + id + '.json'
    ocr_path = path + '/unlv_xml_ocr/' + id + '.json'
    print(image_path, gt_path, ocr_path)
    print(os.path.isfile(image_path),os.path.isfile(gt_path),os.path.isfile(ocr_path))
    os.system("/home/srq/Projects/TablesMachineLearning/cmake-build-debug/TablesMachineLearning " + image_path+ ' '+ ocr_path + ' ' + gt_path)