import os
import xml.etree.ElementTree as ET
import sys

traindata_gtpath = '/home/jayce/datasets/oil_defect_project/aug_1_split/train_xml/'# os.path.join(sys.path[1], 'dataset/oil/labels/train_xml/')     # '/home/jayce/datasets/2021/train/box/'
traindata_gtnamelist = os.listdir(traindata_gtpath)
traindata_num = len(traindata_gtnamelist)
valdata_gtpath = '/home/jayce/datasets/oil_defect_project/aug_1_split/val_xml/'# os.path.join(sys.path[1], 'dataset/oil/labels/val_xml/')
valdata_gtnamelist = os.listdir(valdata_gtpath)
valdata_num = len(valdata_gtnamelist)
trainsave_path = '/home/jayce/datasets/oil_defect_project/aug_1_split/yolo_label/train/' # os.path.join(sys.path[1], 'dataset/oil/labels/train/')
valsave_path = '/home/jayce/datasets/oil_defect_project/aug_1_split/yolo_label/val/'  # os.path.join(sys.path[1], 'dataset/oil/labels/val/')

# traindata_gtpath = '/home/jayce/datasets/oil_labels/pascal_voc/rectify1/outputs/'    # '/home/jayce/datasets/2021/train/box/'
# traindata_gtnamelist = os.listdir(traindata_gtpath)
# traindata_num = len(traindata_gtnamelist)
# label_save_path = '/home/jayce/datasets/oil_labels/yolo5notation/rectify1_yolo/'
def name_to_label(name):
    numbers = {
        'stain' : str(0),
        'flaw': str(1),
        'burn': str(2),

    }
    return numbers.get(name, None)


for i in range(traindata_num):
    f = open(trainsave_path + traindata_gtnamelist[i][:-4] + ".txt", 'a')

    gtpath = traindata_gtpath + traindata_gtnamelist[i]
    print(gtpath)
    gt = ET.parse(gtpath).getroot()
    if gt.find('size') == None:
        continue
    img_width = int(gt.find('size').find('width').text)
    img_height = int(gt.find('size').find('height').text)

    object = gt.iter('object')
    for i in object:
        name = name_to_label(i.find('name').text)
        if (name != '4'):
            box = i.find('bndbox')
            center_x = str(((int(box.find('xmax').text) + int(box.find('xmin').text)) / 2) / img_width)
            center_y = str(((int(box.find('ymax').text) + int(box.find('ymin').text)) / 2) / img_height)
            box_width = str(((int(box.find('xmax').text) - int(box.find('xmin').text))) / img_width)
            box_height = str(((int(box.find('ymax').text) - int(box.find('ymin').text))) / img_height)

            text = name + ' ' + center_x + ' ' + center_y + ' ' + box_width + ' ' + box_height + '\n'
            # text = name + ' ' + center_x + ' ' + center_y + ' ' + box_width + ' ' + box_height + '\n'
            # print(text)
            # f.writelines([name, center_x, center_y, box_width, box_height])
            f.write(text)
    f.close()

for i in range(valdata_num):
    f = open(valsave_path + valdata_gtnamelist[i][:-4] + ".txt", 'a')

    gtpath = valdata_gtpath + valdata_gtnamelist[i]
    gt = ET.parse(gtpath).getroot()
    if gt.find('size') == None:
        continue
    img_width = int(gt.find('size').find('width').text)
    img_height = int(gt.find('size').find('height').text)

    object = gt.iter('object')
    for i in object:
        name = name_to_label(i.find('name').text)
        if (name != '4'):
            box = i.find('bndbox')
            center_x = str(((int(box.find('xmax').text) + int(box.find('xmin').text)) / 2) / img_width)
            center_y = str(((int(box.find('ymax').text) + int(box.find('ymin').text)) / 2) / img_height)
            box_width = str(((int(box.find('xmax').text) - int(box.find('xmin').text))) / img_width)
            box_height = str(((int(box.find('ymax').text) - int(box.find('ymin').text))) / img_height)

            text = name + ' ' + center_x + ' ' + center_y + ' ' + box_width + ' ' + box_height + '\n'
            # text = name + ' ' + center_x + ' ' + center_y + ' ' + box_width + ' ' + box_height + '\n'
            print(text)
            # f.writelines([name, center_x, center_y, box_width, box_height])
            f.write(text)
    f.close()