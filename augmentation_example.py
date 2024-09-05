import os

import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A

KEYPOINT_COLOR = (0, 255, 0) # Green

image_dir = r'C:\\4022term\\Tello\\Yolo\\Gates_first_annotated\\train\\images'
label_dir = r'C:\\4022term\\Tello\\Yolo\\Gates_first_annotated\\train\\labels'
aug_image_dir = r'C:\\4022term\\Tello\\Yolo\\Gates_first_annotated\\train\\aug_images'
aug_label_dir = r'C:\\4022term\\Tello\\Yolo\\Gates_first_annotated\\train\\aug_labels'
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_cent, y_cent, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    x_min, x_max, y_min, y_max = int((x_cent-w/2)*640), int((x_cent+w/2)*640), int((y_cent-h/2)*640), int((y_cent + h/2)*640)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=15):
    image = image.copy()

    for (x, y, v) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(image)
    plt.show()

def read_annotations(label_path):
    annotations = []
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            keypoints = [(float(parts[i]), float(parts[i + 1]), int(parts[i + 2])) for i in range(5, len(parts), 3)]
            annotations.append((class_id, x_center, y_center, width, height, keypoints))
    return annotations

def save_keypoints(label_path, annotations):
    with open(label_path, 'w') as file:
        for ann in annotations:
            class_id, x_center, y_center, width, height, keypoints = ann
            keypoints_flat = ''
            # print (keypoints)
            for i in range(len(keypoints)):
                keypoints_flat = keypoints_flat + str(keypoints[i][0]/640.0) + ' ' + str(keypoints[i][1]/640.0) + ' ' + str(keypoints[i][2]) + ' '
            # keypoints_flat = ' '.join(f'{' for i in range(len(keypoints)) for j in range(3))
            file.write(f"{class_id} {x_center} {y_center} {width} {height} {keypoints_flat[:-1]}\n")
            # print(f"{class_id} {x_center} {y_center} {width} {height} {keypoints_flat}")
            # print("----------------------------------")

# random.seed(7)
transform = A.Compose([
        A.RandomSizedCrop(min_max_height=(520, 640), height=640, width=640, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.HueSaturationValue(p=0.5),
            A.RGBShift(p=0.7)
        ], p=1),
        A.RandomBrightnessContrast(p=0.5)
    ],
    keypoint_params=A.KeypointParams(format='xy'),
    bbox_params=A.BboxParams(format='yolo')
)
category_ids = [0]
category_id_to_name = {0: 'gate'}


#-------------------------------       one picture example


aug_label_path = r"C:\4022term\Parrot\DataSets\Gate\train\labels\IMG_8457_png.rf.b959bcf7662a13204ac3c8b377acb32d_aug_2.txt"
aug_image_path = r"C:\4022term\Parrot\DataSets\Gate\train\images\IMG_8457_png.rf.b959bcf7662a13204ac3c8b377acb32d_aug_2.jpg"

img_annotations = read_annotations(aug_label_path)
print(img_annotations[0])
class_id, x_center, y_center, width, height, kp = img_annotations[0]

# keypoints = [(x*640, y*640) for x, y, _ in img_annotations[2][5]]
bbox = [[x_center, y_center, width, height, class_id]]
# visiblity = [v for _, _, v in img_annotations[2][5]]
# print(bbox)

image = cv2.imread(aug_image_path)
# cv2.imshow("read image", image)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("bbox ====", (int((bbox[0][0] - bbox[0][2]/2)*640) , int((bbox[0][1]-bbox[0][3]/2)*640)), (int((bbox[0][0]+bbox[0][2]/2)*640) , int((bbox[0][1]+bbox[0][3]/2)*640)))
annotated_frame = cv2.rectangle(image,(int((bbox[0][0] - bbox[0][2]/2)*640) , int((bbox[0][1]-bbox[0][3]/2)*640)), (int((bbox[0][0]+bbox[0][2]/2)*640) , int((bbox[0][1]+bbox[0][3]/2)*640)), (0, 255, 0), 3)
cv2.imshow("annotated_frame", annotated_frame)
cv2.waitKey(0)

# transformed = transform(image=image, keypoints=keypoints, bboxes = bbox)

# cv2.imwrite(aug_image_path, cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))

# keypoints_new = [keypoints[i] + (visiblity[i],) for i in range(len(visiblity))]

# bbox_new = transformed['bboxes'][0]

# save_keypoints(aug_label_path, [(bbox_new[4], bbox_new[0], bbox_new[1], bbox_new[2], bbox_new[3], keypoints_new)])

# vis_keypoints(image, keypoints_new)

# visualize_bbox(image, bbox, category_ids )

# visualize(image, bbox, category_ids, category_id_to_name)