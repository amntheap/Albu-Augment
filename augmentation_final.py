import os

import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A

KEYPOINT_COLOR = (0, 255, 0) # Green

ret = 0

keypoint_classes = [
    'right',
    'left'
]

bbox_classes = ["gate_down",
                "gate_up"
                ]

image_dir = r'C:\4022term\Parrot\DataSets\Gate\valid\images'
label_dir = r'C:\4022term\Parrot\DataSets\Gate\valid\labels'
aug_image_dir = r'C:\4022term\Parrot\DataSets\Gate\train_aug\images'
aug_label_dir = r'C:\4022term\Parrot\DataSets\Gate\train_aug\labels'
aug_image_dir_trash = r'C:\4022term\Parrot\DataSets\Gate\train_aug\trash'
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

        A.CLAHE(
            clip_limit=4.0,  # Union[float, Tuple[float, float]]
            tile_grid_size=(8, 8),  # Tuple[int, int]
            always_apply=False,  # bool
            p=0.5,  # float
        ),


        A.OneOf([
            A.Blur(
                blur_limit=3,  # Union[int, Tuple[int, int]]
                always_apply=False,  # bool
                p=0.3,  # float
                ),
            A.GaussianBlur(
                blur_limit=(1, 3),  # Union[int, Tuple[int, int]]
                sigma_limit=10,  # Union[float, Tuple[float, float]]
                always_apply=False,  # bool
                p=1.0,  # float
                )
        ]),


        A.OneOf([
            A.HueSaturationValue(p=0.5),
            A.RGBShift(p=0.7)
        ], p=1),
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),  # Tuple[float, float, float, float]
            angle_lower=0,  # float
            angle_upper=1,  # float
            num_flare_circles_lower=10,  # int
            num_flare_circles_upper=40,  # int
            src_radius=150,  # int
            src_color=(255, 255, 255),  # Tuple[int, int, int]
            always_apply=False,  # bool
            p=0.5,  # float
        ),
        A.RandomBrightnessContrast(p=0.5)
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, label_fields=['keypoint_classes'], ),
    bbox_params=A.BboxParams(format='yolo'),
    additional_targets= { 'keypoints1': 'keypoints',
                        'keypoints2': 'keypoints',
                        'keypoints3': 'keypoints',
                        'keypoints4': 'keypoints',
                        'keypoints4': 'keypoints',
                        'keypoints4': 'keypoints',
                        'keypoints5': 'keypoints',
                        'keypoints6': 'keypoints',
                        'keypoints7': 'keypoints',
                        'keypoints8': 'keypoints',
                        'keypoints9': 'keypoints',
                        'keypoints10': 'keypoints',
                        'keypoints11': 'keypoints',
                        'keypoints12': 'keypoints',
                        'keypoints13': 'keypoints',
                        'keypoints14': 'keypoints',
                        'keypoints15': 'keypoints',
                        'keypoints16': 'keypoints',
                        'keypoints17': 'keypoints',
                        'keypoints18': 'keypoints',
                        'keypoints19': 'keypoints',
                        'keypoints20': 'keypoints',
                        }
)
category_ids = [0]
category_id_to_name = {0: 'gate'}


#-------------------------------           augmenting Dataset

for filename in os.listdir(image_dir):
    if filename.endswith('.jpg'):
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(image_path)

        annotations = read_annotations(label_path)

        bbox = []
        # class_id1= x_center1= y_center1= width1= height1= keypoints1= class_id2= x_center2= y_center2= width2= height2= keypoints2= class_id3= x_center3= y_center3= width3= height3= keypoints3= class_id4= x_center4= y_center4= width4= height4= keypoints4 = -1



        keypoints_list = []
        visiblity_list = []
        class_id_list = []
        x_center_list = []
        y_center_list = []
        width_list = []
        height_list = []
        
        keypoints_new_list = []
        
        for i in range(len(annotations)):
            class_id, x_center, y_center, width, height, keypoints = annotations[i]
            keypoints = [(x*639, y*639) for x, y, _ in annotations[i][5]]
            visiblity = [v for _, _, v in annotations[i][5]]
            bbox.append([x_center, y_center, width, height, class_id])
            keypoints_list.append(keypoints)
            visiblity_list.append(visiblity)
            class_id_list.append(class_id)
            x_center_list.append(x_center)
            y_center_list.append(y_center)
            width_list.append(width)
            height_list.append(height)
        
        
        
        

        # print(visiblity)
        print("--------------------")
        for i in range(10):  # Generate 5 augmented versions
            # transformed = transform(image=image, keypoints=keypoints1, keypoint_classes = keypoint_classes, bboxes = bbox, bbox_classes=bbox_classes,keypoints1 = keypoints1)
            aug_image_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
            aug_image_filename_trash = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
            aug_label_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.txt"
            
            aug_image_path = os.path.join(aug_image_dir, aug_image_filename)
            aug_image_path_trash = os.path.join(aug_image_dir_trash, aug_image_filename_trash)
            aug_label_path = os.path.join(aug_label_dir, aug_label_filename)

            l_anno = len(annotations)        
            ex_code = 'transformed = transform(image=image, keypoint_classes = keypoint_classes, bboxes = bbox'
            for i in range(len(annotations)):
                ex_code += f', keypoints{i+1} = keypoints_list[{i}]'
            ex_code += ')'
            exec(ex_code)
            
            
            keypoints_new_list = []
            for i in range(len(annotations)):
                keypoints_new = [transformed[f'keypoints{i+1}'][j] + (visiblity_list[i][j],) for j in range(len(visiblity_list[i]))]
                keypoints_new_list.append(keypoints_new)
                
            # if l_anno > 1:
            #     if l_anno > 2:
            #         if l_anno > 3:
            #             if l_anno > 4:
            #                 transformed = transform(image=image, keypoint_classes = keypoint_classes, bboxes = bbox, keypoints1 = keypoints_list[0], keypoints2 = keypoints_list[1], keypoints3 = keypoints3, keypoints4 = keypoints4 )
            #                 keypoints_new4 = [transformed['keypoints4'][j] + (visiblity4[j],) for j in range(len(visiblity4))]
            #                 keypoints_new_list.append(keypoints_new4)
            #             else:
            #                 transformed = transform(image=image, keypoint_classes = keypoint_classes, bboxes = bbox, keypoints1 = keypoints1, keypoints2 = keypoints2, keypoints3 = keypoints3)
            #             keypoints_new3 = [transformed['keypoints3'][j] + (visiblity3[j],) for j in range(len(visiblity3))]
            #             keypoints_new_list.append(keypoints_new3)
            #         else:
            #             print(bbox)
            #             print(bbox_classes)
            #             transformed = transform(image=image, keypoint_classes = keypoint_classes, bboxes = bbox, keypoints1 = keypoints1, keypoints2 = keypoints2)
            #         keypoints_new2 = [transformed['keypoints2'][j] + (visiblity2[j],) for j in range(len(visiblity2))]
            #         keypoints_new_list.append(keypoints_new2)
            #     else:
            #         transformed = transform(image=image, keypoint_classes = keypoint_classes, bboxes = bbox, keypoints1 = keypoints1)
            #     keypoints_new1 = [transformed['keypoints1'][j] + (visiblity1[j],) for j in range(len(visiblity1))]
            #     keypoints_new_list.append(keypoints_new1)

            # keypoints_new_list.reverse()
            
            try:
                

                

                
                

                bbox_new = transformed['bboxes']


                ret = ret+1
                print("Writing...       ", ret, "           label\n")
                annotations_new = []
                for i in range(len(annotations)):
                    annotations_new.append((bbox_new[i][4], bbox_new[i][0], bbox_new[i][1], bbox_new[i][2], bbox_new[i][3], keypoints_new_list[i]))
                save_keypoints(aug_label_path, annotations_new)
                cv2.imwrite(aug_image_path, cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
                # vis_keypoints(transformed['image'], keypoints_new)
                # visualize(transformed['image'], transformed['bboxes'], category_ids, category_id_to_name)
            except:
                cv2.imwrite(aug_image_path_trash, cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
                print("trashed.... ", annotations[0])

                pass




#-------------------------------       one picture example


# aug_label_path = "C:\\4022term\Tello\Yolo\Gates_first_annotated\\train\\labels_a\\1_png.rf.185ce1047253827fc266a3e8fa472bea_aug.txt"
# aug_image_path = "C:\\4022term\Tello\Yolo\Gates_first_annotated\\train\\image_a\\1_png.rf.185ce1047253827fc266a3e8fa472bea_aug.jpg"

# img_annotations = read_annotations("C:\\4022term\Tello\Yolo\Gates_first_annotated\\train\\labels\\1_png.rf.185ce1047253827fc266a3e8fa472bea.txt")
# class_id, x_center, y_center, width, height, keypoints = img_annotations[0]
# keypoints = [(x*640, y*640) for x, y, _ in img_annotations[0][5]]
# bbox = [[x_center, y_center, width, height, class_id]]
# visiblity = [v for _, _, v in img_annotations[0][5]]

# image = cv2.imread("C:\\4022term\Tello\Yolo\Gates_first_annotated\\train\\images\\1_png.rf.185ce1047253827fc266a3e8fa472bea.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# transformed = transform(image=image, keypoints=keypoints, bboxes = bbox)

# cv2.imwrite(aug_image_path, cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))

# keypoints_new = [transformed['keypoints'][i] + (visiblity[i],) for i in range(len(visiblity))]

# bbox_new = transformed['bboxes'][0]

# save_keypoints(aug_label_path, [(bbox_new[4], bbox_new[0], bbox_new[1], bbox_new[2], bbox_new[3], keypoints_new)])

# # vis_keypoints(transformed['image'], keypoints_new)

# visualize(transformed['image'], transformed['bboxes'], category_ids, category_id_to_name)y