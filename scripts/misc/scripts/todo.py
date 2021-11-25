### albumentations

import os

import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.color import label2rgb

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
import random

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)

def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2, **kwargs):
    #height, width = img.shape[:2]

    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img

def visualize_titles(img, bbox, title, color=BOX_COLOR, thickness=2, font_thickness = 2, font_scale=0.35, **kwargs):
    #height, width = img.shape[:2]
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    
    ((text_width, text_height), _) = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, title, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR,
                font_thickness, lineType=cv2.LINE_AA)
    return img


def augment_and_show(aug, image, mask=None, bboxes=[], categories=[], category_id_to_name=[], filename=None, 
                     font_scale_orig=0.35, 
                     font_scale_aug=0.35, show_title=True, **kwargs):

    augmented = aug(image=image, mask=mask, bboxes=bboxes, category_id=categories)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_aug = cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB)

    for bbox in bboxes:
        visualize_bbox(image, bbox, **kwargs)

    for bbox in augmented['bboxes']:
        visualize_bbox(image_aug, bbox, **kwargs)

    if show_title:
        for bbox,cat_id in zip(bboxes, categories):
            visualize_titles(image, bbox, category_id_to_name[cat_id], font_scale=font_scale_orig, **kwargs)
        for bbox,cat_id in zip(augmented['bboxes'], augmented['category_id']):
            visualize_titles(image_aug, bbox, category_id_to_name[cat_id], font_scale=font_scale_aug, **kwargs)

    
    if mask is None:
        f, ax = plt.subplots(1, 2, figsize=(8, 8))
        
        ax[0].imshow(image)
        ax[0].set_title('Original image')
        
        ax[1].imshow(image_aug)
        ax[1].set_title('Augmented image')
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))
        
        if len(mask.shape) != 3:
            mask = label2rgb(mask, bg_label=0)            
            mask_aug = label2rgb(augmented['mask'], bg_label=0)
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask_aug = cv2.cvtColor(augmented['mask'], cv2.COLOR_BGR2RGB)
            
        ax[0, 0].imshow(image)
        ax[0, 0].set_title('Original image')
        
        ax[0, 1].imshow(image_aug)
        ax[0, 1].set_title('Augmented image')
        
        ax[1, 0].imshow(mask, interpolation='nearest')
        ax[1, 0].set_title('Original mask')

        ax[1, 1].imshow(mask_aug, interpolation='nearest')
        ax[1, 1].set_title('Augmented mask')

    f.tight_layout()
    if filename is not None:
        f.savefig(filename)
        
    return augmented['image'], augmented['mask'], augmented['bboxes']

def find_in_dir(dirname):
    return [os.path.join(dirname, fname) for fname in sorted(os.listdir(dirname))]

def strong_aug(p=0.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            #CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)


light = Compose([
    #RandomSizedCrop((224-100, 224+100), 224, 224),
    ShiftScaleRotate(),
    #A.RGBShift(),
    #A.Blur(),
    #A.GaussNoise(),
    #A.ElasticTransform(),
    #A.Cutout(p=1)
],p=1)

heavy = Compose([
    #RandomBrightness(p=0.5),
    #Rotate(p=0.5),
    Flip(p=0.5),
    #A.CenterCrop(p=0.5) #height=224, width=224, 
    #A.VerticalFlip(p=0.5)
])

### GO

image = train_x[0]

for i in range(5):
    #image = image.astype(np.uint8)
    augment_and_show(strong_aug(1), image)


## Generators

def data_gen(x, y, batch_size):
    data = np.zeros((batch_size,
                     dataclass.height,
                     dataclass.width,
                     dataclass.channels), dtype=np.float32)
    
    labels = np.zeros((batch_size, dataclass.num_labels), dtype=np.float32)
    
    steps = len(x) // batch_size
    indices = np.arange(len(x))
    
    print(f'data_gen : {steps} and {len(x)}')
    
    i = 0
    while True:
        np.random.shuffle(indices)
        
        count = 0
        next_batch = indices[(i*batch_size):(i+1)*batch_size]
        for idx in next_batch:
            data[count] = seq.augment_image(x[idx])
            labels[count] = y[idx]
            
            #plt.imshow(data[count])
            #plt.show()
            
            count += 1
            
            if count == batch_size-1:
                break
        i += 1
        yield data, labels
    
        if i >= steps:
            i = 0

"""
import scikitplot as skplt
import matplotlib.pyplot as plt

preds_ = model.predict(test_x)
print(preds_)
ground_truth_ = test_y

skplt.metrics.plot_roc_curve(ground_truth_, preds_)
plt.show()

from sklearn.metrics import roc_curve, auc
import pandas as pd

df = pd.DataFrame(ground_truth, columns=['label'])
num_normal = len(df[df['label'] == 0])

fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(dataclass.labels.keys()):
    print(idx, c_label)
    
    if idx == 0:
        fpr, tpr, thresholds = roc_curve(ground_truth[:num_normal], preds[:num_normal])
    else:
        fpr, tpr, thresholds = roc_curve(ground_truth[num_normal:], preds[num_normal:])
    
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))

    c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
"""<

"""
epoch = 1
sample_x = train_x[0:1]
sample_y = train_y[0:1]
for bx, by in data_gen(train_x[0:1], train_y[0:1], batch_size=4):
    sx, sy = bx, by
    break

fig, axes = plt.subplots(1, 4, figsize=(12, 4))
for i in range(4):
    axes[i].imshow(sx[i])
    axes[i].set_title(str(sy[i]))

fig, axes = plt.subplots(1, 5, figsize=(12, 4)) #, constrained_layout=True)
axes[0].imshow(sample_x[0])
axes[0].set_title('original image ' + str(sample_y[0]))

for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for batch_x, batch_y in data_gen(sample_x, sample_y, batch_size=4):
        
        for i in range(len(batch_x)):
            axes[i+1].imshow(batch_x[i])
            axes[i+1].set_title(str(batch_y[i]))
        
        batches += 1
        if batches >= 1:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
"""
