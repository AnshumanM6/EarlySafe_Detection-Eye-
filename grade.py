import os
import sys
import math
import numpy as np
import cv2
from skimage import measure
import pandas as pd
from tqdm import tqdm , notebook
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
sys.path.append("pretrained-models/pretrained-models.pytorch-master/")
import pretrainedmodels

def scale_radius(src, img_size, padding=False):
    x = src[src.shape[0] // 2, ...].sum(axis=1)
    r = (x > x.mean() / 10).sum() // 2
    yx = src.sum(axis=2)
    region_props = measure.regionprops((yx > yx.mean() / 10).astype('uint8'))
    yc, xc = np.round(region_props[0].centroid).astype('int')
    x1 = max(xc - r, 0)
    x2 = min(xc + r, src.shape[1] - 1)
    y1 = max(yc - r, 0)
    y2 = min(yc + r, src.shape[0] - 1)
    dst = src[y1:y2, x1:x2]
    dst = cv2.resize(dst, dsize=None, fx=img_size/(2*r), fy=img_size/(2*r))
    if padding:
        pad_x = (img_size - dst.shape[1]) // 2
        pad_y = (img_size - dst.shape[0]) // 2
        dst = np.pad(dst, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), 'constant')
    return dst

    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, transform=None, img_size=288, save_img=True):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.img_size = img_size
        self.save_img = save_img

    def __getitem__(self, index):
        img_path, label = self.img_paths[index], self.labels[index]
        
        img = cv2.imread(img_path)
        try:
            img = scale_radius(img, img_size=self.img_size, padding=False)
        except Exception as e:
            img = img
        if self.save_img:
            cv2.imwrite('processed/%s' %os.path.basename(img_path), img)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


    def __len__(self):
        return len(self.img_paths)


def get_model(model_name='resnet18', num_outputs=None, pretrained=True,
              freeze_bn=False, dropout_p=0, **kwargs):

    pretrained = None
    model = pretrainedmodels.__dict__[model_name](num_classes=1000,
                                                  pretrained=pretrained)


    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(in_features, num_outputs)

    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    return model
	
def predg(img):

    test_img_paths = [img]

    test_labels = np.zeros(1)
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_set = Dataset(
        test_img_paths,
        test_labels,
    transform=test_transform)
    
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
    num_workers=0)
	
    l1_probs={}
    # create model
    model = get_model(model_name='se_resnext50_32x4d',
                  num_outputs=1,
                  pretrained=False,
                  freeze_bn=True,
                  dropout_p=0)
    #model = model.cuda()
    model.eval()

    probs = []
    for fold in range(2):
        model.load_state_dict(torch.load('se_resnext50_32x4d/model_%d.pth' % (fold+1),map_location=torch.device('cpu')))

        probs_fold = []
        with torch.no_grad():
            for i, (input, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
#                input = input.cuda()
                output = model(input)

                probs_fold.extend(output.data.cpu().numpy()[:, 0])
        probs_fold = np.array(probs_fold)
        probs.append(probs_fold)

    probs = np.mean(probs, axis=0)
    l1_probs['se_resnext50_32x4d'] = probs

    del model
    torch.cuda.empty_cache()

    # create model
    model = get_model(model_name='se_resnext101_32x4d',
                  num_outputs=1,
                  pretrained=False,
                  freeze_bn=True,
                  dropout_p=0)
    #model = model.cuda()
    model.eval()

    probs = []
    for fold in range(5):
        model.load_state_dict(torch.load('se_resnext101_32x4d/model_%d.pth' % (fold+1),map_location=torch.device('cpu')))

        probs_fold = []
        with torch.no_grad():
            for i, (input, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
#            input = input.cuda()
                output = model(input)

                probs_fold.extend(output.data.cpu().numpy()[:, 0])
    probs_fold = np.array(probs_fold)
    probs.append(probs_fold)

    probs = np.mean(probs, axis=0)
    l1_probs['se_resnext101_32x4d'] = probs

    del model
    torch.cuda.empty_cache()

    preds = 0.5 * l1_probs['se_resnext50_32x4d'] + 0.5 * l1_probs['se_resnext101_32x4d']

    thrs = [0.5, 1.5, 2.5, 3.5]
    preds[preds < thrs[0]] = 0
    preds[(preds >= thrs[0]) & (preds < thrs[1])] = 1
    preds[(preds >= thrs[1]) & (preds < thrs[2])] = 2
    preds[(preds >= thrs[2]) & (preds < thrs[3])] = 3
    preds[preds >= thrs[3]] = 4
    preds = preds.astype('int')

    print(preds)
    return preds

#predg('16727_left.jpg')

