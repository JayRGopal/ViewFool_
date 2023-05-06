import torch
from torchvision import models
from torchvision import transforms
import cv2
from PIL import Image
import math
import numpy as np
# from zmq import device
import os
import logging

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(dir(models))

from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.models import resnet152
from functools import partial
import sys
sys.path.append('/cifs/data/tserre_lrs/projects/prj_video_imagenet/mae')
import models_vit
import timm
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch.nn as nn


def test_baseline(path, label, model='resnet', is_mean=False):

    images_path = path

    images_name = []
    fileList=os.listdir(images_path)
    n = 0
    for i in fileList:
        images_name.append(fileList[n])
        n = n+1
    images_data = [] # opencv
    tensor_data = [] # pytorch tensor

    for name in images_name:
        if is_mean:
            if name == '100.png':
                img = cv2.imread(images_path + name)
            else:
                continue
        else:
            img = cv2.imread(images_path + name)
        # print(f"name: {images_path+name}, opencv image shape: {img.shape}") # (h,w,c)
        images_data.append(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)

        transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        tensor = transform(img_pil)
        #print(f"tensor shape: {tensor.shape}, max: {torch.max(tensor)}, min: {torch.min(tensor)}") # (c,h,w)
        tensor = torch.unsqueeze(tensor, 0) # 返回一个新的tensor,对输入的既定位置插入维度1
        #print(f"tensor shape: {tensor.shape}, max: {torch.max(tensor)}, min: {torch.min(tensor)}") # (1,c,h,w)
        tensor_data.append(tensor)

    print(f'JAY & NICK ADDED THIS: MODEL = {model}')
    if model=='resnet':
        for unique_iterator in range(10):
            print('JAY & NICK - AT THE RESNET PATH')
        model = models.resnet50(pretrained=False)
        checkpoint = '/HOME/scz1972/run/rsw_/NeRFAttack/NeRF/ckpts/resnet50-0676ba61.pth'
        model.load_state_dict(torch.load(checkpoint))

    elif model=='vit':
        model = models.vit_b_16(pretrained=False)
        checkpoint = '/HOME/scz1972/run/rsw_/NeRFAttack/NeRF/ckpts/vit_b_16-c867db91.pth'
        model.load_state_dict(torch.load(checkpoint))

    else:
        print('LOADING IN CUSTOM MODEL IN PREDICT.PY TEST_BASELINE()')
        model_path = model
        model_description = partial(models_vit.__dict__['vit_base_patch16'], num_classes=1000,
        drop_path_rate=0.1, global_pool=True)
        model = model_description()
        state_dict = torch.load(model_path)['model']
        model.load_state_dict(state_dict)

        num_gpus = torch.cuda.device_count()
        all_parallel_devices = [i for i in range(num_gpus)] #list of GPU IDs to use for model evaluation
        
        model=nn.DataParallel(model, device_ids=all_parallel_devices).cuda()

    # model = models.resnet101(pretrained=True)
    # model = models.inception_v3(pretrained=True)
    # model = models.googlenet(pretrained=True)
    # model = models.vit_b_16(pretrained=True)


    model.eval()

    with open("~/Neurips2023/ViewFool_/classifier/imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]


    print("In test_baseline of predict.py, line 104")

    acc = 0

    # for x in range(len(tensor_data)):
    #     prediction = resnet50(tensor_data[x])
    #     #print(prediction.shape) # [1,1000]

    #     _, index = torch.max(prediction, 1)
    #     percentage = torch.nn.functional.softmax(prediction, dim=1)[0] * 100
    #     print(f"result: {classes[index[0]]}, {percentage[index[0]].item()}")

    #     if classes[index[0]] == 'microphone, mike':
    #         acc += 1

    # acc = acc/len(tensor_data)
    # print("acc:", acc)


    with torch.no_grad():
        for x in range(len(tensor_data)):

            top_num = 1
            prediction = model(tensor_data[x])
            ps = torch.exp(prediction)
            topk, topclass = ps.topk(top_num, dim=1)
            class_ = []
            for i in range(top_num):
                class_.append(classes[topclass.cpu().numpy()[0][i]])
            # print("Output class : ", class_)

            true_label = np.zeros((1, 1000))
            true_label[:, 817] = 1.0
            true_label = torch.from_numpy(true_label)
            loss_func = torch.nn.CrossEntropyLoss()
            # print('loss:', loss_func(prediction, true_label))
            # print('score:', np.max(ps.cpu().numpy())/np.sum(ps.cpu().numpy()))

            for i in range(len(topclass.cpu().numpy()[0])):
                if classes[topclass.cpu().numpy()[0][i]] == label:
                    acc += 1
    
    acc = acc/len(tensor_data)
    print("acc:", acc)
        