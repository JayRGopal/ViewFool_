from rendering_image import render_image
import numpy as np
import cv2 as cv
from PIL import Image
from torchvision import models
from torchvision import transforms
import torch
import torch.nn as nn
#import matplotlib.pyplot as plt
from datasets.opts import get_opts



# Custom imports to load in an MAE model - Jay added

from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.models import resnet152
from functools import partial
import sys
sys.path.append('/cifs/data/tserre_lrs/projects/prj_video_imagenet/mae')
import models_vit
import timm
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch.nn as nn




'''
    Define the adaptability of each NES solution
'''

def metric(prediction, label, target_label, target_flag):
    loss_func = nn.CrossEntropyLoss()
    if target_flag == False:
        # For untargeted attacks, a higher loss value indicates that the attack was more successful
        loss = loss_func(prediction, label)
    else:
        # For a targetless attack, the larger the negative value of loss, the smaller and closer the loss, the more successful the attack
        loss = - loss_func(prediction, target_label)
    return loss


def compute_ver(sigma, mu, num_sample=1000):
  # Calculate the entropy of the multivariate Gaussian distribution

  random = np.zeros([num_sample, 6])
  gamma = np.random.normal(loc=mu[0], scale=sigma[0], size=num_sample)
  th = np.random.normal(loc=mu[1], scale=sigma[1], size=num_sample)
  phi = np.random.normal(loc=mu[2], scale=sigma[2], size=num_sample)
  r = np.random.normal(loc=mu[3], scale=sigma[3], size=num_sample)
  a = np.random.normal(loc=mu[4], scale=sigma[4], size=num_sample)
  b = np.random.normal(loc=mu[5], scale=sigma[5], size=num_sample)
  random[:, 0] = gamma
  random[:, 1] = th
  random[:, 2] = phi
  random[:, 3] = r
  random[:, 4] = a
  random[:, 5] = b
  mu = random.mean(axis=0)
  var = (random - mu).T @ (random - mu) / random.shape[0]

  loss_var = - np.log(np.linalg.det(var))
  loss_var = 0.03 * loss_var
  return loss_var

def comput_fitness(solution):
    '''

    Args:
        solution: The value of the parameter currently sampled
    Returns:
        reward: Fitness value
    '''
    args = get_opts()

    # Render an image using the resulting parameters
    with torch.no_grad():
        x = render_image(solution) # ndarray [W,H,C]
    img_pil = Image.fromarray(x)

    transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    tensor = transform(img_pil)
    # print(f"tensor shape: {tensor.shape}, max: {torch.max(tensor)}, min: {torch.min(tensor)}") # (c,h,w)
    tensor = torch.unsqueeze(tensor, 0)
    # print(f"tensor shape: {tensor.shape}, max: {torch.max(tensor)}, min: {torch.min(tensor)}") # (1,c,h,w)


    # PUTTING IN CUSTOM MODELS
    
    model_path = '/cifs/data/tserre_lrs/projects/prj_video_imagenet/models_to_test/outs_finetune_e2D_d2D_pretrain_vitbase_patch16_224_IN_jump4_checkpoint-99.pth'
    model_description = partial(models_vit.__dict__['vit_base_patch16'], num_classes=1000,
    drop_path_rate=0.1, global_pool=True)
    model = model_description()
    state_dict = torch.load(model_path)['model']
    model.load_state_dict(state_dict)

    num_gpus = torch.cuda.device_count()
    all_parallel_devices = [i for i in range(num_gpus)] #list of GPU IDs to use for model evaluation
    
    model=nn.DataParallel(model, device_ids=all_parallel_devices).cuda()

    #model = models.resnet50(pretrained=True)
    #checkpoint = '/HOME/scz1972/run/rsw_/NeRFAttack/NeRF/ckpts/resnet50-0676ba61.pth'

    # model = models.inception_v3(pretrained=False)
    # checkpoint = '/HOME/scz1975/run/rsw_/NeRFAttack/NeRF/ckpts/inception_v3_google-0cc3c7bd.pth'

    #model = models.vit_b_16(pretrained=False)
    #checkpoint = '/HOME/scz1972/run/rsw_/NeRFAttack/NeRF/ckpts/vit_b_16-c867db91.pth'

    #model.load_state_dict(torch.load(checkpoint))





    model.eval()
    with torch.no_grad():
        # Get the predicted softmax vector

        # ORIGINAL:
        #prediction = model(tensor)

        # USES GPU:
        prediction = model(tensor.cuda())
        prediction = prediction.cpu()

    true_label = np.zeros((1, 1000))
    true_label[:, args.label] = 1.0
    true_label = torch.from_numpy(true_label)

    target_label = np.zeros((1, 1000))
    """
    TODO: Figure out why they have these classes here:
    584: hair slide
    650: microphone, mike
    """

    target_label[:, args.target_label] = 1.0
    target_label = torch.from_numpy(target_label)

    reward = metric(prediction, label=true_label, target_label=target_label, target_flag=args.target_flag)
    reward = reward.cpu().detach().numpy()
    return reward






