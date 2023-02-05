import cv2
import numpy as np
import os.path as osp
import glob
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torchvision
import mmcv
from mmedit.models import build_model
from mmcv.runner import load_checkpoint


def create_image_dict(INPUT_DIR):
    """
    Create a dictionary {image name:image}
    """
    image_dict = {}
    exts = ['png']

    files =  [glob.glob(INPUT_DIR + f'*.{ext}') for ext in exts]
    
    for i in files:
        for f in i:
            fname = Path(f).stem
            image_dict[fname] = mmcv.imread(f, channel_order='rgb')
    return image_dict


def main(config_path, checkpoint_path, INPUT_DIR, OUTPUT_DIR):

    image_dict = create_image_dict(INPUT_DIR)

    cfg = mmcv.Config.fromfile(config_path)

    model = build_model(cfg.model)

    load_checkpoint(model, checkpoint_path, map_location='cpu')
    # load_checkpoint(model, checkpoint_path, map_location='cuda')

    print('Model processing input images...')

    for image_name, img_LQ in image_dict.items():       
        img_LQ = torch.from_numpy(img_LQ).float().permute(2, 0, 1).unsqueeze(0)
        # img_LQ = torch.from_numpy(img_LQ).cuda().float().permute(2, 0, 1).unsqueeze(0)

        img_LQ = img_LQ / 255
        
        with torch.no_grad():
            img_SR = model.forward_test(img_LQ)['output']

        img_SR = torch.clamp(img_SR, 0, 1)
        img_SR = img_SR.squeeze(0).permute(1, 2, 0).numpy()    

        img_SR = img_SR * 255 
     
        image_SR = Image.fromarray(img_SR.astype(np.uint8))    
        image_SR.save(osp.join(OUTPUT_DIR, image_name + '.png'))

    print('Complete')



if __name__ == '__main__':

    config_path = './srresnet_project.py'
    checkpoint_path = './best_model_HGimages/iter_285k.pth'

    INPUT_DIR = './test/LQ/' 
    OUTPUT_DIR = './best_model_HGimages/'
 
    main(config_path, checkpoint_path, INPUT_DIR, OUTPUT_DIR)