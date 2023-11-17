# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.datasets import datasets
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg


def main():
    device = 'cuda'

    # run DECA
    deca_cfg.model.use_tex = False
    # 不用 pytorch3d 也可以
    # deca_cfg.rasterizer_type = 'pytorch3d'
    deca_cfg.model.extract_tex = True
    deca = DECA(config = deca_cfg, device=device)
    
    # load test images 
    testdata = datasets.TestData('TestSamples/AFLW2000/image00302.jpg')

    # img_dict: ['image', 'imagename', 'tform', 'original_image'])
    for img_dict in testdata:
        images = img_dict['image'].to(device)[None,...]      # BCHW, here B=1
        with torch.no_grad():
            codedict = deca.encode(images)
            deca_code_shape = codedict['shape']     # [B, 100], FLAME parameters (shape 𝜷)
            deca_code_exp = codedict['exp']         # [B, 50], FLAME parameters (expression 𝝍)
            deca_code_pose = codedict['pose']       # [B, 6], FLAME parameters (pose 𝝍)
            deca_code_tex = codedict['tex']         # [B, 50], albedo parameters
            deca_code_cam = codedict['cam']         # [B, 3], camera 𝒄
            deca_code_light = codedict['light']     # [B, 9, 3], lighting parameters l
            # 以上就是 236 dimensional latent code.
            deca_code_images = codedict['images']   # [B, C, H, W]
            deca_code_detail = codedict['detail']   # [B, 128]
            print('ok')
        
if __name__ == '__main__':
    main()
