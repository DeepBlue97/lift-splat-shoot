"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.fix_nuscenes_formatting()

        print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    
    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
    
    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']  # 900, 1600
        fH, fW = self.data_aug_conf['final_dim']  # 128, 352
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])  # 0.2215754461206792
            resize_dims = (int(W*resize), int(H*resize))  # (354, 199)
            newW, newH = resize_dims  # 354, 199
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH  # 63
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))  # 1
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)  # (1, 63, 353, 191)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])  # 1.177658597821802
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            """ samp
                'token': 'acd8c59c4b564ebb8e7f52086c9b7ec7'
                'sample_token': '91d9058720eb4c25a172236e14e11085'
                'ego_pose_token': 'acd8c59c4b564ebb8e7f52086c9b7ec7'
                'calibrated_sensor_token': 'da8dac846b474a8a9a95cd4d929b6097'
                'timestamp': 1535385096862404
                'fileformat': 'jpg'
                'is_key_frame': True
            """
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])  # os.path.join('/home/YQ_Wang/ocean/datasets/nuscenes/mini', 'samples/CAM_FRONT/n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385096862404.jpg')
            img = Image.open(imgname)  # img.size: (1600, 900)
            post_rot = torch.eye(2)  # post_rot.shape: torch.Size([2, 2])
            post_tran = torch.zeros(2)  # post_tran.shape: torch.Size([2])

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            """ sens
                'token': 'da8dac846b474a8a9a95cd4d929b6097' <str> -- Unique record identifier.
                'sensor_token': '725903f5b62f56118f4094b46a4470d8' <str> -- Foreign key pointing to the sensor type.
                'translation': [1.72200568478, 0.00475453292289, 1.49491291905] <float> [3] -- Coordinate system origin in meters: x, y, z.
                'rotation': [0.5077241387638071, -0.4973392230703816, 0.49837167536166627, -0.4964832014373754] <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z.
                'camera_intrinsic': [[1266.417203046554, 0.0, 816.2670197447984], [0.0, 1266.417203046554, 491.50706579294757], [0.0, 0.0, 1.0]] <float> [3, 3] -- Intrinsic camera calibration. Empty for sensors that are not cameras.
            """
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)  
            """
            sens['rotation']: [0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755]
            rot: tensor([[ 5.6848e-03, -5.6367e-03,  9.9997e-01],
                         [-9.9998e-01, -8.3712e-04,  5.6801e-03],
                         [ 8.0507e-04, -9.9998e-01, -5.6413e-03]])
            """
            tran = torch.Tensor(sens['translation'])  # [1.70079118954, 0.0159456324149, 1.51095763913]

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            """
            resize: 0.2178168484673551
            resize_dims: (348, 196)
            crop: (0, 40, 352, 168)
            flip: True
            rotate: 1.7486513318363883
            """
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )
            """
            img.size: (352, 128)
            post_rot2: tensor([[ 0.2215,  0.0046],
                               [-0.0046,  0.2215]])
            post_tran2: tensor([ -3.5728, -59.3354])
            """
            
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                       nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    def get_binimg(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])  # LiDAR translation
        rot = Quaternion(egopose['rotation']).inverse  # LiDAR rotation
        img = np.zeros((self.nx[0], self.nx[1]))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':  # inst['category_name']: 'vehicle.truck'  # 只检测vehicle
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)  # box转到LiDAR坐标
            box.rotate(rot)  # box转到LiDAR朝向

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img).unsqueeze(0)

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]
        
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data(rec, nsweeps=3)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index] 
        """
        'token': 'cd9964f8c3d34383b16e9c2997de1ed0'
        'timestamp': 1535657108301401
        'prev': '9ab95de13c2d432ebb678ebb5da1ac5e' or ''
        'next': '9ab95de13c2d432ebb678ebb5da1ac5e' or ''
        'scene_token': '2fc3753772e241f2ab2cd16a784cc680'
        'data': 
            'RADAR_FRONT': 'c47c9acb05c8407695b7d965bf2a899c'
            'RADAR_FRONT_LEFT': '10095cfa09114f05bd8e6b44a81328ec'
            'RADAR_FRONT_RIGHT': '200e9412a4d94128a73a97e72a4c56a8'
            'RADAR_BACK_LEFT': '6f36b7b66b9649278b0b3c2b77784f17'
            'RADAR_BACK_RIGHT': '2ee7d375080542f99f5dd7df6aa7890f'
            'LIDAR_TOP': '80bddcd6f16b4bcaa645e3fcbd0de6ef'
            'CAM_FRONT': 'b04ad7e3c86d47fc98dba25595b2327e'
            'CAM_FRONT_RIGHT': '920a8ba176bb4197b2aebaeb8aa51ee1'
            'CAM_BACK_RIGHT': 'f3ffbf4ec43741a49f3375b29caf1306'
            'CAM_BACK': '3318f32d61464751afd80275aa9e5c5f'
            'CAM_BACK_LEFT': '01c012528cc14c23a593cc96ccf73eea'
            'CAM_FRONT_LEFT': 'd6206af179d44816b034e3cbd8ab5eea'

        # len(self.ixes[0]['anns']) = 11
        'anns': [
            'baa09c10842741048a621a15b7330c18',
            '4c869aead13f415a99b14455db02a2b5',
            '23a0a4fdeb92435fa531cce0ae769bb7', 
            ...] 

        """

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        """
        imgs.shape: torch.Size([5, 3, 128, 352])
        rots.shape: torch.Size([5, 3, 3])
        trans.shape: torch.Size([5, 3])
        intrins.shape: torch.Size([5, 3, 3])
        post_rots.shape: torch.Size([5, 3, 3])
        post_trans.shape: torch.Size([5, 3])
        """
        binimg = self.get_binimg(rec)  # 获取GT  # binimg.shape: torch.Size([1, 200, 200])
        
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name):
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=os.path.join(dataroot, version),
                    verbose=False)
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
    }[parser_name]
    traindata = parser(nusc, is_train=True, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf)
    valdata = parser(nusc, is_train=False, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader
