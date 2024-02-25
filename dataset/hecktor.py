import os
import glob
import nibabel as nib
import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils import data
from sklearn.model_selection import train_test_split


def get_slice_id(file_name):
    return int(os.path.basename(file_name).replace('.nii.gz', '').split('_')[-1])


class Hecktor22(data.Dataset):
    def __init__(self, args, augs=None):
        super(Hecktor22, self).__init__()
        self.root = args.dataroot
        self.scale_size = (args.img_width, args.img_height)
        self.augs = augs

        random_seed = args.random_seed
        patient_ids = os.listdir(self.root)
        patients_train, patients_test = train_test_split(patient_ids, test_size=0.22, random_state=random_seed)
        
        if args.phase == 'train':
            patients = patients_train
        elif args.phase == 'test':
            patients = patients_test

        gt_masks = []
        imgs = []
        for id in patients:
            imgs_patient = glob.glob(f'{self.root}/{id}/images/*')
            imgs_patient.sort(key = lambda x: get_slice_id(x))

            gt_folders = glob.glob(f'{self.root}/{id}/labels/*')
            gt_patient = []
            for img in imgs_patient:
                slice_name = os.path.basename(img)
                gt_slice = []
                for gt_dir in gt_folders:
                    gt_path = f'{gt_dir}/{slice_name}'
                    gt_slice.append(gt_path)
                gt_patient.append(gt_slice)
            
            imgs.extend(imgs_patient)
            gt_masks.extend(gt_patient)
        
        self.imgs = imgs
        self.labels = gt_masks            

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = nib.load(self.imgs[index]).get_fdata()
        gt = []
        for gt_path in self.labels[index]:
            gt.append(nib.load(gt_path).get_fdata())
        
        masks = list()
        for mask in gt:
            mask = mask[:, :, 0]
            
            gtvp = mask.copy()
            gtvp[gtvp != 1] = 0

            gtvn = mask.copy()
            gtvn[gtvn != 2] = 0
            gtvn[gtvn == 2] = 1

            mask_ = np.stack((gtvp, gtvn))
            masks.append(mask_)
        
        n_valid_masks = len(masks)
        while len(masks) < 4:
            masks.append(mask_)

        return {'image': torch.tensor(img, dtype=torch.float32), 
                'mask': masks,
                'name': self.imgs[index],
                'n_valid_masks': n_valid_masks}
