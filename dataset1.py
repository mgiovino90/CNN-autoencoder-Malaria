# import packages
import cv2
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


# make custom dataset
class custom_dataset(Dataset):
  def __init__(self, img_dir, num_img, transform= None):
    self.img_dir= img_dir
    self.transform= transform
    self.num_img= num_img

  def __len__(self):
    img_str= self.img_dir + "/*.png"
    # image list
    img_list= glob.glob(img_str)
    if self.num_img is not None:
      img_list= img_list[:self.num_img]
    return len(img_list)

  def __getitem__(self, idx):
    img_str= self.img_dir + "/*.png"
    # image list
    img_list= glob.glob(img_str)
    # read image
    img0= cv2.imread(img_list[idx])
    # convert to hsv
    img_hsv= cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
    # normalize, convert to float32
    s_norm= img_hsv[:, :, 1]/255.0
    sat1= s_norm.reshape(s_norm.shape[0], s_norm.shape[1], 1)
    sat2= sat1.astype(np.float32)
    if self.transform is not None:
      sat2= self.transform(sat2)
    return sat2
