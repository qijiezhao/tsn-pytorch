import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import skvideo.io
import cv2
import torch
from log import log

class VideoRecord(object):
    def __init__(self,root_path,row):
        self._data=row
        self.root_path=root_path

    @property
    def path(self):
        if self._data[0][0]=='/':
            return self._data[0]
        else:
            return os.path.join(self.root_path,self._data[0])
    
    @property
    def frame_id(self):
        return int(self._data[1])
    @property
    def num_frames(self):
        return int(self._data[2])
    @property
    def label(self):
        return int(self._data[3])

    @property
    def posi_time(self):
        #return self.posi_time
        return [(int(self._data[i*2+3]),int(self._data[i*2+4])) for i in range((len(self._data)-3)/2)\
        if int(self._data[i*2+4])<int(self._data[1])]

class PerFrameData(data.Dataset):
    def __init__(self,root_path,list_file,
                 num_segments=1,data_gap=1,new_length=1,modality='RGB',
                 image_tmpl='img_{:05d}.jpg',
                 transform=None,test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.data_gap=data_gap
        self.modality = modality
        self.channels=self.new_length*3 if self.modality=='RGB' else self.new_length*2 # only consider RGB and Flow
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.test_mode = test_mode
        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()


    def _is_trainable(self,line_list):
        label_=int(line_list[-1])
        if self.test_mode:
            return True
        if label_==0 and not self.test_mode:
            return False
        else:
            return True
    def _parse_list(self):
        self.video_list = \
        [VideoRecord(self.root_path,x.strip().split(' ')) for x in open(self.list_file,'rb') if self._is_trainable(x.strip().split(' '))]
        log.l.info('===========> training samples all contains {} frames'.format(len(self.video_list)))

    def _load_image(self,img_path,add_id=None):
        if self.modality=='RGB':
            return [Image.open(img_path).convert('RGB')]

        elif self.modality=='Flow':
            img_path=self.video_list[self.index+add_id].path
            x_img=Image.open(img_path.replace('@','x')).convert('L')
            y_img=Image.open(img_path.replace('@','y')).convert('L')
            return [x_img, y_img]


    def get(self,record,sample_indices=None):
        path=record.path
        ind=record.frame_id
        images=list()
        if self.modality=='RGB':
            process_data=torch.zeros(self.channels,224,224)
            process_data=self.transform(self._load_image(path))
        else:
            for j in range(self.new_length):
                seg_imgs=self._load_image(path,j)
                images.extend(seg_imgs)

            process_data=self.transform(images)
        return process_data, record.label 


    def _get_test_indices(self,record,data_gap):
        num_frames=record.num_frames
        posi_time=record.posi_time
        posi_frame_inds=[(_[0],_[1]) for _ in posi_time]
        if_posi_inds=np.zeros(num_frames)

        for i in posi_frame_inds:
            if_posi_inds[i[0]:i[1]]=1
        self.labels=[record.label if if_posi_inds[_]==1 else 0 for _ in range(1,num_frames,data_gap)]
        return range(1,num_frames,data_gap)


    def __getitem__(self,index):
        self.index=index
        if not self.test_mode:
            record=self.video_list[min(index*self.data_gap+randint(0,self.data_gap),len(self.video_list)-1)]

        else:
            record=self.video_list[index*self.data_gap]
        return self.get(record)
        

    def __len__(self):
        return len(self.video_list)/self.data_gap