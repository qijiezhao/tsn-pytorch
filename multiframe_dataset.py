import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import skvideo.io
import cv2
import torch,time
from ops.utils import PSP_oneVSall
from log import log

class VideoRecord(object):
    def __init__(self,root_path,row):
        self._data=row
        self.root_path=root_path
        self.item='train' if self._data[0][0]=='/' else 'nontrain'
    @property
    def path(self):
        if self._data[0][0]=='/':
            return self._data[0]
        else:
            return os.path.join(self.root_path,self._data[0])
    
    @property
    def num_frames(self):
        return int(self._data[1])
    @property
    def label(self):
        return int(self._data[2])

    @property
    def posi_time(self):
        #return self.posi_time
        return [(int(self._data[i*2+3]),int(self._data[i*2+4])) for i in range((len(self._data)-3)/2)\
        if int(self._data[i*2+4])<int(self._data[1])]

class MultiFrame(data.Dataset):
    def __init__(self,root_path,list_file,train_source=None,
                 temporal_length=8,temporal_segs=4,
                 modality='RGB',data_gap=1,data_channel=3,
                 image_tmpl='img_{:05d}.jpg',
                 posi_samples=3,nega_samples=0,
                 posi_threshold=0.9,nega_threshold=0.1,
                 transform=None,test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.data_gap=data_gap
        self.temporal_length=temporal_length
        self.temporal_segs=temporal_segs
        self.modality = modality
        self.posi_samples=posi_samples
        self.nega_samples=nega_samples
        self.posi_threshold=posi_threshold
        self.nega_threshold=nega_threshold
        self.channels=data_channel
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.test_mode = test_mode
        #self.new_length = 1 if self.modality=='RGB' else 2
        self.fore_frames=(self.temporal_length*self.temporal_segs*self.data_gap-1)/2
        self.back_frames=(self.temporal_length*self.temporal_segs*self.data_gap)-self.fore_frames-1

        if train_source==None:
            log.l.info('No training source is chosed')
        self._set_train_source(train_source)
        self._parse_list()

    def _set_train_source(self,train_source):
        if len(train_source)==2:
            if train_source[0]=='train' and train_source[1]=='val':
                self.train_item=0
        else:
            if train_source[0]=='train':
                self.train_item=1
            else: # train_source[0]=='val':
                self.train_item=2

    def _is_from_train(self,line_list):
        if self.train_item==0 or self.test_mode:
            return True
        elif line_list[0][0]=='/' and self.train_item == 1:
            return True
        elif line_list[0][0]!='/' and self.train_item == 2:
            return True
        else:
            return False

    def _parse_list(self):
        self.video_list = \
        [VideoRecord(self.root_path,x.strip().split(' ')) for x in open(self.list_file,'rb') if self._is_from_train(x.strip().split(' '))]
        if not self.test_mode:
            log.l.info('========> training source is ID-{}(tip: 0--train+val, 1--train, 2--val), all have {} files to train <========='.format(self.train_item,len(self.video_list)))
        else:
            pass # TBA

    def _load_image(self,record,idx=None):
        img_path=record.path
        if self.modality=='RGB':
            return [Image.open(os.path.join(img_path,self.image_tmpl.format(idx))).convert('RGB')]

        elif self.modality=='Flow':
            #img_path=self.video_list[idx].path
            x_img=Image.open(os.path.join(img_path,self.image_tmpl.format('x',idx))).convert('L')
            y_img=Image.open(os.path.join(img_path,self.image_tmpl.format('y',idx))).convert('L')

            return [x_img, y_img]


    def get(self,record,sample_indices=None):

        images=list()

        #if not self.test_mode:
        for seg_ind in sample_indices:
            p=int(seg_ind)
            # s1 = randint(p-self.fore_frames,p-self.fore_frames+self.segment_length)
            # s2 = randint(p-self.fore_frames+self.segment_length,p+back_frames-self.segment_length)
            # s3 = randint(p+back_frames-self.segment_length,p+back_frames)
            add_inds=0 # if 4*8 > num_frames
            for i in range(self.temporal_segs):
                s_= p-self.fore_frames + i * self.temporal_length * self.data_gap + 1
                e_= p-self.fore_frames + (i + 1) * self.temporal_length * self.data_gap + 1

                add_inds= e_-record.num_frames if e_>record.num_frames else 0 # the last segment 

                for j in range(s_, e_, 1):
                    ind=min(j-add_inds,record.num_frames)
                    seg_imgs = self._load_image(record,ind)
                    images.extend(seg_imgs)

        process_data=self.transform(images)
        # else:
        #     for j in range(self.new_length):
        #         seg_imgs=self._load_image(path,ind,j)
        #         images.extend(seg_imgs)

        #     process_data=self.transform(images)
        return process_data


    def _get_test_indices(self,record,data_gap):
        num_frames=record.num_frames
        posi_time=record.posi_time
        posi_frame_inds=[(_[0],_[1]) for _ in posi_time]
        if_posi_inds=np.zeros(num_frames)

        for i in posi_frame_inds:
            if_posi_inds[i[0]:i[1]]=1
        self.labels=[record.label if if_posi_inds[_]==1 else 0 for _ in range(1,num_frames,data_gap)]
        return range(1,num_frames,data_gap)

    def _check_segment_label(self,record,begin_indice):
        time_segment=(begin_indice-self.fore_frames,begin_indice+self.back_frames)

        if PSP_oneVSall(time_segment,record.posi_time)>self.posi_threshold:
            return 'nega'
        elif PSP_oneVSall(time_segment,record.posi_time)<self.nega_threshold:
            return 'posi'
        else:
            return 'nothing'

    def _get_val_indices(self,record):

        training_indices=[]
        training_lables=[]
        posi_num,nega_num=0,0
        c=time.time()
        while (posi_num<self.posi_samples) or (nega_num<self.nega_samples): # sample posi and nega.
            begin_indice=randint(self.fore_frames,record.num_frames-self.back_frames)

            if self._check_segment_label(record,begin_indice)=='posi':
                if posi_num<self.posi_samples:
                    training_indices.append(begin_indice)
                    training_lables.append(record.label)
                    posi_num+=1

            elif self._check_segment_label(record,begin_indice)=='nega':
                if nega_num<self.nega_samples:
                    training_indices.append(begin_indice)
                    training_lables.append(0)
                    nega_num+=1
            if time.time()-c>3:
                log.l.info('{} meets error when finding positive samples'.format(record.path))
                training_indices.append(record.num_frames/2)
                training_lables.append(record.label)
                posi_num+=1

        return np.array(training_indices),np.array(training_lables)

    def _get_train_indices(self,record):

        training_indices=[]
        training_lables=[]
        posi_num=0  # trimmed data, so no nega samples

        while posi_num< self.posi_samples + self.nega_samples:
            begin_indice=randint(self.fore_frames,max(self.fore_frames+1,record.num_frames-self.back_frames))
            training_indices.append(begin_indice)
            training_lables.append(record.label)
            posi_num+=1
        return np.array(training_indices), np.array(training_lables)


    def __getitem__(self,index):

        record=self.video_list[index]
        # if self.test_mode:
        #     sample_indices = self._get_test_indices(record)
        #     return self.get(record,sample_indices)
        # else:
        if record.item=='nontrain': # so extract validation frames
            sample_indices ,labels = self._get_val_indices(record)
        else:
            sample_indices ,labels = self._get_train_indices(record)

        assert len(sample_indices)==(self.posi_samples+self.nega_samples),\
        'samples not equal, dataloader error!'
        return self.get(record,sample_indices),labels
        
    # def extend_labels_segments(self,labels):
    #     labels_out=[]
    #     for label in labels:
    #         labels_out.extend([label]*self.num_segments)
    #     return np.array(labels_out)

    def __len__(self):
        return len(self.video_list)