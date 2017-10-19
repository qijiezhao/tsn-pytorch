import os,sys
import numpy as np

input_frame_list=sys.argv[1]
out_video_list=sys.argv[2]

frames_dic={}
for line in open(input_frame_list,'rb').readlines():
    line_list=line.strip().split(' ')
    video_name=line_list[0].split('/')[-1]
    frames_dic[video_name]=line_list[1]+' '+line_list[2]

out_list=[]
for key,value in frames_dic.items():
    path=key.split('_')[1]+'/'+key+'.avi'
    out_list.append(path+' '+value)

with open(out_video_list,'wb')as fw:
    fw.write('\n'.join(out_list))