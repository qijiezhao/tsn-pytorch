import argparse
import time,sys

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from multiframe_dataset import MultiFrame
from models import TSN
from transforms import *
from ops import ConsensusModule
from log import log
from ops.utils import get_AP_video
from IPython import embed

parser = argparse.ArgumentParser(
    description="Multi-frame testing: video-level curve requirement")
parser.add_argument('dataset', type=str, choices=['thumos14','activity-net'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--frames_root',type=str,default='../../../temporal_action_localization/data/')
parser.add_argument('--data_gap',default=3,type=int)    
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=1, choices=[1,10])
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--test_segments',default=3,type=int)
parser.add_argument('--data_workers',default=1,type=int)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')
parser.add_argument('--batch_size',type=int,default=100)
args = parser.parse_args()

log.l.info('Input command:\n python2 '+ ' '.join(sys.argv))


if args.dataset == 'activity-net':
    num_class = 201
elif args.dataset== 'thumos14':
    num_class = 21
else:
    raise ValueError('Unknown dataset '+args.dataset)

net = TSN(num_class, args.test_segments, args.modality,
          base_model=args.arch,
          consensus_type=args.crop_fusion_type,
          dropout=args.dropout)


checkpoint = torch.load(args.weights)
log.l.info("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))


# data loading type is different from training
data_loader = torch.utils.data.DataLoader(
        MultiFrame(args.frames_root, args.test_list, num_segments=args.test_segments,
                   new_length=1 if args.modality == "RGB" else 5,data_gap=args.data_gap,
                   modality=args.modality,
                   image_tmpl="img_{:05d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else args.flow_prefix+"{}_{:05d}.jpg",
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.data_workers, pin_memory=True)


if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))

net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
net.eval()
data_gen = enumerate(data_loader)
total_num = len(data_loader.dataset)
output = []

log.l.info('=================== Now, it\'s testing ==================')

proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

for i, (data, label) in data_gen: # every batch contains frames from a same video, so contains multiple labels
    if i >= max_num:
        break
    rst = eval_video((i, data, label))
    output.append(rst)
    cnt_time = time.time() - proc_start_time
    log.l.info('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                        total_num,            
                                                                        float(cnt_time) / (args.batch_size*(i+1))))


video_ls = np.array([x[1].numpy() for x in output])
video_labels=[]
for i in video_ls:
    video_labels.extend(i)
video_labels=np.array(video_labels)

out=[]
for num,i in enumerate(output):
    if num==0:
        out=i[0]
    else:
        out=np.vstack([out,i[0]])

output=np.array(out)

if args.save_scores is not None:

    name_list = [x.strip().split()[0] for i,x in enumerate(open(args.test_list)) if i%args.data_gap==0][:max_num*args.batch_size]
    if len(name_list)>len(output):
        name_list=name_list[:len(output)]
    else:
        output=output[:len(name_list)]
    order_dict = {e:i for i, e in enumerate(sorted(name_list))}

    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)
    for i in range(len(output)):
        reorder_output[i] = output[i]
        reorder_label[i] = video_labels[i]

    np.savez(args.save_scores, scores=reorder_output, labels=reorder_label)



