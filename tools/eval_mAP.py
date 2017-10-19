import argparse
import sys,os
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn import preprocessing
lb=preprocessing.LabelBinarizer()

Actions=['BG','BaseballPitch','BasketballDunk','Billiards','CleanAndJerk','CliffDiving',\
         'CricketBowling','CricketShot','Diving','FrisbeeCatch','GolfSwing','HammerThrow','HighJump',\
         'JavelinThrow','LongJump','PoleVault','Shotput','SoccerPenalty','TennisSwing',
         'ThrowDiscus','VolleyballSpiking']

parser = argparse.ArgumentParser(description='This script is to compute mAP for per frame score')
parser.add_argument('score_files',nargs='+',type=str,default=None)
parser.add_argument('--score_weights',nargs='+',type=float,default=None)
args=parser.parse_args()
score_npz_files=[np.load(x) for x in args.score_files]

if args.score_weights is None:
    score_weights=[1] *len(score_npz_files)
else:
    score_weights=args.score_weights
    assert len(score_weights)==len(score_npz_files),'Only {} weight specified for a total of {} score files'.\
        format(len(score_weights),len(score_npz_files))

for num,x in enumerate(score_npz_files):
    if num==0:
        score_list=x['scores']
        label_list=x['labels']
    else:
        score_list=np.vstack([score_list,x['scores']])
        label_list=np.vstack([label_list,x['labels']])

n_class=21
lb.fit(range(n_class))
label_list=lb.transform(label_list)

APs=[0]
for i in range(1,n_class,1):
    APs.append(average_precision_score(label_list[:,i],score_list[:,i]))

print '\n'.join(['Action {} gets AP: {}'.format(Actions[i],APs[i]) for i in range(1,n_class,1)])
print '\n =====> mAP is :\n\t\t {}'.format(np.mean(APs))
