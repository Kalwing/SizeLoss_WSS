import numpy as np
from scipy import stats
import os


paths_labels = [
    ('../Done/seg_0', "Without CoordConv"),
    # ('../paper/first', "With a CoordConv block on the first layer"),
    # ('../paper/conv', "With CoordConv on each convolutional block (but not transposed)"),
    ('../paper/all', "With CoordConv on all convolutional block"),
]

VAL = True
SUPERVISION_LVL = "sizeloss_e"
PATH = "results/segthor"

dists = []
for path, label in paths_labels:
    try:
        trainpath = os.path.join(path, PATH, SUPERVISION_LVL)
        npypath = os.path.join(trainpath, F'{"val_dice" if VAL else "tra_loss"}.npy')
        dice = np.load(npypath)
    except FileNotFoundError:
        continue
    with open(os.path.join(trainpath, "best_epoch.txt")) as fin:
        best_epoch = int(fin.readline())
    print(best_epoch)
    d1 = dice[best_epoch][:,1] # Get the distribution for the option selected
    dists.append(d1)

print(np.mean(dists[0]), np.mean(dists[1]))
stats.ttest_ind(dists[0], dists[1])
