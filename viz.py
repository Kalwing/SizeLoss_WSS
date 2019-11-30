import matplotlib.pyplot as plt
import os
import numpy as np


PATH = "./segthor_results/segthor/sizeloss_e/"
print(os.listdir("."))
iters = os.listdir(PATH)
PATH_ITER = PATH + iters[-1] + '/val'
slice = os.listdir(PATH_ITER)

fig, ax = plt.subplots(1, 10, figsize=(25, 2.25))
fig1, ax1 = plt.subplots(1, 10, figsize=(25, 2.25))

for i in range(10):
    img = plt.imread("data/SEGTHOR-Aug/val/img/" + slice[i])
    gt = plt.imread("data/SEGTHOR-Aug/val/gt/" + slice[i])
    pred = plt.imread(os.path.join(PATH_ITER, slice[i]))
    ax[i].imshow(img)
    ax1[i].imshow(img)
    ax[i].imshow(gt, alpha=0.5)
    ax1[i].imshow(pred, alpha=0.5)
fig.suptitle("Ground Truth")
fig1.suptitle("Prediction using common bounds at the last epoch")

fig.savefig('tmp.png')
img = plt.imread('tmp.png')
fig1.savefig('tmp.png')
img1 = plt.imread('tmp.png')
plt.show()

fig = plt.figure(figsize=(25, 20))
height = img.shape[0]
width = img.shape[1]
final = np.zeros((2*height, width, img.shape[-1]))
print(final[0:height, 0:width].shape)
final[:height, :width] = img
final[height:2*height, :width] = img1

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

ax.imshow(final)
fig.savefig("comparison.png")
os.remove('tmp.png')
