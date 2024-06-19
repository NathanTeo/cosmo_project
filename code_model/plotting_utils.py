import matplotlib.pyplot as plt
import numpy as np

def plot_img_grid(subfig, imgs, grid_row_num, title, wspace=.2, hspace=.2, subplot_titles=None):
    axs = subfig.subplots(grid_row_num, grid_row_num)
    for i in range(grid_row_num):
        for j in range(grid_row_num):
            axs[i, j].imshow(imgs[(grid_row_num)*i+j], interpolation='none')
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].axis('off')
            if subplot_titles is not None:
                axs[i, j].set_title('{:.4f}'.format(subplot_titles[(grid_row_num)*i+j]))
    subfig.subplots_adjust(wspace=wspace, hspace=hspace)         
    subfig.suptitle(title, y=0.95)

def marginal_sums(img):
    y_sum = [y for y in img.sum(axis=1)]
    x_sum = [x for x in img.sum(axis=0)]
    return y_sum, x_sum
        
def plot_marginal_sums(imgs, subfig, grid_row_num, title):
    axs = subfig.subplots(grid_row_num*grid_row_num)
    for i, ax in enumerate(axs):
        y_sum, x_sum = marginal_sums(imgs[i])
        ax.plot(y_sum, label='y')
        ax.plot(x_sum, label='x')
    subfig.suptitle(title, y=0.95)

def stack_imgs(imgs):
    for i, sample in enumerate(imgs):
        if i == 0:
            stacked_img = sample
        else:
            stacked_img = np.add(stacked_img, sample)
    return stacked_img

def plot_stacked_imgs(ax, stacked_img, title):
    ax.imshow(stacked_img)
    ax.set_title(title)