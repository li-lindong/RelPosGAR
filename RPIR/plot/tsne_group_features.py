import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import seaborn as sns

dim = 2
label = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
              'l_set', 'l-spike', 'l-pass', 'l_winpoint']
color = ['royalblue', 'lightcoral', 'violet', 'gold', 'silver',
         'limegreen', 'aqua', 'teal', 'thistle', 'saddlebrown',
         'grey', 'indigo']

dirs = [
    # r"/mnt/disk/LiLinDong/RPIR/results/rpir_128_N_N_N_N_N_N_N_N_N_late_GA_N",
    # r"/mnt/disk/LiLinDong/RPIR/results/rpir_128_4_6_N_N_add_N_N_N_N_late_GA_N",
    # r"/mnt/disk/LiLinDong/RPIR/results/rpir_128_N_N_4_6_add_x_N_topx_N_late_GA_N",
    # r"/mnt/disk/LiLinDong/RPIR/results/rpir_128_4_6_4_6_add_x_N_topx_N_late_GA_N",
    r"/mnt/disk/LiLinDong/RPIR/results/rpir_128_4_6_4_6_add_x_N_topx_N_early_GA_N"
]


for dir in dirs:
    features = np.load(os.path.join(dir, 'group_features.npy'))
    gt = np.load(os.path.join(dir, 'gt_activities.npy'))

    # tsne = PCA(n_components=dim).fit_transform(ind_features)
    tsne = manifold.TSNE(n_components=dim, init='pca', random_state=5).fit_transform(features)
    x_min, x_max = tsne.min(0), tsne.max(0)
    tsne_norm = (tsne - x_min) / (x_max - x_min)

    # 设置seaborn的样式
    sns.set_style("whitegrid")

    # 设置全局字体大小
    plt.rcParams.update({
        'font.size': 14,  # 全局字体大小
        'axes.titlesize': 16,  # 标题字体大小
        'axes.labelsize': 14,  # 坐标轴标签字体大小
        'xtick.labelsize': 12,  # x轴刻度字体大小
        'ytick.labelsize': 12,  # y轴刻度字体大小
        'legend.fontsize': 12,  # 图例字体大小
    })

    # 画图
    plt.figure(figsize=(8, 8))
    for i in range(len(label)):
        idxs_i = np.where(gt == i)[0]  # 取出预测值为第i类的索引
        plt.scatter(tsne_norm[idxs_i, 0], tsne_norm[idxs_i, 1],
                    s=20,  # 增大点的大小
                    c=color[i],
                    edgecolor='black',  # 添加边缘颜色
                    linewidth=0.3,
                    alpha=0.7,  # 调整透明度
                    label=label[i])
    # 设置图例为 2行4列，横向排列
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fancybox=True, shadow=False)

    # 隐藏坐标轴（可选）
    # plt.axis('off')

    # 自动调整布局以防止重叠
    plt.tight_layout()

    plt.savefig(os.path.join(dir, 'tsne_group_features_{}.png'.format(dir.split('/')[-1])), dpi=600)
    plt.show()