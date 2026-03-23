import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    dirs = [
        r"/mnt/disk/LiLinDong/RPIR/results/rpir_128_4_6_4_6_add_x_N_topx_N_early_GA_N",
        r"/mnt/disk/LiLinDong/RPIR/results/rpir_128_4_6_4_6_add_x_N_topx_N_late_GA_N"
    ]

    for dir in dirs:
        features = np.load(os.path.join(dir, 'group_features.npy'))
        gt = np.load(os.path.join(dir, 'gt_activities.npy'))

        linear_clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
        linear_clf.fit(features, gt)
        pred_linear = linear_clf.predict(features)
        acc_linear = accuracy_score(gt, pred_linear)

        print("{}: {}".format(dir, acc_linear))