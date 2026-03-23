import numpy as np
import os

def fisher_ratio(features, labels):
    """
    features: numpy array of shape (N, D)
    labels: numpy array of shape (N,)
    """
    features = np.array(features)
    labels = np.array(labels)

    classes = np.unique(labels)
    overall_mean = np.mean(features, axis=0)

    S_B = 0.0  # between-class scatter
    S_W = 0.0  # within-class scatter

    for c in classes:
        class_features = features[labels == c]
        class_mean = np.mean(class_features, axis=0)
        n_c = class_features.shape[0]

        # Between-class scatter
        S_B += n_c * np.sum((class_mean - overall_mean) ** 2)

        # Within-class scatter
        S_W += np.sum((class_features - class_mean) ** 2)

    return S_B / S_W


if __name__ == '__main__':

    dirs = [
        r"/mnt/disk/LiLinDong/RPIR/results/rpir_128_4_6_4_6_add_x_N_topx_N_early_GA_N",
        r"/mnt/disk/LiLinDong/RPIR/results/rpir_128_4_6_4_6_add_x_N_topx_N_late_GA_N"
    ]

    for dir in dirs:
        features = np.load(os.path.join(dir, 'group_features.npy'))
        gt = np.load(os.path.join(dir, 'gt_activities.npy'))

        FR = fisher_ratio(features, gt)

        print("{}: {}".format(dir, FR))