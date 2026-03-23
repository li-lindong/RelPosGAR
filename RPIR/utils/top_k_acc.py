import numpy as np


def top_k_accuracy(y_true, y_pred, k=1):
    """
    计算Top-k准确率。

    参数:
    y_true: 真实标签的数组。
    y_pred: 模型预测的概率数组，每一行代表一个样本，每一列代表一个类别的概率。
    k: Top-k的值，默认为1。

    返回:
    top_k_acc: Top-k准确率。
    """
    assert len(y_true) == len(y_pred), "The length of y_true and y_pred must be equal."
    top_k_indices = np.argsort(y_pred, axis=1)[:, -k:]  # 获取前k个最高概率的索引
    match = np.any(top_k_indices == y_true[:, None], axis=1)  # 检查是否包含真实标签
    top_k_acc = np.mean(match)  # 计算准确率
    return top_k_acc


if __name__ == '__main__':
    # 示例数据
    y_true = np.array([0, 2, 1, 3])  # 真实标签
    y_pred = np.array([
        [0.2, 0.1, 0.7, 0.0],
        [0.1, 0.3, 0.0, 0.6],
        [0.6, 0.1, 0.2, 0.1],
        [0.1, 0.3, 0.4, 0.2]
    ])  # 预测概率

    # 计算Top-1和Top-5准确率
    top_1_accuracy = top_k_accuracy(y_true, y_pred, k=1)
    top_2_accuracy = top_k_accuracy(y_true, y_pred, k=2)

    print(f"Top-1 Accuracy: {top_1_accuracy:.2f}")
    print(f"Top-2 Accuracy: {top_2_accuracy:.2f}")