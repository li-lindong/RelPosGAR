
#
Top1_Acc_Early = [88.930, 88.631, 88.856, 88.631, 88.631, 88.706, 88.332, 88.182, 88.556, 88.182, 88.332, 88.407, 88.332, 87.883, 87.883, 88.033, 87.958]
Top5_Acc_Early = [99.327, 99.327, 99.327, 99.327, 99.327, 99.327, 99.327, 99.327, 99.327, 99.327, 99.327, 99.327, 99.327, 99.327, 99.327, 99.327, 99.327]
Top1_Acc_Late = [89.155, 89.155, 89.155, 89.155, 89.155, 89.155, 89.155, 89.155, 89.155, 89.155, 89.155, 89.155, 89.155, 89.155, 89.155, 89.155, 89.155]
Top5_Acc_Late = [99.551, 99.551, 99.551, 99.551, 99.551, 99.551, 99.551, 99.551, 99.551, 99.551, 99.551, 99.551, 99.551, 99.551, 99.551, 99.551, 99.551]

import matplotlib.pyplot as plt
import numpy as np

# 横轴：ID Switch 样本比例（0% ~ 100%）
x = np.linspace(0, 100, len(Top1_Acc_Early))

# 画图
plt.figure(figsize=(5, 3))   # 控制画布尺寸（宽, 高，单位英寸）

plt.plot(x, Top1_Acc_Early, marker='o', linewidth=1.0, markersize=3, label='Early Fusion')
plt.plot(x, Top1_Acc_Late, marker='s', linewidth=1.0, markersize=3, label='Late Fusion')

plt.xlabel('ID Switch Sample Ratio (%)')
plt.ylabel('Top1 Accuracy (%)')
plt.legend(frameon=False)    # 去掉 legend 边框
plt.grid(True)

plt.tight_layout(pad=0.2)    # 核心：压缩内部留白
plt.savefig(
    "Curve_ID_Switch_Press_Test.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.02          # 核心：压缩外部留白
)
plt.show()
