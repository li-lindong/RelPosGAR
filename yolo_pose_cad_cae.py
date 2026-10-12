import os
import torch
from ultralytics import YOLO
from tqdm import tqdm

# Load a model
model = YOLO("yolo11x-pose.pt")  # load an official model


data_dir = r"/mnt/disk/LiLinDong/dataset/CAD/seqs"

skeleton_data = {}
not_detect = []
for seq_id in range(1, 75, 1):
    seq_dir = r"{}/seq{}".format(data_dir, str(seq_id).zfill(2))
    img_list = os.listdir(seq_dir)
    if 'annotations.txt' in img_list:
        img_list.remove("annotations.txt")
    img_list.sort()

    for img_name in img_list:
        img_path = os.path.join(seq_dir, img_name)
        results = model(img_path, imgsz=2560, half=False, iou=0.3, conf=0.1, device='cuda')[0]
        boxes = results.boxes.xyxy.cpu()
        keypoints = results.keypoints.data.cpu()
        key = "seq{}-{}".format(str(seq_id).zfill(2), img_name.replace(".jpg", ""))
        value = {"boxes": boxes, "keypoints": keypoints}
        if len(keypoints) == 0:
            not_detect.append([seq_id, img_name])
        skeleton_data[key] = value

print(not_detect)
torch.save(skeleton_data, os.path.join(data_dir, "skeleton_cad_cae.pth"))
# Predict with the model
# results = model("/mnt/disk/LiLinDong/dataset/CAD/seqs/seq01/frame0001.jpg")  # predict on an image
#
# print(results)
#
# # Access the results
# for result in results:
#     xy = result.keypoints.xy  # x and y coordinates
#     xyn = result.keypoints.xyn  # normalized
#     kpts = result.keypoints.data  # x, y, visibility (if available)

