import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import segment_anything.utils.augmentation as T
import glob
import numpy as np

class YoloDataset(Dataset):
    def __init__(self, img_folder, transforms=None):
        self.img_folder = img_folder
        self.transforms = transforms
        self.img_files = glob.glob(os.path.join(img_folder[0],"*.jpg"))
        self.label_files = glob.glob(os.path.join(img_folder[1],"*.txt"))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")
        img = img.resize((1024 , 1024))
        w, h = img.size

        # Load labels
        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                label = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                boxes.append([x_center, y_center, width, height])
                labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms: # WHAT IS THIS TRANSFORMS?
            img, target = self.transforms(img, target)

        #make coco format targets 
        #normalized xywh to pixel xyxy format
        coco_boxes = xywhn2xyxy(boxes)
        # import pdb; pdb.set_trace()
        new_target = {}
        new_target['boxes'] = torch.tensor(coco_boxes, dtype=torch.float32)
        new_target['labels'] = torch.tensor(labels, dtype=torch.int64)
        new_target['image_id'] = torch.tensor(extract_id_from_path(img_path), dtype=torch.int64)
        new_target['orig_size'] = torch.as_tensor([int(h), int(w)]) # torch.tensor(orig_size, dtype=torch.int64)
        new_target['size'] = torch.as_tensor([int(h), int(w)])# torch.tensor(list(img4.shape[:2]), dtype=torch.int64)

        #calculate center points of new boxes
        # new_boxes = torch.as_tensor(boxes4[:, 1:], dtype=torch.float32)
        # centre_points = torch.cat((new_boxes[:, ::2].mean(1, True), new_boxes[:, 1::2].mean(1, True)), 1)
        # new_target["center"] = centre_points
        # return img, new_target
        return {"image": img, "target": new_target}

def xywhn2xyxy(x, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y

def extract_id_from_path(path):
    img_name = path.split('\\')[-1].strip('.jpg').strip('v2_')
    return int(img_name)

#TODO : Add data augmentation later, transforms.py file from DETR
def make_yolo_transforms(image_set):

    if image_set == 'train':
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            T.FlipLR(fliplr=0.5)
        ])
        return normalize
    
    if image_set == 'val':
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return normalize
    
    raise ValueError(f'unknown {image_set}')

def build_dataset(image_set, args):
    root = Path(args.yolo_path)
    assert root.exists(), f'provided YOLO path {root} does not exist'
    PATHS = {
        # "train": (root / "train2017", root / "annotations" / "instances_train2017.json"),
        "train": (root / "training" / "images", root / "training" / "labels"),
        "val": (root / "validation" / "images", root / "validation" / "labels"),
    }
    
    img_folder = PATHS[image_set]
    dataset = YoloDataset(img_folder, transforms=make_yolo_transforms(image_set))
    return dataset