import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import wandb
import warnings
import torchvision
import torchvision.transforms as transforms
import cv2
from PIL import Image
import os
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm 

from models.builder import MODEL_GETTER

def copy_file(source_directory, destination_directory, filename):
    """
    Utility function used to copy a file from a source_directory to a destination_directory
    """
    destination_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_directory/filename, destination_directory/filename)

def organize_test_dataset():
    source_directory = "./data_bird/test"
    destination_directory = ""
    with os.scandir(source_directory) as it:
        for entry in it:
            if entry.is_file():
                img_index = entry.name.split('.')[0]  # The index is the name of the image except the extension

                destination_directory = "./organize_test/undefined"
                
                copy_file(Path(source_directory), Path(destination_directory), entry.name)

    return destination_directory

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 istrain: bool,
                 root: str,
                 data_size: int,
                 return_index: bool = False):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """ basic information """
        self.root = root
        self.data_size = data_size
        self.return_index = return_index

        """ declare data augmentation """
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

        # 448:600
        # 384:510
        # 768:
        if istrain:
            # transforms.RandomApply([RandAugment(n=2, m=3, img_size=data_size)], p=0.1)
            # RandAugment(n=2, m=3, img_size=sub_data_size)
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.RandomCrop((data_size, data_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        normalize
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.CenterCrop((data_size, data_size)),
                        transforms.ToTensor(),
                        normalize
                ])

        """ read all data information """
        self.data_infos = self.getDataInfo(root)


    def getDataInfo(self, root):
        data_infos = []
        folders = os.listdir(root)
        folders.sort() # sort by alphabet
        print("test num: ", len(folders))
        for pic in folders:
            data_path = root+"/"+pic
            name = pic.replace(".jpg", "")
            data_infos.append({"path":data_path, "label":0, "img_name":name})
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # get data information.
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        img_name = self.data_infos[index]["img_name"]
        # read image by opencv.
        img = cv2.imread(image_path)
        img = img[:, :, ::-1] # BGR to RGB.
        
        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)
        
        if self.return_index:
            # return index, img, sub_imgs, label, sub_boundarys
            return img_name, img, label
        
        # return img, sub_imgs, label, sub_boundarys
        return img, label

config = {
        'use_fpn': True,
        'fpn_size': 1536,
        'use_selection': True,
        'num_classes': 200,
        'num_selects': {
            "layer1":2048,
            "layer2":512,
            "layer3":128,
            "layer4":32
        },
        'use_combiner': True
        }

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # prepare testloader & classes
    root = organize_test_dataset()

    test_set = ImageDataset(istrain=False, root=root, data_size=384, return_index=True)
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=1, shuffle=True, batch_size=4)
    
    classes_name = os.listdir("./data_bird/train")
    classes_name.sort()

    # prepare model
    model = MODEL_GETTER["swin-t"](
        use_fpn = config["use_fpn"],
        fpn_size = config["fpn_size"],
        use_selection = config["use_selection"],
        num_classes = config["num_classes"],
        num_selects = config["num_selects"],
        use_combiner = config["use_combiner"],
    )

    checkpoint = torch.load("./records/CUB200-T2/T3000/backup/best.pt", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']
    # print(model)

    # generate submission.csv
    ans_pre = []
    ans_id = []
    df = None
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_id, (img_name, datas, labels) in enumerate(tqdm(test_loader)):
            
            datas = datas.to(device)
            outs = model(datas)

            tmp = torch.softmax(outs["comb_outs"], dim = -1)

            for i, pre_score in enumerate(tmp):
                ans_id.append(img_name[i])
                
                pre = pre_score.argmax().item()
                ans_pre.append(classes_name[pre])
            
            #if batch_id == 5:
            #   break

        df = pd.DataFrame({'id':ans_id, 'label':ans_pre})
        df.to_csv("submission.csv", index = False)
        
