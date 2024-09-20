# 首先安裝必要的庫
# !pip install torch torchvision tqdm
# !git clone https://github.com/ultralytics/yolov5

import torch
from IPython.display import Image, clear_output
import os

# 確保使用最新版本的YOLOv5
os.chdir('yolov5')
!git pull
os.chdir('..')

# 準備數據集
# 假設您已經準備好了數據集,並按照YOLOv5的格式組織
# 數據集結構應該如下:
# dataset/
#   images/
#     train/
#     val/
#   labels/
#     train/
#     val/

# 創建數據集配置文件 dataset.yaml
dataset_config = """
path: ../dataset  # 數據集根目錄
train: images/train  # 訓練圖像相對路徑
val: images/val  # 驗證圖像相對路徑

# 類別
nc: 1  # 類別數量
names: ['gold_coin']  # 類別名稱
"""

with open('dataset.yaml', 'w') as f:
    f.write(dataset_config)

# 訓練模型
!python yolov5/train.py --img 640 --batch 16 --epochs 100 --data dataset.yaml --weights yolov5s.pt

# 在驗證集上進行檢測
!python yolov5/detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source dataset/images/val

# 顯示檢測結果
Image(filename='runs/detect/exp/image1.jpg', width=600)

# 使用訓練好的模型進行推理
def detect_coins(image_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')
    results = model(image_path)
    results.print()  # 打印檢測結果
    results.show()  # 顯示帶有邊界框的圖像

# 使用示例
detect_coins('path_to_your_test_image.jpg')