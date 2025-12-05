import torch
import torchvision
import cv2

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cv2 version:", cv2.__version__)
print("cuda available:", torch.cuda.is_available())
