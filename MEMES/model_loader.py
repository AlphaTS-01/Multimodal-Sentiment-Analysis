import gdown
import torch

file_id = "1zkmtxweRt6mPVqxHDVOYLWGkHYJctqC1"
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, "Multimodal_model.pth", quiet=False)
model = torch.load("Multimodal_model.pth")