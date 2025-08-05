import torch
import csv
from net import FlexiCell
from utils import *
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
from torch import nn
import shutil
from track import predict_dataset_2
from generate_trace import get_trace, get_video  # Uses modified generate_trace
from matplotlib import pyplot as plt
import os
import cv2

def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    img = clahe.apply(img)
    return img

def enhance(img):
    img = np.clip(img * 1.2, 0, 255)
    img = img.astype(np.uint8)
    return img

def test(test_path, result_path):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    model= FlexiCell(n_channels=1, n_classes=2)
    model.eval()  
    model = model.to(device)  # Move model to device (GPU or CPU)

    # Load model state
    print(f"Loading model from checkpoints/CP_epoch50.pth")
    try:
        # Load with map_location to ensure the model loads on CPU
        checkpoint = torch.load('checkpoints/CP_epoch50.pth', map_location=device)
        
        # Handle state dict keys with module prefix
        new_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith("module."):
                new_state_dict[key[7:]] = value  # Remove `module.` prefix
            else:
                new_state_dict[key] = value  # Keep the key as is
        
        model.load_state_dict(new_state_dict)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    #model.load_state_dict(torch.load('checkpoints/CP_epoch50.pth', map_location=device))
    
    test_data = TestDataset(test_path, transform=x_transforms)
    dataloader = DataLoader(test_data, batch_size=1, num_workers=2)
    with torch.no_grad():
        for index, (x, img_path) in enumerate(dataloader):
        #for index, x in enumerate(dataloader):
            x = x.to(device)  # Move input to the device
            y = model(x)
            y = y.cpu()  # Move output back to CPU
            y = torch.squeeze(y)
            img_y = torch.sigmoid(y).numpy()
            img_y = (img_y * 255).astype(np.uint8)  # Convert to uint8 format
            # Get original image to match dimensions
            original_img = cv2.imread(img_path[0], -1)
            if original_img is not None:
                h, w = original_img.shape
                # Resize prediction to match original dimensions
                img_y = cv2.resize(img_y, (w, h), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(result_path, "predict_" + str(index).zfill(6) + '.tif'), img_y)
            else:
                print(test_path, "prediction finish!")

def process_img():
    img_root = "data/test/"
    n = len(os.listdir(img_root))
    for i in range(n):
        img_path = os.path.join(img_root, str(i).zfill(6) + ".tif")
        img = cv2.imread(img_path, -1)
        img = np.uint8(np.clip((0.02 * img + 60), 0, 255))
        cv2.imwrite(img_path, img)

def process_predictResult(source_path, result_path):
    if not os.path.isdir(result_path):
        print('Creating RES directory')
        os.mkdir(result_path)

    names = os.listdir(source_path)
    names = [name for name in names if '.tif' in name]
    names.sort()

    for name in names:
        predict_result = cv2.imread(os.path.join(source_path, name), -1)
        # Binarize image
        ret, predict_result = cv2.threshold(predict_result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Connected components and labeling
        ret, markers = cv2.connectedComponents(predict_result)
        
        # Convert markers to uint16 for saving as .tif
        markers = np.uint16(markers)
        
        cv2.imwrite(os.path.join(result_path, name), markers)

def useAreaFilter(img, area_size):
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_new = np.stack((img, img, img), axis=2)

    for cont in contours:
        area = cv2.contourArea(cont)
        if area < area_size:
            img_new = cv2.fillConvexPoly(img_new, cont, (0, 0, 0))

    img = img_new[:, :, 0]
    return img

def delete_file(path):
    if not os.path.isdir(path):
        print(path, "does not exist!")
        os.mkdir(path)
        return
    file_list = os.listdir(path)
    for file in file_list:
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    print(path, "has been cleaned!")

def createFolder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(path, "has been created.")
    else:
        print(path, "has already existed.")

if __name__ == "__main__":
    test_folders = os.listdir("testing")
    test_folders = [os.path.join("testing/", folder) for folder in test_folders]
    test_folders.sort()
    
    for folder in test_folders:
        test_path = os.path.join(folder, "test")
        test_result_path = os.path.join(folder, "test_result")
        res_path = os.path.join(folder, "res")
        res_result_path = os.path.join(folder, "res_result")
        track_result_path = os.path.join(folder, "track_result")
        trace_path = os.path.join(folder, "trace")

        createFolder(test_result_path)
        createFolder(res_path)
        createFolder(res_result_path)
        createFolder(track_result_path)
        createFolder(trace_path)

        test(test_path, test_result_path)
        process_predictResult(test_result_path, res_path)

        result = os.listdir(res_path)
        for picture in result:
            image = cv2.imread(os.path.join(res_path, picture), -1)
            image = useAreaFilter(image, 100)
            cv2.imwrite(os.path.join(res_result_path, picture), image)
        
        print("Starting tracking")
        # Track
        predict_result = res_result_path
        predict_dataset_2(predict_result, track_result_path)

        # Uses modified generate_trace (no bounding boxes, trajectory lines, division info)
        get_trace(test_path, track_result_path, trace_path)
        get_video(trace_path)
