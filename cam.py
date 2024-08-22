import pandas as pd
import numpy as np
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
import cv2

def get_CAM_Image(df, pl_model, n_samples, cam_img_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = pl_model.model
    model.to(device)

    # Get the last conv layer of model
    target_layers = [model.feature[-1]]  

    df_sample = df.sample(n_samples)
    for i, row in enumerate(df_sample.itertuples(index=False)):
        # Load the RGB image from the .npy file
        rgb_image_path = cam_img_dir / row.file_name
        rgb_image = np.load(rgb_image_path)
        
        # Normalize the image
        rgb_image = np.float32(rgb_image) / 255.0
        input_tensor = preprocess_image(rgb_image, mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]).to(device)
        
        target = [ClassifierOutputTarget(row.label)]

        # Create a GradCAM instance
        with GradCAM(model=model, target_layers=target_layers) as cam:
            cam.batch_size = 1
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=target,
                                aug_smooth=True,  
                                eigen_smooth=True)
            
            # Create the cam and show it on the image
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            
            # Save the image
            cam_image_path = cam_img_dir / f"Cam{i}.png"
            cv2.imwrite(str(cam_image_path), cam_image)

        


    



