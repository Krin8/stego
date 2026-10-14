import numpy as np
import cv2
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from utils import unnorm

def dense_crf(image_tensor: torch.FloatTensor, output_logits: torch.FloatTensor):
    """
    Pure Python / OpenCV implementation of Mean Field Dense CRF.
    Avoids the broken pydensecrf C++ bindings.
    """
    image = np.array(VF.to_pil_image(unnorm(image_tensor)))[:, :, ::-1] # BGR uint8
    H, W = image.shape[:2]
    
    output_logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze()
    
    C = output_logits.shape[0]
    U = -output_logits.cpu().numpy() # Unary potentials (negative logits)
    
    Q = F.softmax(output_logits, dim=0).cpu().numpy() # Initial Probability Q
    
    MAX_ITER = 10
    POS_W = 3
    POS_XY_STD = 1.0
    Bi_W = 4
    Bi_XY_STD = 67.0
    Bi_RGB_STD = 3.0
    
    for i in range(MAX_ITER):
        spatial_out = np.zeros_like(Q)
        bilateral_out = np.zeros_like(Q)
        
        image_f32 = image.astype(np.float32)
        for c in range(C):
            spatial_out[c] = ndi.gaussian_filter(Q[c], sigma=POS_XY_STD)
            bilateral_out[c] = cv2.ximgproc.jointBilateralFilter(
                image_f32, Q[c].astype(np.float32), 
                -1, Bi_RGB_STD, Bi_XY_STD
            )
            
        # Potts model compatibility transform
        spatial_msg = spatial_out.sum(axis=0) - spatial_out
        bilateral_msg = bilateral_out.sum(axis=0) - bilateral_out
        
        pairwise = POS_W * spatial_msg + Bi_W * bilateral_msg
        
        # Local Update
        Q_logits = -U - pairwise
        
        # Normalize (Softmax)
        Q_logits = Q_logits - Q_logits.max(axis=0, keepdims=True)
        Q = np.exp(Q_logits)
        Q = Q / Q.sum(axis=0, keepdims=True)
        
    return Q
