import cv2
from V7 import run_swarm
from V8 import run_Hill
from V9 import run_genetic
from V10 import run_Differential

import numpy as np

from skimage.metrics import structural_similarity as ssim

# get_dataset(r'C:\Users\karti\Desktop\Studies\AI\Project\motion_blurred', r'C:\Users\karti\Desktop\Studies\AI\Project\sharp', 256, 256, r'./256/motion')

def laplace(image):
    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Compute the variance of the Laplacian
    variance = np.var(laplacian)

    # Return the Blurriness Index
    return variance

# create_model(10, r'./256/motion', r'./motion256V1')
print("\nSwarm: ")
swarm = run_swarm()
print("\nHill: ")
hill = run_Hill()
print("\nGenetic: ")
genetic = run_genetic()
print("\nDifferential: ")
differential = run_Differential()

image = cv2.imread('./gt.jpg', cv2.IMREAD_GRAYSCALE)
ssim_none = ssim(image, image)
ssim_Swarm = ssim(image, swarm)
ssim_Hill = ssim(image, hill)
ssim_genetic = ssim(image, genetic)
ssim_diff = ssim(image, differential)

print(f"The SSIMs are: {ssim_none}:{laplace(image)}, {ssim_Swarm}:{laplace(swarm)}, {ssim_Hill}:{laplace(hill)}, {ssim_genetic}:{laplace(genetic)}, {ssim_diff}:{laplace(differential)}")
