import numpy as np
import math
from PIL import Image
import os, sys
from nanba import raywarp_kernel

rt = 8.0 # throat radius

WIDTH = 800
HEIGHT = 600
aspect = float(WIDTH) / float(HEIGHT)

COMPUTE_WIDTH = 800
COMPUTE_HEIGHT = int(COMPUTE_WIDTH / aspect)

cam_pos = np.array([0.0, -2.0, 40.0]) 
cam_target = np.array([0.0, 0.0, 0.0])
fov_y_deg = 60.0

if __name__ == "__main__":
    
    try:
        sp1_img_pil = Image.open("image1.png").convert("RGB")
        SPACE1_IMAGE_DATA = np.array(sp1_img_pil, dtype=np.uint8)
        SPACE1_HEIGHT, SPACE1_WIDTH, _ = SPACE1_IMAGE_DATA.shape

        sp2_img_pil = Image.open("image2.jpg").convert("RGB")
        SPACE2_IMAGE_DATA = np.array(sp2_img_pil, dtype=np.uint8)
        SPACE2_HEIGHT, SPACE2_WIDTH, _ = SPACE2_IMAGE_DATA.shape

        print("Successfully loaded image files.")

    except FileNotFoundError as e:
        print(f"Exitting: {e}. Can't do wormholing without two images.")
        sys.exit()

    fwd = cam_target - cam_pos # vector 
    fwd_norm = np.linalg.norm(fwd)
    fwd = fwd / fwd_norm # unit vector

    world_up = np.array([0.0, 1.0, 0.0])
    
    right = np.cross(fwd, world_up)
    right_norm = np.linalg.norm(right)
    right = right / right_norm

    up = np.cross(right, fwd)

    # View frustum parameters
    # aspect = float(WIDTH) / float(HEIGHT) # Moved up
    tan_half_fov = math.tan(math.radians(fov_y_deg)/2)
    
    compute_pixels = np.zeros((COMPUTE_HEIGHT, COMPUTE_WIDTH, 3), dtype=np.uint8) # This will hold the rendered image
    
    print(f"Starting render at {COMPUTE_WIDTH}x{COMPUTE_HEIGHT}...")
    
    raywarp_kernel(compute_pixels, COMPUTE_WIDTH, COMPUTE_HEIGHT,
                    cam_pos, right, up, fwd,
                    tan_half_fov, aspect,
                    rt, 1,
                    SPACE1_IMAGE_DATA, SPACE1_WIDTH, SPACE1_HEIGHT,
                    SPACE2_IMAGE_DATA, SPACE2_WIDTH, SPACE2_HEIGHT)
    
    print("Render complete. Upscaling...")
    
    img_low = Image.fromarray(compute_pixels, 'RGB')
    img_high = img_low.resize((WIDTH, HEIGHT), resample=Image.NEAREST)
    
    base_filename = "wormhole"
    extension = ".png"
    counter = 1
    output_filename = f"{base_filename}{extension}"
    while os.path.exists(output_filename):
        output_filename = f"{base_filename}_{counter}{extension}"
        counter += 1
    
    img_high.save(output_filename)
    print(f"Image saved to {output_filename}")
    img_high.show()