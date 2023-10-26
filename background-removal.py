import os
import argparse
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from lang_sam import LangSAM
import numpy as np

# image name from images in input directory

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", "-d", type=str, default='input/', help="path to image directory")
parser.add_argument("--image_name", "-i", type=str, default='INHS_FISH_005052.jpg', help="image name with extension")
parser.add_argument("--out_dir", "-o", type=str, default='output/', help="path to output")
args = parser.parse_args()


img_name = args.image_name
input_dir = args.image_dir
out_dir = args.out_dir

# loading the pillow image

img = Image.open(os.path.join(input_dir, f'{img_name}'))
image_pil = img.convert("RGB")


# running the GroundedSAM

text_prompt, BOX_THRESHOLD = "fish", 0.30
model = LangSAM()
masks, boxes, phrases, logits = model.predict(image_pil, text_prompt, box_threshold=BOX_THRESHOLD)  

# check whether the GroundedSAM finds a fish mask or not

if len(masks) == 0:
    
    print('GroundedSAM is not able to find a fish in the image.')
    
else:
    
    # only condisering the first seg map
    
    mask = masks[0].detach().cpu().numpy()
    
    # bg-removed image -> mod_img
    img_array = np.asarray(img).copy()
    z_r, z_c = np.where(mask == 0)
    img_array[z_r, z_c] = np.array([255, 255, 255])
    mod_img = Image.fromarray(img_array.astype(np.uint8))
    
    # save the mask and mod_img
    mask_pil = Image.fromarray((mask*255).astype(np.uint8))
    
    mask_dir = os.path.join(out_dir, 'mask')
    os.makedirs(mask_dir, exist_ok=True)
    
    bg_dir = os.path.join(out_dir, 'bg_removed')
    os.makedirs(bg_dir, exist_ok=True)
    
    mask_pil.save(os.path.join(mask_dir, f'{img_name}'))
    mod_img.save(os.path.join(bg_dir, f'{img_name}'))
    
    print('Mask area: {:.2f}% of the entire image'.format(100*mask.sum()/(mask.shape[0]* mask.shape[1])))
    