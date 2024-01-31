import os
import argparse
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from lang_sam import LangSAM
import numpy as np
from tqdm import tqdm
import jsonlines

input_dir = '/raid/maruf/Bird/images/'
output_file = 'output/bounding_box_annotation.jsonl'


img_name_list = os.listdir(input_dir)
img_name_list = [img_name for img_name in img_name_list if img_name.split('.')[-1]=='jpg'][:5000]

TRAIT_LIST=["beak", "eye", "wings", "head", "tail"]
PROMPT_LIST=["beak of bird", "eye of bird", "wings of bird", "head of bird", "tail of bird"]

# LangSAM model : GroundingDINO + SAM
model = LangSAM()

writer = jsonlines.open(output_file, mode='w')

for img_name in tqdm(img_name_list):
    
    img = Image.open(os.path.join(input_dir, f'{img_name}'))
    image_pil = img.convert("RGB")


    for trait, prompt_trait in zip(TRAIT_LIST, PROMPT_LIST):

        text_prompt, BOX_THRESHOLD = prompt_trait, 0.10
        masks, boxes, phrases, logits = model.predict(image_pil, text_prompt, box_threshold=BOX_THRESHOLD) 
        
        if len(boxes) == 0:
            continue

        output_dict = dict()
        output_dict['image-name'] = img_name
        output_dict['trait'] = trait

        # Draw bounding boxes on the image
        number_of_option = 0

        for box, phrase, logit in zip(boxes, phrases, logits):
            if trait not in phrase:
                continue
            else:
                output_dict[f'Option-{number_of_option}-box'] = f"{box.detach().cpu().numpy()}"
                output_dict[f'Option-{number_of_option}-phrase'] = phrase
                output_dict[f'Option-{number_of_option}-logit'] = logit.item()
                number_of_option += 1
                output_dict['number_of_options'] = number_of_option
                
                if number_of_option > 6:
                    break
                    
        writer.write(output_dict)
        
    writer.close()
    writer = jsonlines.open(output_file, mode='a')
    
writer.close()
