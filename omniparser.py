from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor,  get_dino_model, get_yolo_model
import torch
from ultralytics import YOLO
from PIL import Image
from typing import Dict, Tuple, List
import io
import base64


config = {
    'som_model_path': 'finetuned_icon_detect.pt',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'caption_model_path': 'Salesforce/blip2-opt-2.7b',
    'draw_bbox_config': {
        'text_scale': 0.8,
        'text_thickness': 2,
        'text_padding': 3,
        'thickness': 3,
    },
    'BOX_TRESHOLD': 0.05,
    'batch_size': 32,  
    'cache_models': True,  
    'element_types': ['button', 'viewport', 'textbox', 'dropdown', 'checkbox', 'radiobutton', 'slider', 'toggle']  
}


class Omniparser(object):
    def __init__(self, config: Dict):
        self.config = config
        self.device = config['device']
        
        # Load models with caching if enabled
        if config.get('cache_models', True):
            self.som_model = get_yolo_model(model_path=config['som_model_path'])
            self.som_model.to(self.device)
        
        # Initialize model cache
        self._model_cache = {}
        self._last_image_size = None
        self._last_boxes = None

    @torch.inference_mode()
    def parse(self, image_path: str):
        print('Parsing image:', image_path)
        
        # Optimize OCR processing
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            image_path, 
            display_img=False, 
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={
                'paragraph': True,
                'text_threshold': 0.85,
                'link_threshold': 0.8,
                'canvas_size': 2560,
                'mag_ratio': 1.5
            }
        )
        text, ocr_bbox = ocr_bbox_rslt

        # Process in batches for better performance
        draw_bbox_config = self.config['draw_bbox_config']
        BOX_TRESHOLD = self.config['BOX_TRESHOLD']
        batch_size = self.config.get('batch_size', 32)

        # Check if we can reuse cached results
        current_image = Image.open(image_path)
        current_size = current_image.size
        
        if (self._last_image_size == current_size and 
            self._last_boxes is not None and 
            self.config.get('cache_models', True)):
            dino_labled_img, label_coordinates = self._last_boxes
        else:
            dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
                image_path,
                self.som_model,
                BOX_TRESHOLD=BOX_TRESHOLD,
                output_coord_in_ratio=False,
                ocr_bbox=ocr_bbox,
                draw_bbox_config=draw_bbox_config,
                caption_model_processor=None,
                ocr_text=text,
                use_local_semantics=True,
                batch_size=batch_size
            )
            
            # Cache results
            self._last_image_size = current_size
            self._last_boxes = (dino_labled_img, label_coordinates)

        # Enhanced element type detection
        element_types = self.config.get('element_types', ['button'])
        
        # Process results with element type detection
        image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
        return_list = []
        
        for i, (k, coord) in enumerate(label_coordinates.items()):
            element_info = {
                'from': 'omniparser',
                'shape': {
                    'x': coord[0],
                    'y': coord[1],
                    'width': coord[2],
                    'height': coord[3]
                }
            }
            
            # Determine element type
            if i < len(parsed_content_list):
                content = parsed_content_list[i].split(': ')[1].lower()
                element_type = 'text'  # default type
                
                # Check for specific element types
                for type_name in element_types:
                    if type_name in content:
                        element_type = type_name
                        break
                
                element_info.update({
                    'text': parsed_content_list[i].split(': ')[1],
                    'type': element_type
                })
            else:
                element_info.update({
                    'text': 'None',
                    'type': 'icon'
                })
            
            return_list.append(element_info)

        return [image, return_list]

    def __del__(self):
        # Clean up cached models
        self._model_cache.clear()
        if hasattr(self, 'som_model'):
            del self.som_model

parser = Omniparser(config)
image_path = 'examples/pc_1.png'

#  time the parser
import time
s = time.time()
image, parsed_content_list = parser.parse(image_path)
device = config['device']
print(f'Time taken for Omniparser on {device}:', time.time() - s)
