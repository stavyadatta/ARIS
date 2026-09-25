import cv2
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import base64
import io
import requests

class InternVL:
    def __init__(self, 
                 model_path='OpenGVLab/InternVL2-8B', 
                 input_size=448, 
                 max_num=12, 
                 generation_config=None):
        """
        Initialize the InternVL model for image description.
        
        Args:
            model_path (str): Path to the InternVL model
            input_size (int): Size to resize input images
            max_num (int): Maximum number of image splits
            generation_config (dict): Configuration for text generation
        """
        # Default generation configuration
        self.generation_config = generation_config or {
            'max_new_tokens': 1024, 
            'do_sample': False
        }
        
        # Image preprocessing parameters
        self.input_size = input_size
        self.max_num = max_num
        
        # Model and tokenizer
        self.model, self.tokenizer = self._load_model(model_path)
        
        # Image transformation
        self.transform = self._build_transform(input_size)
    
    def _load_model(self, path):
        """
        Load the InternVL model and tokenizer.
        
        Args:
            path (str): Path to the model
        
        Returns:
            tuple: Model and tokenizer
        """
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().cuda()
        
        tokenizer = AutoTokenizer.from_pretrained(
            path, 
            trust_remote_code=True, 
            use_fast=False
        )
        
        return model, tokenizer
    
    def _build_transform(self, input_size):
        """
        Build image transformation pipeline.
        
        Args:
            input_size (int): Size to resize images
        
        Returns:
            torchvision.transforms.Compose: Image transformation
        """
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """
        Find the closest aspect ratio for image preprocessing.
        
        Args:
            aspect_ratio (float): Original image aspect ratio
            target_ratios (list): Possible aspect ratios
            width (int): Original image width
            height (int): Original image height
            image_size (int): Target image size
        
        Returns:
            tuple: Best aspect ratio
        """
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        
        return best_ratio
    
    def _dynamic_preprocess(self, image):
        """
        Preprocess image with dynamic splitting.
        
        Args:
            image (PIL.Image): Input image
        
        Returns:
            list: Processed image splits
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        target_ratios = set(
            (i, j) for n in range(1, self.max_num + 1) 
            for i in range(1, n + 1) 
            for j in range(1, n + 1) 
            if i * j <= self.max_num and i * j >= 1
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, self.input_size
        )
        
        target_width = self.input_size * target_aspect_ratio[0]
        target_height = self.input_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        for i in range(blocks):
            box = (
                (i % (target_width // self.input_size)) * self.input_size,
                (i // (target_width // self.input_size)) * self.input_size,
                ((i % (target_width // self.input_size)) + 1) * self.input_size,
                ((i // (target_width // self.input_size)) + 1) * self.input_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        # Add thumbnail if multiple splits
        if len(processed_images) != 1:
            thumbnail_img = image.resize((self.input_size, self.input_size))
            processed_images.append(thumbnail_img)
        
        return processed_images
    
    def __call__(self, image, question='<image>\nPlease describe the image shortly.'):
        """
        Generate description for the input image.
        
        Args:
            image (numpy.ndarray or str or PIL.Image): Input image
            question (str, optional): Prompt for image description
        
        Returns:
            str: Image description
        """
        # Convert input to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Preprocess images
        images = self._dynamic_preprocess(image)
        pixel_values = [self.transform(img) for img in images]
        pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()
        
        # Generate description
        response = self.model.chat(
            self.tokenizer, 
            pixel_values, 
            question, 
            self.generation_config
        )
        
        return response

# Example usage
if __name__ == "__main__":
    # Initialize the streamer
    streamer = InternVL()
    
    # Describe an image
    image_path = '/workspace/database/face_db/some.png'
    description = streamer(image_path)
    print(description)
