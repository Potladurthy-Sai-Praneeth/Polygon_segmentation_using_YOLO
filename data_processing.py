import pandas as pd
import cv2
import numpy as np
import os
import yaml
from pathlib import Path
import albumentations as A
from tqdm import tqdm
import argparse
from config import *
from utils import *

class YOLODataProcessor:
    def __init__(self, parquet_path, output_dir):
        """
        Initialize YOLO data processor
        
        Args:
            parquet_path: Path to the parquet file containing annotations
            output_dir: Directory to save processed YOLO dataset
        """
        assert parquet_path is not None, "Parquet path must be provided"
        assert output_dir is not None, "Output directory must be provided"
        assert Path(parquet_path).exists(), f"Parquet file {parquet_path} does not exist"
        
        self.parquet_path = parquet_path
        self.output_dir = Path(output_dir)
        self.data = pd.read_parquet(parquet_path)
        
        self.setup_directories()
        self.imgsz = None  
        
        self.setup_augmentations()

    def setup_directories(self):
        """
        Create YOLO dataset directory structure.
        As the dataset is limited we can use CPU to perform augmentations and store the images locally.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        for split in ['train', 'val']:
            os.makedirs(self.output_dir / 'images' / split, exist_ok=True)
            os.makedirs(self.output_dir / 'labels' / split, exist_ok=True)

    def setup_augmentations(self):
        """
        Setup augmentation pipelines for training data
        Since dataset is very limited we rely on heavy augmentations to generate more data.
        Heavy augmentations are applied to the training set, while light augmentations are applied to the validation set.
        """
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=45, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(p=0.3),
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        self.val_transform = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


    def apply_augmentation(self, image, normalized_keypoints, split):
        """
        Apply augmentation to image and keypoints
        Args:
            image: Input image to augment
            normalized_keypoints: List of keypoints in normalized format (0-1)
            split: Dataset split ('train', 'val')
        Returns:
            aug_image: Augmented image
            aug_normalized_keypoints: Augmented keypoints in normalized format (0-1)
        """
        
        if split == 'test':
            return image, normalized_keypoints
        
        image_height, image_width = image.shape[:2]
        if self.imgsz is None:
            self.imgsz = max(image_height, image_width)
        
        pixel_keypoints = convert_to_pixel_coordinates(normalized_keypoints, image_width, image_height)
        
        all_keypoints = []
        keypoint_counts = []
        
        for keypoint_group in pixel_keypoints:
            all_keypoints.extend(keypoint_group)
            keypoint_counts.append(len(keypoint_group))
        
        try:
            if split == 'train':
                transformed = self.train_transform(image=image, keypoints=all_keypoints)
            else:  
                transformed = self.val_transform(image=image, keypoints=all_keypoints)
            
            aug_image = transformed['image']
            aug_keypoints = transformed['keypoints']
            
            aug_pixel_keypoints = []
            start_idx = 0
            
            for count in keypoint_counts:
                end_idx = start_idx + count
                group_keypoints = []
                for i in range(start_idx, min(end_idx, len(aug_keypoints))):
                    group_keypoints.append(aug_keypoints[i])
                
                while len(group_keypoints) < count:
                    if group_keypoints:
                        if len(group_keypoints) >= 2:
                            first_point = group_keypoints[0]
                            last_point = group_keypoints[-1]
                            new_point = [(first_point[0] + last_point[0]) / 2, 
                                       (first_point[1] + last_point[1]) / 2]
                            group_keypoints.append(new_point)
                        else:
                            group_keypoints.append(group_keypoints[-1]) 
                    else:
                        group_keypoints.append([0, 0])
                
                aug_pixel_keypoints.append(group_keypoints)
                start_idx = end_idx
            
            aug_height, aug_width = aug_image.shape[:2]
            aug_normalized_keypoints = convert_to_normalized_coordinates(aug_pixel_keypoints, aug_width, aug_height)

            return aug_image, aug_normalized_keypoints
            
        except Exception as e:
            print(f"Augmentation failed: {e}")
            return image, normalized_keypoints
    
    def process_dataset(self, augmentation_factor=3):
        """
        Process entire dataset with augmentations
        This function processes the dataset by applying augmentations and saving the images and keypoints.
        It handles both training and validation splits, applying heavy augmentations to the training set and light augmentations to the validation set.
        It saves the processed images and their corresponding keypoints in the specified output directory.
        Args:
            augmentation_factor: Number of augmentations to apply per image in training set
        Returns:
            None
        """
        print("Processing dataset...")
        
        for split in ['train', 'val']:
            split_data = self.data[self.data['partition'] == split]
            print(f"Processing {split} split: {len(split_data)} images")
            
            for idx, row in tqdm(split_data.iterrows(), total=len(split_data)):
                img_path = os.path.join(os.getcwd(), row['asset_url'])
                annotations = row['annotations'][0]
                
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not load image {img_path}")
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                normalized_keypoints = extract_keypoints(annotations,flatten=True)
                
                self.save_sample(image, normalized_keypoints, f"{idx}_original", split)
                
                if split == 'train':
                    for aug_idx in range(augmentation_factor):
                        try:
                            aug_image, aug_keypoints = self.apply_augmentation(
                                image, normalized_keypoints, split
                            )
                            self.save_sample(
                                aug_image, aug_keypoints, 
                                f"{idx}_aug_{aug_idx}", split
                            )
                        except Exception as e:
                            print(f"Warning: Augmentation failed for {idx}_aug_{aug_idx}: {e}")
                elif split == 'val':
                    for aug_idx in range(3):
                        try:
                            aug_image, aug_keypoints = self.apply_augmentation(
                                image, normalized_keypoints, split
                            )
                            self.save_sample(
                                aug_image, aug_keypoints, 
                                f"{idx}_aug_{aug_idx}", split
                            )
                        except Exception as e:
                            print(f"Warning: Augmentation failed for {idx}_aug_{aug_idx}: {e}")
    
    def save_sample(self, image, normalized_keypoints_list, filename, split):
        """
        Save image and corresponding YOLO label with normalized coordinates
        This function saves the image in the specified output directory and creates a label file with normalized keypoints.
        It ensures that the keypoints are formatted correctly for YOLO format, which requires normalized coordinates (0-1 range) for each keypoint.
        Args:
            image: Image to save
            normalized_keypoints_list: List of keypoints in normalized format (0-1)
            filename: Filename to save the image and label
            split: Dataset split ('train', 'val')
        Returns:
            None
        """
        img_path = self.output_dir / 'images' / split / f"{filename}.jpg"
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_path), image_bgr)
        
        label_path = self.output_dir / 'labels' / split / f"{filename}.txt"
        with open(label_path, 'w') as f:
            for normalized_keypoints in normalized_keypoints_list:
                if len(normalized_keypoints) >= 6:  # At least 3 points (6 coordinates)
                    coords_str = ' '.join([f"{coord:.6f}" for coord in normalized_keypoints])
                    f.write(f"0 {coords_str}\n")  # Class 0 for roof
    
    def create_yaml_config(self):
        """
        Create YOLO dataset configuration file
        This function generates a YAML configuration file for the YOLO dataset, specifying paths to training, validation, and test images,
        number of classes, and class names. The configuration file is essential for training YOLO models
        as it provides the necessary metadata about the dataset structure.
        """
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,  # Number of classes
            'names': ['roof'],
            'imgsz': self.imgsz,
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Dataset configuration saved to: {yaml_path}")
        return yaml_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset for YOLO")
    parser.add_argument("--augmentation_factor", type=int, default=5, help="Number of augmentations per image in training set")
    
    args = parser.parse_args()
    AUGMENTATION_FACTOR = args.augmentation_factor

    processor = YOLODataProcessor(DATASET_PATH, OUTPUT_DIR)
    processor.process_dataset(augmentation_factor=AUGMENTATION_FACTOR)
    yaml_path = processor.create_yaml_config()
    
    print("Dataset processing completed!")
    print(f"Dataset saved to: {OUTPUT_DIR}")
    print(f"YAML config: {yaml_path}")
