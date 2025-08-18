import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import json
from tqdm import tqdm
from shapely.geometry import Polygon
from config import *
from utils import *
import torch


class Metrics:
    @staticmethod
    def validate_inputs(predictions, ground_truth):
        """
        Validate inputs for metrics calculation
        Args:
            predictions: List of predicted annotations
            ground_truth: List of ground truth annotations
        Returns:
            bool: True if inputs are valid
        """
        if len(predictions) != len(ground_truth):
            print(f"Warning: Length mismatch - predictions: {len(predictions)}, ground_truth: {len(ground_truth)}")
            return False
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            if not isinstance(pred, dict) or not isinstance(gt, dict):
                print(f"Warning: Invalid data type at index {i}")
                return False
                
        return True
    
    @staticmethod
    def calculate_metrics(predictions, ground_truth, iou_threshold=0.7):
        """
        Calculate precision, recall, F1, and mIoU
        This function computes the precision, recall, F1 score, and mean IoU for the predicted and ground truth annotations.
        Args:
            predictions: List of predicted annotations
            ground_truth: List of ground truth annotations
            iou_threshold: IoU threshold for considering a match
        Returns:
            metrics: Dictionary containing precision, recall, F1 score, and mean IoU
        """
        
        if not Metrics.validate_inputs(predictions, ground_truth):
            raise ValueError("Invalid inputs for metrics calculation")
            
        total_gt, total_pred, matched, total_iou = 0, 0, 0, 0.0

        for gt, pred in zip(ground_truth, predictions):
            pred_objs = pred.get('objects', [])
            gt_objs = gt.get('objects', [])

            total_pred += len(pred_objs)
            total_gt += len(gt_objs)

            if len(pred_objs) == 0 or len(gt_objs) == 0:
                continue

            iou_matrix = np.zeros((len(pred_objs), len(gt_objs)))
            for i, pred_obj in enumerate(pred_objs):
                pred_keypoints = pred_obj.get('keyPoints', [])
                if not pred_keypoints:
                    continue
                pred_points = pred_keypoints[0].get('points', [])
                if len(pred_points) < 3:  # Need at least 3 points for a polygon
                    continue
                    
                for j, gt_obj in enumerate(gt_objs):
                    gt_keypoints = gt_obj.get('keyPoints', [])
                    if not gt_keypoints:
                        continue
                    gt_points = gt_keypoints[0].get('points', [])
                    if len(gt_points) < 3:  
                        continue
                    
                    try:
                        iou_matrix[i, j] = polygon_iou(pred_points, gt_points)
                    except Exception as e:
                        print(f"Error calculating IoU: {e}")
                        iou_matrix[i, j] = 0.0

            pred_matched = set()
            gt_matched = set()
            
            valid_matches = []
            for i in range(len(pred_objs)):
                for j in range(len(gt_objs)):
                    if iou_matrix[i, j] >= iou_threshold:
                        valid_matches.append((i, j, iou_matrix[i, j]))
            
            valid_matches.sort(key=lambda x: x[2], reverse=True)
            
            for pred_idx, gt_idx, iou_val in valid_matches:
                if pred_idx not in pred_matched and gt_idx not in gt_matched:
                    matched += 1
                    total_iou += iou_val
                    pred_matched.add(pred_idx)
                    gt_matched.add(gt_idx)

        precision = matched / total_pred if total_pred > 0 else 0.0
        recall = matched / total_gt if total_gt > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = total_iou / matched if matched > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mean_iou': mean_iou,
            'total_predicted_roofs': total_pred,
            'total_ground_truth_roofs': total_gt,
            'matched_roofs': matched
        }


class YOLOValidationProcessor:
    
    def __init__(self, model_path):
        """
        Initialize YOLO model for validation
        Args:
            model_path: Path to the trained YOLO model file
        """

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")  
        
        self.model = YOLO(model_path)
 
    
    def process_image(self, image_path):
        """
        Process single image and return formatted result
        This function takes an image path, runs the YOLO model to get predictions, and formats the results.
        Args:
            image_path: Path to the input image
        Returns:
            result: Dictionary containing image path and detected objects with their attributes
        """

        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")  
        
        predictions = self.model.predict(str(image_path))
        objects = []
        
        if predictions and predictions[0].masks is not None:
            masks = predictions[0].masks.data.cpu().numpy()
            boxes = predictions[0].boxes
            
            original_image = cv2.imread(str(image_path))
            original_shape = original_image.shape
            
            for i, mask in enumerate(masks):
                class_label = self.model.names[int(boxes.cls[i])]
                confidence = boxes.conf[i]
                
                bbox = boxes.xywhn[i].cpu().numpy()
                
                polygon_points = mask_to_polygon(mask, original_shape)
                obj = create_object(class_label, confidence, bbox, polygon_points)
                objects.append(obj)

        return create_result(objects)

    def process_dataset_from_parquet(self, dataset_parquet_path, output_path, subset_name):
        """
        Process dataset subset from parquet file based on partition column
        This function reads a parquet file, filters by the specified subset name, processes each image,
        and saves visualizations.
        Args:
            dataset_parquet_path: Path to the parquet file containing dataset
            output_path: Directory to save visualizations
            subset_name: Name of the subset to process (e.g., 'val', 'test')
        Returns:
            results: List of processed results for the specified subset, ground truth, and DataFrame of results
        """
        
        print(f"Processing {subset_name} dataset from parquet...")
        
        df = pd.read_parquet(dataset_parquet_path)
        subset_df = df[df['partition'] == subset_name]
        
        if len(subset_df) == 0:
            print(f"No data found for partition '{subset_name}'")
            return []
        
        print(f"Found {len(subset_df)} images in {subset_name} partition")
        
        results = []
        gt = []
        df_rows = []
        for idx, row in tqdm(subset_df.iterrows(), desc=f"Processing {subset_name}", total=len(subset_df)):
            try:
                df_rows.append(row)
                image_asset_url = row.get('asset_url', '')
                if not image_asset_url:
                    continue
                image_path = image_asset_url
                if not Path(image_path).exists():
                    print(f"Image not found: {image_path}")
                    continue

                result = self.process_image(image_path)
                results.append(result)
                
                gt_result = None
                if len(row.get('annotations')) > 0:
                    gt_result = row['annotations'][0]
                    gt.append(gt_result)
            
                visualize_predictions(image_path, result, gt_result, output_path, subset_name)
                df_rows[-1]['annotations'] = [result]
                df_rows[-1]['uuid'] = idx

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                results.append(create_result([]))
                gt.append({'objects': []})
                continue

        if df_rows:
            df_results = pd.DataFrame(df_rows)
            print(f"Saved visualizations to {Path(output_path) / f'{subset_name}_visualizations'}")

        return results, gt , df_results

    def validate(self, predictions, ground_truth):
        """
        Calculate and print validation metrics
        This function computes precision, recall, F1 score, and mean IoU for the predictions against ground truth.
        Args:
            predictions: List of predicted annotations
            ground_truth: List of ground truth annotations
        Returns:
            metrics: Dictionary containing precision, recall, F1 score, and mean IoU"""
        metrics = Metrics.calculate_metrics(predictions, ground_truth)

        print("\n" + "="*50)
        print("VALIDATION SUMMARY")
        print("="*50)
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"Total Predicted Roofs: {metrics['total_predicted_roofs']}")
        print(f"Total Ground Truth Roofs: {metrics['total_ground_truth_roofs']}")
        print(f"Matched Roofs: {metrics['matched_roofs']}")
        print("="*50)
        
        return metrics



def main():
    processor = YOLOValidationProcessor(MODEL_PATH)

    val_predictions, val_ground_truth, val_df = processor.process_dataset_from_parquet(DATASET_PATH,  OUTPUT_DIR, "val")
    test_predictions, test_ground_truth, test_df = processor.process_dataset_from_parquet(DATASET_PATH,  OUTPUT_DIR, "test")

    assert len(val_predictions) == len(val_ground_truth), "Validation predictions and ground truth lengths do not match"

    try:
        output_file = Path(OUTPUT_DIR) / f"output.parquet"
        final_df = pd.concat([val_df, test_df], axis=0)
        final_df.set_index('uuid', inplace=True)
        final_df.to_parquet(output_file, index=True)

        if val_ground_truth and val_predictions:
            metrics = processor.validate(val_predictions, val_ground_truth)
            
            with open(Path(OUTPUT_DIR) / "validation_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
        else:
            print("No validation data available for metrics calculation")
            
    except Exception as e:
        print(f"Could not perform validation: {e}")
        raise e

if __name__ == "__main__":
    main()