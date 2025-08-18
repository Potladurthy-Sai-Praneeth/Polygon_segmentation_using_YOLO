import numpy as np
import cv2
import json
from tqdm import tqdm
from shapely.geometry import Polygon
import pandas as pd
from pathlib import Path


def extract_keypoints(annotations,flatten=False,is_object=False,get_pixel_coordinates=False,image_width=None,image_height=None):
        """
        Extract already normalized keypoints from annotations.
        This functions parses the annotaions and extracts the keypoints which are already normalized (0-1).
        Args:
            annotations: Dictionary containing annotations for the image
            flatten: If True, flattens the keypoints into a single list. This flag is helpful when writing all keypoints for YOLO format.
            is_object: If True, annotations correspond to annotations['objects'] directly.
            get_pixel_coordinates: If True, converts normalized coordinates to pixel coordinates based on image dimensions.
            image_width: Width of the image, required if get_pixel_coordinates is True.
            image_height: Height of the image, required if get_pixel_coordinates is True.
        Returns:
            all_keypoints: List of keypoints for each object in the image representing the vertex of a polygon. May be flattened or not based on the flatten flag.
        """
        all_keypoints = []
        if is_object:
            data = annotations
        else:
            data = annotations['objects']
        for obj in data:
            keypoints_array = obj['keyPoints']
            normalized_points = []
            for point in keypoints_array[0]['points']:
                if get_pixel_coordinates and image_width is not None and image_height is not None:
                    # Convert normalized coordinates to pixel coordinates
                    point['x'] = int(point['x'] * image_width)
                    point['y'] = int(point['y'] * image_height)
                else:
                    # Ensure the point is normalized (0-1 range)
                    point['x'] = max(0, min(1, point['x']))
                    point['y'] = max(0, min(1, point['y']))
                if flatten:
                    normalized_points.extend([point['x'], point['y']])
                else:
                    normalized_points.append([point['x'], point['y']])
            all_keypoints.append(normalized_points)
        return all_keypoints


def convert_to_pixel_coordinates(normalized_keypoints, image_width, image_height):
        """
        Convert normalized keypoints to pixel coordinates for augmentation.
        A helper function to convert normalized keypoints (0-1 range) to pixel coordinates based on image dimensions.
        Args:
            normalized_keypoints: List of keypoints in normalized format (0-1)
            image_width: Width of the image
            image_height: Height of the image
        Returns:
            pixel_keypoints: List of keypoints in pixel format
        """
        pixel_keypoints = []
        for keypoint_list in normalized_keypoints:
            pixel_points = []
            for i in range(0, len(keypoint_list), 2):
                pixel_x = int(keypoint_list[i] * image_width)
                pixel_y = int(keypoint_list[i+1] * image_height)
                pixel_points.append([pixel_x, pixel_y])
            pixel_keypoints.append(pixel_points)
        return pixel_keypoints


def convert_to_normalized_coordinates(pixel_keypoints, image_width, image_height):
    """
    Convert pixel keypoints back to normalized coordinates.
    A helper function to convert pixel keypoints back to normalized format (0-1 range).
    Args:
        pixel_keypoints: List of keypoints in pixel format
        image_width: Width of the image
        image_height: Height of the image
    Returns:
        normalized_keypoints: List of keypoints in normalized format (0-1)
    """
    normalized_keypoints = []
    for pixel_points in pixel_keypoints:
        normalized_points = []
        for point in pixel_points:
            if len(point) < 2 or not all(isinstance(coord, (int, float)) for coord in point[:2]):
                continue  
            norm_x = point[0] / image_width
            norm_y = point[1] / image_height
            norm_x = max(0, min(1, norm_x))
            norm_y = max(0, min(1, norm_y))
            normalized_points.extend([norm_x, norm_y])
        normalized_keypoints.append(normalized_points)
    return normalized_keypoints


def mask_to_polygon(mask, original_image_shape):
    """
    Convert binary mask to normalized polygon points.
    This function finds the largest contour in the binary mask and approximates it to a polygon.
    It then normalizes the polygon points based on the original image dimensions.
    Args:
        mask: Binary mask of the object
        original_image_shape: Shape of the original image (height, width, channels)
    Returns:
        points: Array of normalized polygon points in the format we want to save.
        If no contours are found, returns an empty array.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([], dtype=object)
    
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    img_height, img_width = original_image_shape[:2]
    mask_height, mask_width = mask.shape
    
    scale_x = img_width / mask_width
    scale_y = img_height / mask_height
    
    points = []
    for point in approx_polygon:
        x, y = point[0]
        img_x = x * scale_x
        img_y = y * scale_y
        points.append({
            'category': None, 'classLabel': None, 'confidence': None,
            'visible': None, 'x': float(img_x / img_width), 'y': float(img_y / img_height), 'z': None
        })
    
    return np.array(points, dtype=object)



def create_object(class_label, confidence, bbox, polygon_points):
    """
    Create object structure from YOLO predictions
    This function creates an object structure that includes the class label, confidence score, bounding box, and polygon points.
    Args:
        class_label: Class label of the detected object
        confidence: Confidence score of the detection
        bbox: Bounding box coordinates in YOLO format (x, y, width, height)
        polygon_points: Normalized polygon points representing the object
    Returns:
        A dictionary representing the object with its attributes."""
    x, y, w, h = bbox  
    
    keypoints = np.array([{
        'category': None,
        'points': polygon_points,
        'type': None
    }], dtype=object)
    
    return {
        'category': None, 'classLabel': class_label, 'confidence': float(confidence),
        'height': float(h), 'keyPoints': keypoints, 'texts': None,
        'user_review': 'approved', 'width': float(w), 'x': float(x), 'y': float(y)
    }
   
def create_result(objects):
    """
    This function creates the final result structure that includes the image path and a list of detected objects.
    Args:
        objects: List of detected objects
    Returns:
        A dictionary representing the result with image path and objects.
    """
    return {
        'classes': np.array([], dtype=object),
        'embeddings': np.array([], dtype=object),
        'keyPoints': np.array([], dtype=object),
        'objects': np.array(objects, dtype=object),
        'source': None,
        'source_model_uuid': None,
        'texts': np.array([], dtype=object),
        'type': 'prediction',
        'user_review': 'pending_approval'
    }

def polygon_iou(pred_points, gt_points):
    """
    Calculate IoU between two polygons
    This function calculates the Intersection over Union (IoU) between two polygons represented by their vertex points.
    Args:
        pred_points: List of points representing the predicted polygon
        gt_points: List of points representing the ground truth polygon
    Returns:
        IoU value as a float. Returns 0.0 if either polygon is invalid or if there are not enough points to form a polygon.
    """
    try:
        pred_coords = [(p['x'], p['y']) for p in pred_points if len(pred_points) > 0]
        gt_coords = [(p['x'], p['y']) for p in gt_points if len(gt_points) > 0]
        
        if len(pred_coords) < 3 or len(gt_coords) < 3:
            return 0.0
        
        pred_poly = Polygon(pred_coords)
        gt_poly = Polygon(gt_coords)
        
        if not pred_poly.is_valid or not gt_poly.is_valid:
            return 0.0
        
        intersection = pred_poly.intersection(gt_poly).area
        union = pred_poly.union(gt_poly).area
        return intersection / union if union > 0 else 0.0
    except:
        return 0.0


def draw_polygon_from_keypoints(image, keypoints, options=None):
    """
    Draw polygon from keypoints similar to testing notebook
    This function draws a polygon on the image based on the provided keypoints.
    It can also fill the polygon and customize the line color and thickness.
    Args:
        image: Input image on which to draw the polygon
        keypoints: List of keypoints representing the polygon vertices
        options: Dictionary containing options for drawing the polygon
    Returns:
        result_image: Image with the overlayed polygon 
    """
    if options is None:
        options = {}
    
    line_color = options.get('line_color', (0, 255, 0))
    line_thickness = options.get('line_thickness', 2)
    fill_polygon = options.get('fill_polygon', True)
    fill_color = options.get('fill_color', (0, 255, 0))
    fill_alpha = options.get('fill_alpha', 0.3)
    
    result_image = image.copy()
    
    if len(keypoints) < 3:
        return result_image
    
    pts = np.array(keypoints)
    
    if fill_polygon:
        overlay = result_image.copy()
        cv2.fillPoly(overlay, [pts], fill_color)
        result_image = cv2.addWeighted(result_image, 1 - fill_alpha, overlay, fill_alpha, 0)
    
    cv2.polylines(result_image, [pts], True, line_color, line_thickness)
    
    return result_image


def visualize_predictions(image_path, prediction_result, gt_result, output_dir, subset_name):
    """
    Visualize predictions with polygon drawing for validation and test
    This function reads an image, draws polygons for ground truth and predictions, and saves the visualized image.
    Args:
        image_path: Path to the input image
        prediction_result: Dictionary containing prediction results with objects and their keypoints
        gt_result: Dictionary containing ground truth results with objects and their keypoints
        output_dir: Directory to save the visualized images
        subset_name: Name of the subset ('val' or 'test')
    Returns:
        None
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read image: {image_path}")
            return
        
        image_height, image_width = image.shape[:2]
        
        if subset_name == "val":
            # Here we visualize both ground truth and predictions
            combined_image = np.zeros((image_height, image_width * 2, 3), dtype=np.uint8)
            
            gt_image = image.copy()
            if gt_result and 'objects' in gt_result:
                gt_keypoints = extract_keypoints(gt_result['objects'], is_object=True, get_pixel_coordinates=True, image_width=image_width, image_height=image_height)
                gt_options = {
                    'line_color': (0, 255, 0),  # Green
                    'line_thickness': 3,
                    'fill_polygon': True,
                    'fill_color': (0, 255, 0),
                    'fill_alpha': 0.3
                }
                for points in gt_keypoints:
                    gt_image = draw_polygon_from_keypoints(gt_image, points, gt_options)
            
            pred_image = image.copy()
            if prediction_result and 'objects' in prediction_result:
                pred_keypoints = extract_keypoints(prediction_result['objects'], is_object=True, get_pixel_coordinates=True, image_width=image_width, image_height=image_height)
                pred_options = {
                    'line_color': (0, 0, 255), # Red
                    'line_thickness': 3,
                    'fill_polygon': True,
                    'fill_color': (0, 0, 255),
                    'fill_alpha': 0.3
                }
                for points in pred_keypoints:
                    pred_image = draw_polygon_from_keypoints(pred_image, points, pred_options)
            
            combined_image[:, :image_width] = gt_image
            combined_image[:, image_width:] = pred_image
            
            cv2.putText(combined_image, "Ground Truth", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined_image, "Predictions", (image_width + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            final_image = combined_image
            
        else:
            # For test subset, we only visualize predictions
            final_image = image.copy()
            if prediction_result and 'objects' in prediction_result:
                pred_keypoints = extract_keypoints(prediction_result['objects'], is_object=True, get_pixel_coordinates=True, image_width=image_width, image_height=image_height)
                pred_options = {
                    'line_color': (0, 0, 255), 
                    'line_thickness': 3,
                    'fill_polygon': True,
                    'fill_color': (0, 0, 255),
                    'fill_alpha': 0.3
                }
                for points in pred_keypoints:
                    final_image = draw_polygon_from_keypoints(final_image, points, pred_options)
        
        image_name = Path(image_path).stem
        output_path = Path(output_dir) / f"{subset_name}_visualizations"
        output_path.mkdir(exist_ok=True)

        output_file = output_path / f"{image_name}_visualization.jpg"
        cv2.imwrite(str(output_file), final_image)
        
    except Exception as e:
        print(f"Error visualizing polygons for {image_path}: {e}")
        raise e