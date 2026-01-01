import cv2
import os
import numpy as np
from pathlib import Path
import argparse

class FaceCropper:
    def __init__(self, face_cascade_path=None, anime_mode=False):
        self.anime_mode = anime_mode
        
        if face_cascade_path and os.path.exists(face_cascade_path):
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        else:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if self.face_cascade.empty():
            raise ValueError("Failed to load face cascade classifier")
    
    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self.anime_mode:
            all_faces = []
            
            faces_haar = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,  # Smaller scale factor for more precise detection
                minNeighbors=3,    # Lower threshold to catch more faces
                minSize=(40, 40),  # Larger minimum size to avoid false positives
                maxSize=(0, 0)     # No maximum size limit
            )
            all_faces.extend(faces_haar)
            
            faces_enhanced = self.detect_anime_faces_enhanced(image)
            all_faces.extend(faces_enhanced)
            
            if len(all_faces) < 2:  # If we found less than 2 faces, try color method
                faces_color = self.detect_anime_faces_alternative(image)
                all_faces.extend(faces_color)
                if len(faces_color) > 0:
                    print("Using color-based anime face detection")
            
            if all_faces:
                faces = np.array(all_faces)
                
                faces = self.remove_duplicate_detections(faces)
                
                filtered_faces = []
                img_area = image.shape[0] * image.shape[1]
                
                for face in faces:
                    x, y, w, h = face
                    face_area = w * h
                    aspect_ratio = w / h
                    
                    if (face_area >= (img_area * 0.015) and  # At least 1.5% of image
                        face_area <= (img_area * 0.3) and     # Not more than 30% of image
                        0.6 <= aspect_ratio <= 1.4 and       # Reasonable aspect ratio
                        w >= 50 and h >= 50):                # Minimum size
                        
                        filtered_faces.append(face)
                
                faces = np.array(filtered_faces)
            else:
                faces = np.array([])
        else:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
        
        return faces
    
    def detect_anime_faces_alternative(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        skin_ranges = [
            (np.array([0, 10, 60], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8)),
            (np.array([0, 20, 70], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8)),
            (np.array([0, 5, 80], dtype=np.uint8), np.array([20, 100, 255], dtype=np.uint8))
        ]
        
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for lower_skin, upper_skin in skin_ranges:
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            combined_mask = cv2.bitwise_or(combined_mask, skin_mask)
        
        kernel = np.ones((3, 3), np.uint8)  # Smaller kernel for more precise detection
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        potential_faces = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            aspect_ratio = w / h
            area = w * h
            img_area = image.shape[0] * image.shape[1]
            
            if (0.6 <= aspect_ratio <= 1.4 and 
                50 <= area <= (img_area * 0.6) and  # Reduced max area
                w >= 40 and h >= 40 and  # Increased minimum size
                area >= (img_area * 0.01)):  # Must be at least 1% of image
                
                potential_faces.append((x, y, w, h))
        
        return np.array(potential_faces) if potential_faces else np.array([])
    
    def detect_anime_faces_enhanced(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        potential_faces = []
        img_area = image.shape[0] * image.shape[1]
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            aspect_ratio = w / h
            area = w * h
            
            if (0.7 <= aspect_ratio <= 1.3 and 
                100 <= area <= (img_area * 0.4) and  # Smaller max area to avoid hands/arms
                w >= 50 and h >= 50 and  # Larger minimum size
                area >= (img_area * 0.02)):  # Must be at least 2% of image
                
                roi = gray[y:y+h, x:x+w]
                if roi.size > 0:
                    std_dev = np.std(roi)
                    if std_dev > 20:  # Good contrast threshold
                        potential_faces.append((x, y, w, h))
        
        return np.array(potential_faces) if potential_faces else np.array([])
    
    def remove_duplicate_detections(self, faces, overlap_threshold=0.5):
        if len(faces) <= 1:
            return faces
        
        areas = [w * h for x, y, w, h in faces]
        sorted_indices = np.argsort(areas)[::-1]
        
        unique_faces = []
        used_indices = set()
        
        for i in sorted_indices:
            if i in used_indices:
                continue
                
            current_face = faces[i]
            unique_faces.append(current_face)
            used_indices.add(i)
            
            for j in range(len(faces)):
                if j in used_indices:
                    continue
                    
                other_face = faces[j]
                
                x1 = max(current_face[0], other_face[0])
                y1 = max(current_face[1], other_face[1])
                x2 = min(current_face[0] + current_face[2], other_face[0] + other_face[2])
                y2 = min(current_face[1] + current_face[3], other_face[1] + other_face[3])
                
                if x2 > x1 and y2 > y1:
                    intersection_area = (x2 - x1) * (y2 - y1)
                    current_area = current_face[2] * current_face[3]
                    other_area = other_face[2] * other_face[3]
                    
                    overlap_ratio = intersection_area / min(current_area, other_area)
                    
                    if overlap_ratio > overlap_threshold:
                        used_indices.add(j)
        
        return np.array(unique_faces)
    
    def crop_face_to_square(self, image, face_rect, output_size=512, padding_ratio=0.3):
        x, y, w, h = face_rect
        img_height, img_width = image.shape[:2]
        
        center_x = x + w // 2
        face_center_y = y + h // 2
        
        space_above_face = int(h * 0.5)   # Space above head (reduced for more zoom)
        space_below_face = int(h * 1.0)   # Space below face for shoulders/upper chest (reduced to stop above breasts)
        
        crop_top = max(0, y - space_above_face)
        
        face_bottom = y + h
        desired_crop_bottom = min(img_height, face_bottom + space_below_face)
        
        desired_crop_height = desired_crop_bottom - crop_top
        
        desired_crop_width = int(w * 1.4)
        
        crop_size = max(desired_crop_height, desired_crop_width)
        
        max_crop_from_image = min(int(img_width * 0.6), int(img_height * 0.6))
        crop_size = min(crop_size, max_crop_from_image)
        
        crop_x1 = max(0, center_x - crop_size // 2)
        crop_x2 = crop_x1 + crop_size
        
        if crop_x2 > img_width:
            crop_x2 = img_width
            crop_x1 = crop_x2 - crop_size
            if crop_x1 < 0:
                crop_x1 = 0
                crop_x2 = min(crop_size, img_width)
        
        crop_y1 = crop_top
        
        crop_y2 = min(desired_crop_bottom, crop_y1 + crop_size, img_height)
        
        if crop_y2 - crop_y1 < crop_size:
            crop_y1 = max(0, crop_y2 - crop_size)
        
        actual_width = crop_x2 - crop_x1
        actual_height = crop_y2 - crop_y1
        square_size = min(actual_width, actual_height)
        
        if square_size < crop_size:
            crop_x1 = max(0, center_x - square_size // 2)
            crop_x2 = min(img_width, crop_x1 + square_size)
            if crop_x2 - crop_x1 < square_size:
                crop_x2 = crop_x1 + square_size
            
            max_bottom = min(img_height, face_bottom + space_below_face)
            crop_y2 = min(max_bottom, crop_y1 + square_size, img_height)
            crop_y1 = max(0, crop_y2 - square_size)
        
        crop_x1 = max(0, min(crop_x1, img_width - 1))
        crop_y1 = max(0, min(crop_y1, img_height - 1))
        crop_x2 = max(crop_x1 + 1, min(crop_x2, img_width))
        crop_y2 = max(crop_y1 + 1, min(crop_y2, img_height))
        
        final_width = crop_x2 - crop_x1
        final_height = crop_y2 - crop_y1
        if final_width != final_height:
            square_final = min(final_width, final_height)
            center_x_final = (crop_x1 + crop_x2) // 2
            crop_x1 = center_x_final - square_final // 2
            crop_x2 = crop_x1 + square_final
            crop_y2 = min(img_height, crop_y1 + square_final)
            crop_y1 = crop_y2 - square_final
            crop_x1 = max(0, min(crop_x1, img_width))
            crop_y1 = max(0, min(crop_y1, img_height))
            crop_x2 = min(img_width, max(crop_x1 + 1, crop_x2))
            crop_y2 = min(img_height, max(crop_y1 + 1, crop_y2))
        
        crop_ratio_width = (crop_x2 - crop_x1) / img_width
        crop_ratio_height = (crop_y2 - crop_y1) / img_height
        
        if crop_ratio_width > 0.95 or crop_ratio_height > 0.95:
            target_ratio = 0.6
            new_width = int(img_width * target_ratio)
            new_height = int(img_height * target_ratio)
            square_size = min(new_width, new_height)
            
            crop_x1 = max(0, center_x - square_size // 2)
            crop_x2 = min(img_width, crop_x1 + square_size)
            crop_y1 = max(0, crop_top)
            crop_y2 = min(img_height, crop_y1 + square_size, face_bottom + space_below_face)
        
        face_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        face_crop_resized = cv2.resize(face_crop, (output_size, output_size))
        
        return face_crop_resized
    
    def process_image(self, image_path, output_dir, output_size=512, padding_ratio=0.3):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return 0
        
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            print(f"No faces detected in {image_path}")
            return 0
        
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = Path(image_path).stem
        
        for i, face_rect in enumerate(faces):
            face_crop = self.crop_face_to_square(
                image, face_rect, output_size, padding_ratio
            )
            
            if len(faces) == 1:
                output_filename = f"{base_name}_profile.jpg"
            else:
                output_filename = f"{base_name}_face_{i+1}_profile.jpg"
            
            output_path = os.path.join(output_dir, output_filename)
            
            cv2.imwrite(output_path, face_crop)
            print(f"Saved: {output_path}")
        
        print(f"Processed {len(faces)} face(s) from {image_path}")
        return len(faces)
    
    def process_directory(self, input_dir, output_dir, output_size=512, padding_ratio=0.3
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return 0, 0
        
        total_images = 0
        total_faces = 0
        
        print(f"Found {len(image_files)} image(s) to process...")
        
        for image_file in image_files:
            try:
                faces_detected = self.process_image(
                    str(image_file), output_dir, output_size, padding_ratio
                )
                total_images += 1
                total_faces += faces_detected
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        
        print(f"\nProcessing complete!")
        print(f"Images processed: {total_images}")
        print(f"Total faces detected: {total_faces}")
        
        return total_images, total_faces

def main():
    parser = argparse.ArgumentParser(description='Auto-crop faces from images to 1x1 profile pictures with ID photo layout (shows head, shoulders, and upper chest)')
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('-o', '--output', default='cropped_faces', 
                       help='Output directory (default: cropped_faces)')
    parser.add_argument('-s', '--size', type=int, default=512,
                       help='Output image size (default: 512)')
    parser.add_argument('-p', '--padding', type=float, default=0.3,
                       help='Padding ratio (legacy parameter, now uses ID photo layout by default)')
    parser.add_argument('--cascade', help='Path to custom face cascade XML file')
    parser.add_argument('--anime', action='store_true',
                       help='Enable anime/cartoon face detection mode')
    
    args = parser.parse_args()
    
    try:
        cropper = FaceCropper(args.cascade, args.anime)
        
        if os.path.isfile(args.input):
            cropper.process_image(args.input, args.output, args.size, args.padding)
        elif os.path.isdir(args.input):
            cropper.process_directory(args.input, args.output, args.size, args.padding)
        else:
            print(f"Error: {args.input} is not a valid file or directory")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
