# 1-1-Face-Cropper

A <strong>Python script</strong> that automatically detects faces in images and crops them into perfect <strong>1x1 square profile pictures</strong>.

This tool is perfect for creating profile pictures from group photos or any image containing people. It automatically handles detection, cropping, and resizing to ensure you get a clean, square avatar every time.

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.6%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://opencv.org/">
    <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  </a>
  <a href="https://numpy.org/">
    <img src="https://img.shields.io/badge/NumPy-Array%20Processing-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/License-Educational-lightgrey?style=for-the-badge" alt="License">
  </a>
</p>

## Features

* **Automatic Face Detection:** Uses OpenCV's Haar Cascade classifier to accurately locate faces.
* **Anime/Cartoon Support:** Specialized detection mode for anime, cartoon, and stylized characters.
* **Multiple Face Support:** Automatically generates separate cropped images for every person detected in a single photo.
* **1x1 Square Output:** Crops faces to a perfect square format, ideal for social media avatars.
* **Customizable Padding:** Adjustable padding logic ensures faces aren't too zoomed in.
* **Batch Processing:** Process single images or entire directories in one go.
* **Wide Format Support:** Works with JPG, PNG, BMP, TIFF, and other common formats.

## Installation

1.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Or install manually:**
    ```bash
    pip install opencv-python numpy
    ```

## Usage

### Command Line Interface

**Process a single image:**
```bash
python crop_face.py path/to/your/image.jpg
```

**Process a directory of images:**
```bash
python crop_face.py path/to/your/photos/folder
```

**Custom output directory:**
```bash
python crop_face.py image.jpg -o my_cropped_faces
```

**Custom output size and padding:**
```bash
python crop_face.py image.jpg -s 1024 -p 0.5
```

**Process anime/cartoon faces:**
```bash
python crop_face.py anime_image.jpg --anime
```

### Command Line Options

* `input`: Input image file or directory (**required**).
* `-o, --output`: Output directory (default: `cropped_faces`).
* `-s, --size`: Output image size in pixels (default: `512`).
* `-p, --padding`: Padding ratio around face (default: `0.3` = 30% extra).
* `--cascade`: Path to custom face cascade XML file (optional).
* `--anime`: Enable anime/cartoon face detection mode.

### Programmatic Usage

You can also import the class into your own Python scripts:

```python
from crop_face import FaceCropper

# Initialize the cropper for human faces
cropper = FaceCropper()

# Initialize the cropper for anime/cartoon faces
anime_cropper = FaceCropper(anime_mode=True)

# Process a single image
faces_detected = cropper.process_image(
    image_path="photo.jpg",
    output_dir="output",
    output_size=512,
    padding_ratio=0.3
)

# Process anime/cartoon faces
anime_faces = anime_cropper.process_image(
    image_path="anime_character.jpg",
    output_dir="anime_output",
    output_size=512,
    padding_ratio=0.3
)

# Process a directory
images_processed, total_faces = cropper.process_directory(
    input_dir="photos",
    output_dir="cropped_faces",
    output_size=512,
    padding_ratio=0.3
)
```

## Examples

### Example 1: Single Person Photo
* **Input:** `family_photo.jpg` (contains 1 person)
* **Output:** `family_photo_profile.jpg` (512x512 square crop)

### Example 2: Group Photo
* **Input:** `team_photo.jpg` (contains 5 people)
* **Output:**
    * `team_photo_face_1_profile.jpg`
    * `team_photo_face_2_profile.jpg`
    * `team_photo_face_3_profile.jpg`
    * `team_photo_face_4_profile.jpg`
    * `team_photo_face_5_profile.jpg`

### Example 3: Batch Processing
```bash
# Process all images in a folder
python crop_face.py photos/ -o profile_pictures/ -s 1024
```

### Example 4: Anime/Cartoon Processing
```bash
# Process anime characters
python crop_face.py anime_screenshots/ --anime -o anime_profiles/
```

## How It Works

1.  **Face Detection:** Uses OpenCV's Haar Cascade classifier to detect faces in the image.
    * **Standard Mode:** Optimized for real human faces.
    * **Anime Mode:** Uses more sensitive parameters and alternative color-based detection for stylized characters.
2.  **Face Cropping:** For each detected face:
    * Calculates the center point of the face.
    * Creates a square crop around the face with configurable padding.
    * Handles edge cases where the crop would extend beyond image boundaries.
3.  **Resizing:** Resizes the cropped region to the specified output size.
4.  **Saving:** Saves each face as a separate 1x1 profile picture.

## Tips for Best Results

### For Human Faces
* **Good Lighting:** Well-lit faces are detected more accurately.
* **Frontal Faces:** The detector works best with faces looking toward the camera.
* **Clear Images:** Higher resolution images generally give better results.

### For Anime/Cartoon Faces
* **Use Anime Mode:** Always use the `--anime` flag for anime, cartoon, or stylized characters.
* **Good Contrast:** Characters with clear facial features work better.
* **Avoid Complex Backgrounds:** Simple backgrounds improve detection accuracy.
* **Default Padding:** Anime mode automatically uses at least 50% padding to avoid over-zooming.

### General Tips
* **Adjust Padding:** Use the `-p` parameter to add more or less space around faces.
* **Output Size:** Larger output sizes (1024+) are better for high-quality profile pictures.

## Troubleshooting

* **No faces detected?**
    * Ensure the image contains clear, well-lit faces.
    * Try adjusting the face detection parameters in the code.
    * Check that OpenCV is properly installed.
* **Poor quality crops?**
    * Increase the output size with `-s` parameter.
    * Adjust padding with `-p` parameter.
    * Use higher resolution source images.
* **Multiple faces not detected?**
    * The default settings work well for most cases.
    * For challenging images, you might need to adjust the `scaleFactor` and `minNeighbors` parameters in the `detect_faces` method.

## Dependencies

* **OpenCV:** For image processing and face detection.
* **NumPy:** For numerical operations.
* **Python 3.6+:** Required for pathlib and f-strings.

## License

This script is provided as-is for educational and personal use.
