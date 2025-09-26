import os
import cv2
import numpy as np
import subprocess
import shutil
from glob import glob

# --- 1. Configuration ---
# Directories
INPUT_DIR = 'test_images/'
PATCHES_DIR = 'generated_patches/'
CRAFT_RESULTS_DIR = './result/' # The folder where test.py will save its results
FINAL_OUTPUT_DIR = 'final_outputs/'

# Patching Parameters
PATCH_SIZE = 768  # Corresponds to --canvas_size in test.py
OVERLAP = 96      # Overlap percentage

# CRAFT Parameters (must match your test.py arguments)
TRAINED_MODEL = 'craft_mlt_25k.pth' # Correct path since it's in the root folder
TEXT_THRESHOLD = 0.7
LINK_THRESHOLD = 0.4
LOW_TEXT = 0.4
CUDA = True # Set to False if you don't have a GPU

# --- 2. Helper Functions ---

def create_patches(image_path, output_dir, patch_size, overlap):
    """Slices a large image into smaller, overlapping patches."""
    print(f"   - Slicing {os.path.basename(image_path)}...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"   - Could not read image: {image_path}")
        return

    img_h, img_w, _ = image.shape
    stride = patch_size - overlap
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    pad_h = max(0, patch_size - img_h)
    pad_w = max(0, patch_size - img_w)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[255,255,255])
        img_h, img_w, _ = image.shape

    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            y_start, x_start = y, x
            y_end, x_end = min(y + patch_size, img_h), min(x + patch_size, img_w)

            patch = image[y_start:y_end, x_start:x_end]
            
            ph, pw, _ = patch.shape
            if ph < patch_size or pw < patch_size:
                pad_ph = patch_size - ph
                pad_pw = patch_size - pw
                patch = cv2.copyMakeBorder(patch, 0, pad_ph, 0, pad_pw, cv2.BORDER_CONSTANT, value=[255,255,255])

            patch_filename = f"{base_filename}_patch_{x_start}_{y_start}.png"
            cv2.imwrite(os.path.join(output_dir, patch_filename), patch)

def parse_box_file(file_path):
    """Reads a CRAFT result file and returns a list of bounding box polygons."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        coords = line.strip().split(',')
        if len(coords) == 8:
            box = np.array([int(c) for c in coords]).reshape(4, 2)
            boxes.append(box)
    return boxes

def non_maximum_suppression(boxes, iou_threshold=0.3):
    """Simplified Non-Maximum Suppression for polygons based on their bounding rectangles."""
    if not boxes:
        return []

    rects = np.array([cv2.boundingRect(box) for box in boxes])
    x1, y1, w, h = rects[:, 0], rects[:, 1], rects[:, 2], rects[:, 3]
    x2, y2 = x1 + w, y1 + h
    areas = w * h
    
    indices = np.argsort(y2)
    
    final_boxes_indices = []
    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[last]
        final_boxes_indices.append(i)
        
        xx1 = np.maximum(x1[i], x1[indices[:last]])
        yy1 = np.maximum(y1[i], y1[indices[:last]])
        xx2 = np.minimum(x2[i], x2[indices[:last]])
        yy2 = np.minimum(y2[i], y2[indices[:last]])
        
        w_intersect = np.maximum(0, xx2 - xx1)
        h_intersect = np.maximum(0, yy2 - yy1)
        
        intersection = w_intersect * h_intersect
        union = areas[i] + areas[indices[:last]] - intersection
        # Avoid division by zero
        iou = intersection / np.maximum(union, 1e-10)
        
        indices = np.delete(indices, np.concatenate(([last], np.where(iou > iou_threshold)[0])))
        
    return [boxes[i] for i in final_boxes_indices]

def save_final_coordinates(boxes, output_path):
    """Saves the final list of bounding box coordinates to a text file."""
    with open(output_path, 'w') as f:
        for box in boxes:
            coords = box.flatten()
            line = ','.join([str(c) for c in coords])
            f.write(line + '\n')

# --- 3. Main Orchestration Logic ---
if __name__ == '__main__':
    # --- STEP 1: SETUP DIRECTORIES ---
    print("--- STEP 1: Setting up directories ---")
    if os.path.exists(PATCHES_DIR): shutil.rmtree(PATCHES_DIR)
    if os.path.exists(CRAFT_RESULTS_DIR): shutil.rmtree(CRAFT_RESULTS_DIR)
    if os.path.exists(FINAL_OUTPUT_DIR): shutil.rmtree(FINAL_OUTPUT_DIR)
    
    os.makedirs(PATCHES_DIR, exist_ok=True)
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
    
    # --- STEP 2: CREATE PATCHES ---
    print("\n--- STEP 2: Slicing P&IDs into patches ---")
    pid_images = glob(os.path.join(INPUT_DIR, '*.png')) + glob(os.path.join(INPUT_DIR, '*.jpg'))
    for image_path in pid_images:
        create_patches(image_path, PATCHES_DIR, PATCH_SIZE, OVERLAP)

    print("Patch creation complete.")

    # --- STEP 3: RUN CRAFT DETECTION ON PATCHES ---
    print("\n--- STEP 3: Running CRAFT detection on all patches ---")
    # <<< THIS COMMAND IS NOW CORRECT because test.py will understand --result_folder
    command = [
        "python", "test.py",
        "--trained_model", TRAINED_MODEL,
        "--test_folder", PATCHES_DIR,
        "--result_folder", CRAFT_RESULTS_DIR,
        "--text_threshold", str(TEXT_THRESHOLD),
        "--link_threshold", str(LINK_THRESHOLD),
        "--low_text", str(LOW_TEXT),
        "--canvas_size", str(PATCH_SIZE),
        "--cuda", "True" if CUDA else "False"
    ]
    subprocess.run(command)
    print("CRAFT detection complete.")

    # --- STEP 4: STITCH RESULTS AND MERGE BOXES ---
    print("\n--- STEP 4: Stitching results back onto original P&IDs ---")
    for original_image_path in pid_images:
        base_filename = os.path.splitext(os.path.basename(original_image_path))[0]
        print(f"   - Processing results for {base_filename}...")
        
        all_global_boxes = []
        
        result_files = glob(os.path.join(CRAFT_RESULTS_DIR, f"res_{base_filename}_patch_*.txt"))
        
        for res_file in result_files:
            try:
                filename_parts = os.path.basename(res_file).replace('.txt', '').split('_')
                x_offset = int(filename_parts[-2])
                y_offset = int(filename_parts[-1])
                
                local_boxes = parse_box_file(res_file)
                
                for box in local_boxes:
                    global_box = box + np.array([x_offset, y_offset])
                    all_global_boxes.append(global_box)
            except (ValueError, IndexError):
                print(f"     - Warning: Could not parse coordinates from filename: {os.path.basename(res_file)}")

        print(f"   - Found {len(all_global_boxes)} boxes before merging.")
        final_boxes = non_maximum_suppression(all_global_boxes, iou_threshold=0.3)
        print(f"   - {len(final_boxes)} boxes remaining after NMS.")
        
        # --- STEP 5: SAVE FINAL OUTPUTS (IMAGE AND COORDINATES) ---
        original_image = cv2.imread(original_image_path)
        if final_boxes:
             cv2.polylines(original_image, [box.astype(np.int32) for box in final_boxes], isClosed=True, color=(0, 0, 255), thickness=2)
        
        image_output_path = os.path.join(FINAL_OUTPUT_DIR, os.path.basename(original_image_path))
        cv2.imwrite(image_output_path, original_image)
        
        coords_output_path = os.path.join(FINAL_OUTPUT_DIR, f"res_{base_filename}.txt")
        save_final_coordinates(final_boxes, coords_output_path)
        
    print("\nProcessing complete! Final images and coordinates saved in:", FINAL_OUTPUT_DIR)