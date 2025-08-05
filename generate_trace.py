import cv2
import os
import numpy as np
import random

# Function to get a colored mask for each contour
def get_coloured_mask(mask):
    colours = [
        [0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255],
        [255, 128, 0], [128, 255, 0], [0, 255, 128], [0, 128, 255], [128, 0, 255], [255, 0, 128],
        [80, 70, 180], [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190], [0, 128, 0], [255, 165, 0]
    ]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    idx = random.randrange(0, len(colours))
    r[mask == 255], g[mask == 255], b[mask == 255] = colours[idx]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def get_trace(image_path, track_path, trace_path):
    track_picture = sorted([file for file in os.listdir(track_path) if ".tif" in file])
    test_image = sorted([file for file in os.listdir(image_path) if ".tif" in file])
    trace_image = []

    for i in range(len(test_image)):
        # Read and process the test image
        image_to_draw = cv2.imread(os.path.join(image_path, test_image[i]), -1)
        image_to_draw = np.stack((image_to_draw,) * 3, axis=2)

        # Read the corresponding tracking image
        if i < len(track_picture):
            result_picture = cv2.imread(os.path.join(track_path, track_picture[i]), -1)
        else:
            # Fixed Unicode issue - use ASCII characters only
            print(f"WARNING: Track image {i} not found, stopping trace generation")
            break
        label_picture = ((result_picture >= 1) * 255).astype(np.uint8)

        # Find contours for colored masks and ID labels
        contours, _ = cv2.findContours(label_picture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Apply colored masks and ID numbers to each contour
        font = cv2.FONT_HERSHEY_SIMPLEX
        for contour in contours:
            # Apply colored mask
            mask = np.zeros_like(image_to_draw[:, :, 0])
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            colored_mask = get_coloured_mask(mask)
            image_to_draw = cv2.addWeighted(image_to_draw, 1, colored_mask, 0.5, 0)
            
            # Add cell ID number at centroid (no division info)
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cell_id = result_picture[cy, cx]
                
                if cell_id != 0:
                    # Display only the cell ID number (no parent info)
                    cv2.putText(image_to_draw, str(cell_id), (cx, cy), font, 0.5, (255, 255, 255), 1)

        # Add the frame number to the top left corner (white)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_to_draw, f"{i}", (10, 30), font, 1, (255, 255, 255), 3)
        
        trace_image.append(image_to_draw)

    # Save traced images (no trajectory lines)
    for i in range(len(trace_image)):
        cv2.imwrite(os.path.join(trace_path, test_image[i]), trace_image[i])

# Function to create video from traced images
def get_video(trace_path):
    directory = trace_path
    pictures = sorted([name for name in os.listdir(directory) if "trace" not in name])
    print(pictures)
    if not pictures:
        print("No images found to generate video.")
        return

    fps = 1  # Frames per second
    image = cv2.imread(os.path.join(directory, pictures[0]), -1)
    size = (image.shape[1], image.shape[0])

    videowriter = cv2.VideoWriter(os.path.join(trace_path, "trace.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, name in enumerate(pictures):
        img = cv2.imread(os.path.join(directory, name), -1)
        cv2.putText(img, str(i), (10, 30), font, 1, (255, 255, 255), 1)
        videowriter.write(img)

    videowriter.release()
    print("Video generation completed.")

# Helper function to create a folder if it doesn't exist
def createFolder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(f"{path} has been created.")
    else:
        print(f"{path} already exists.")

# Main execution
if __name__ == "__main__":
    # Example directories (modify these paths as needed)
    print("Generating trace")
    test_folders = os.listdir("nuclear_dataset")
    test_folders = [os.path.join("nuclear_dataset/", folder) for folder in test_folders]
    test_folders.sort()

    for folder in test_folders:
        test_path = os.path.join(folder, "test")
        track_result_path = os.path.join(folder, "track_result")
        trace_path = os.path.join(folder, "trace1")
        createFolder(trace_path)

        # Ensure trace images are generated before creating the video
        get_trace(test_path, track_result_path, trace_path)
        get_video(trace_path)

    print("Generating trace")

    print("Processing completed.")
