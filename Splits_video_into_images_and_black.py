import tifffile as tiff
import cv2
import os

def convert_lsm_to_frames(lsm_path, output_folder):
    """
    Convert an LSM file to a sequence of images.

    Args:
        lsm_path (str): Path to the input LSM file.
        output_folder (str): Path to the folder where frames will be saved.
    """
    # Verify the LSM file exists
    if not os.path.exists(lsm_path):
        raise FileNotFoundError(f"LSM file not found at {lsm_path}")

    # Load the LSM file
    print(f"Loading LSM file: {lsm_path}")
    try:
        lsm_data = tiff.imread(lsm_path) # Load the LSM file
    except Exception as e:
        raise ValueError(f"Error loading LSM file: {e}")

    print(f"LSM file loaded successfully! Shape: {lsm_data.shape}")
    # Handle multiple channels if needed (e.g., shape: (frames, channels, height, width))
    if len(lsm_data.shape) == 4:
        print("Multiple channels detected. Averaging channels...")
        lsm_data = lsm_data.mean(axis=1) # Combine channels by averaging

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        print(f"Output folder exists: {output_folder}. Frames will overwrite existing files.")

    # Save each frame as an image
    for i, frame in enumerate(lsm_data):
        # Normalize the frame values to 0-255 for saving as an image
        normalized_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        # Define the filename for the current frame
        frame_filename = os.path.join(output_folder, f"frame_{i:04d}.png")

        # Save the frame
        cv2.imwrite(frame_filename, normalized_frame)
        print(f"Saved frame {i} to {frame_filename}")

    print(f"All frames have been saved in: {output_folder}")

# Example usage
if __name__ == "__main__":
    # Define paths
    lsm_path = r"C:\videos_lsm_test_yolov_8\kk fly5 sr1.lsm"
    output_folder =r"C:\videos_lsm_test_yolov_8\frames_lsm\kk fly5 sr1"

    # Run the conversion
    try:
        convert_lsm_to_frames(lsm_path, output_folder)
    except Exception as e:
        print(f"An error occurred: {e}")
