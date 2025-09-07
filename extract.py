import argparse
import os
import zipfile
import subprocess
import shutil

def extract_frames_from_videos(zip_folder, output_folder):
    # Iterate through all zip files
    for file_name in os.listdir(zip_folder):
        if file_name.endswith(".zip"):
            zip_path = os.path.join(zip_folder, file_name)
            
            # Create folder for this zip file
            base_name = os.path.splitext(file_name)[0]
            zip_extract_folder = os.path.join(output_folder, base_name)
            os.makedirs(zip_extract_folder, exist_ok=True)
            
            # Move zip file into its folder
            shutil.move(zip_path, os.path.join(zip_extract_folder, file_name))
            
            # Extract zip contents
            with zipfile.ZipFile(os.path.join(zip_extract_folder, file_name), 'r') as zip_ref:
                zip_ref.extractall(zip_extract_folder)
            
            # Path to video file
            video_path = os.path.join(zip_extract_folder, "video_left.avi")
            if os.path.exists(video_path):
                frames_folder = os.path.join(zip_extract_folder, "frames")
                os.makedirs(frames_folder, exist_ok=True)
                
                # Use ffmpeg to extract frames as JPEGs
                cmd = [
                    "ffmpeg",
                    "-i", video_path,
                    "-q:v", "2",  # quality (2 is high quality)
                    os.path.join(frames_folder, "%09d.jpg")
                ]
                subprocess.run(cmd, check=True)
                print(f"Frames extracted to {frames_folder}")


def accumulate(root_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    os.makedirs(os.path.join(dest_folder, "segmentation"), exist_ok=True)
    os.makedirs(os.path.join(dest_folder, "frames"), exist_ok=True)

    # Walk through all subfolders
    for cDir in os.listdir(root_folder):
        subdir = os.path.join(root_folder, cDir, "segmentation")
        if os.path.isdir(subdir):
            images_folder = os.path.join(root_folder, cDir, "frames")
            if not os.path.exists(images_folder):
                continue  # skip if no matching images folder

            for fname in os.listdir(subdir):
                seg_path = os.path.join(subdir, fname)

                if os.path.isfile(seg_path):
                    # Parse frame number from fname
                    try:
                        frame_num = int(os.path.splitext(fname)[0])  # e.g. "000000123.png" -> 123
                        next_frame_name = f"{frame_num+1:09d}.jpg"   # +1, keep 9-digit format
                        img_path = os.path.join(images_folder, next_frame_name)
                    except ValueError:
                        print(f"Skipping invalid filename: {fname}")
                        continue

                    # Copy segmentation image
                    shutil.copy(seg_path, os.path.join(dest_folder, 'segmentation', f'{cDir}_{fname}'))

                    # Copy next frame as current fname
                    if os.path.exists(img_path):
                        shutil.copy(img_path, os.path.join(dest_folder, 'frames', f'{cDir}_{fname}'))
                    else:
                        print(f"Warning: Next frame not found for {seg_path} -> {next_frame_name}")

    print("Copy complete!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./mmr")
    args = parser.parse_args()
    # Folder containing your zip files
    zip_folder = [f"{args.data_dir}/test/", f"{args.data_dir}/train/"]

    # Output folder for unzipped content
    output_folder = [f"{args.data_dir}/test/", f"{args.data_dir}/train/"]
    for zf, of in zip(zip_folder, output_folder):
        extract_frames_from_videos(zf, of)
        accumulate(zf, of)

if __name__ == "__main__":
    main()
