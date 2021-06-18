import cv2
import os
import glob
import math
import re
import argparse
import logging
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-folder',
                        type=str,
                        default='dataset/splitted_videos',
                        help='Folder path for splitted videos')
    parser.add_argument('--dest-folder',
                        type=str,
                        default='data',
                        help='Folder path for extracted frames')
    args = parser.parse_known_args()[0]
    return args

def check_filenames(save_folder):
    # Check if video has already been extracted to frames
    video_basenames  = set()
    save_folder = os.path.join(save_folder, "*")
    for filepath in glob.glob(save_folder):
        basename = os.path.basename(filepath)
        match = re.match(r'(.*)_frame\d+\.jpg', basename)
        if match:
            video_basenames.update(match.groups())
    return video_basenames

def crop_frame(frame):
    # Crop frame to square
    dimensions = frame
    y1, y2, x1, x2 = 0, dimensions.shape[0], 0, dimensions.shape[1]

    if(dimensions.shape[0] > dimensions.shape[1]):
        cropped_frame = frame[y2-x2:y2, x1:x2]
    else:
        cropped_frame = frame[y1:y2, x2//2-y2//2:x2//2+y2//2]
    return cropped_frame

def extract_frames(path, dest_path):
    # Extract video to frames by two frames per second
    # and save to destination folder
    picture_count = 0
    cap = cv2.VideoCapture(path)
    frameRate = cap.get(5)
    basefile = os.path.basename(path)

    while(cap.isOpened()):
        frameId = cap.get(1)
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate/2) == 0):
            frame = crop_frame(frame)
            filename = "%s_frame%d.jpg" % (basefile, picture_count)
            output_path = os.path.join(dest_path, filename)
            picture_count += 1
            cv2.imwrite(output_path, frame)
    cap.release()
    return picture_count

def read_videos(src_path, dest_path):
    # Extract videos that have valid extensions (.mov, .mp4)
    # and have not yet been extracted
    Path(dest_path).mkdir(parents=True, exist_ok=True)
    allowed_extensions = {".mov", ".mp4"}
    src_path = os.path.join(src_path, "*")
    existing_filenames = check_filenames(dest_path)

    for filepath in glob.glob(src_path):
        basename = os.path.basename(filepath)
        extension = os.path.splitext(filepath)[-1].lower()
        if extension not in allowed_extensions:
            logging.debug("Skip %s, not allowed extension type" % basename)
            continue
        logging.info("Processing %s" % filepath)

        if basename in existing_filenames:
            logging.debug("Video %s already extracted" % basename)
            continue

        frame_count = extract_frames(filepath, dest_path)
        logging.debug("Extracted %i frames" % frame_count)
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()

    # Read and extract training, validation and test videos
    read_videos(os.path.join(args.src_folder, 'train/safe'), os.path.join(args.dest_folder, 'train/safe'))
    read_videos(os.path.join(args.src_folder, 'train/not_safe'), os.path.join(args.dest_folder, 'train/not_safe'))
    read_videos(os.path.join(args.src_folder, 'validation/safe'), os.path.join(args.dest_folder, 'validation/safe'))
    read_videos(os.path.join(args.src_folder, 'validation/not_safe'), os.path.join(args.dest_folder, 'validation/not_safe'))
    read_videos(os.path.join(args.src_folder, 'test/safe'), os.path.join(args.dest_folder, 'test/safe'))
    read_videos(os.path.join(args.src_folder, 'test/not_safe'), os.path.join(args.dest_folder, 'test/not_safe'))