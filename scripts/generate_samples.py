import os
import sys
import csv
import argparse
import datetime
import cv2

# Add project root to path to import face_embedder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ultralytics import YOLO
import face_embedder


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_crop(out_dir, visitor_tag, vid_name, frame_idx, crop_img):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    fname = f"{visitor_tag}_{vid_name}_f{frame_idx}_{ts}.jpg"
    path = os.path.join(out_dir, fname)
    cv2.imwrite(path, crop_img)
    return path


def process_videos(folder='input_videos', out_root='logs/sample', skip=3, max_frames=None, headless=True):
    videos = []
    if not os.path.exists(folder):
        print(f"No folder: {folder}")
        return
    exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
    for f in sorted(os.listdir(folder)):
        if os.path.splitext(f)[1].lower() in exts:
            videos.append(os.path.join(folder, f))

    if not videos:
        print("No videos found to process.")
        return

    detector = YOLO('yolov8n.pt')
    embedder = face_embedder.FaceEmbedder(use_gpu=False)

    ensure_dir(out_root)
    csv_path = os.path.join(out_root, 'events.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['video', 'frame', 'box', 'crop_path', 'has_embedding', 'embedding_len'])

        for vid in videos:
            vid_name = os.path.splitext(os.path.basename(vid))[0]
            print(f"Processing video: {vid}")
            cap = cv2.VideoCapture(vid)
            if not cap.isOpened():
                print(f"Could not open {vid}")
                continue

            frame_idx = 0
            saved_dir = os.path.join(out_root, datetime.datetime.now().strftime('%Y-%m-%d'))
            ensure_dir(saved_dir)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % skip != 0:
                    continue

                if max_frames is not None and frame_idx > max_frames:
                    break

                # Run detection (fast mode)
                try:
                    results = detector.predict(frame, imgsz=640)
                except Exception as e:
                    print(f"Detection error: {e}")
                    continue

                # results could be list-like
                res0 = results[0] if isinstance(results, (list, tuple)) else results
                boxes = getattr(res0, 'boxes', None)
                if boxes is None or boxes.data is None:
                    continue

                for box in boxes.data:
                    x1, y1, x2, y2 = map(int, box[:4])
                    crop = frame[y1:y2, x1:x2]
                    if crop is None or crop.size == 0:
                        continue

                    # Save crop
                    crop_path = save_crop(saved_dir, 'face', vid_name, frame_idx, crop)

                    # Try to compute embedding
                    embedding = None
                    try:
                        embedding = embedder.get_embedding(crop)
                    except Exception as e:
                        embedding = None

                    has_embedding = embedding is not None
                    embedding_len = len(embedding) if embedding is not None else 0
                    writer.writerow([vid_name, frame_idx, f"{x1},{y1},{x2},{y2}", crop_path, has_embedding, embedding_len])

            cap.release()

    print(f"Sample generation finished. CSV: {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='input_videos')
    parser.add_argument('--out', default='logs/sample')
    parser.add_argument('--skip', type=int, default=3)
    parser.add_argument('--max-frames', type=int, default=30)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()

    process_videos(folder=args.folder, out_root=args.out, skip=args.skip, max_frames=args.max_frames, headless=args.headless)
