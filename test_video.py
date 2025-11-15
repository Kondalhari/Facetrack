import cv2
import os
from ultralytics import YOLO
import face_embedder

def test_video_processing(video_path=0, detector=None, embedder=None, show_window=True, max_frames=None):  # 0 = webcam
    """
    Test video processing pipeline without database integration.
    """
    print("Testing video processing pipeline...")
    
    # Initialize models if not provided
    created_local_models = False
    if detector is None or embedder is None:
        print("Loading models...")
        detector = YOLO('yolov8n-face.pt')  # using default YOLO face model
        embedder = face_embedder.FaceEmbedder(use_gpu=False)
        created_local_models = True
    
    # Open video source
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video source: {video_path}")
        return
    
    print(f"Processing video from: {video_path}")
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 3 != 0:  # process every 3rd frame
            continue
            
        # Run detection
        results = detector.track(frame, 
                               persist=True,
                               tracker="bytetrack.yaml",
                               classes=0,
                               verbose=False)
        
        if results and hasattr(results, 'boxes'):
            boxes = results.boxes
            
            # Show detection count
            detect_count = len(boxes.data) if boxes.data is not None else 0
            cv2.putText(frame, f"Detected: {detect_count}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Get face crops and compute embeddings
            if boxes.data is not None:
                for box in boxes.data:
                    x1, y1, x2, y2 = map(int, box[:4])
                    crop = frame[y1:y2, x1:x2]
                    
                    # Get embedding (just to test the pipeline)
                    embedding = embedder.get_embedding(crop)
                    if embedding is not None:
                        # Draw green box for successful embedding
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else:
                        # Draw red box for failed embedding
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        if show_window:
            cv2.imshow("Test Face Processing", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if max_frames is not None and frame_count >= max_frames:
            print(f"Reached max_frames={max_frames} for {video_path}")
            break
    
    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    # If we created models inside this function, attempt to clean up (no-op for most libs)
    if created_local_models:
        try:
            # ultralytics and insightface don't require explicit teardown, but we dereference
            del detector
            del embedder
        except Exception:
            pass

def find_first_video_in_folder(folder='input_videos'):
    """Return the path to the first video file found in `folder` or None if none found."""
    if not os.path.exists(folder):
        return None
    exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
    for fname in sorted(os.listdir(folder)):
        if os.path.splitext(fname)[1].lower() in exts:
            return os.path.join(folder, fname)
    return None


def get_all_videos_in_folder(folder='input_videos'):
    """Return a sorted list of video paths found in folder."""
    if not os.path.exists(folder):
        return []
    exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if os.path.splitext(f)[1].lower() in exts]
    return files


def process_all_videos(folder='input_videos', show_window=True, max_frames_per_video=None):
    """Load models once and process every video in the folder sequentially.

    Args:
        folder: folder to look for videos
        show_window: whether to display frames
        max_frames_per_video: if set, limit frames processed per video (useful for smoke tests)
    """
    videos = get_all_videos_in_folder(folder)
    if not videos:
        print(f"No videos found in {folder}")
        return

    print(f"Processing {len(videos)} videos from {folder}")
    detector = YOLO('yolov8n-face.pt')
    embedder = face_embedder.FaceEmbedder(use_gpu=False)

    for vid in videos:
        print(f"--- Processing {vid} ---")
        test_video_processing(vid, detector=detector, embedder=embedder, show_window=show_window, max_frames=max_frames_per_video)




if __name__ == "__main__":
    # Default behavior: process all videos in `input_videos/` sequentially.
    # For a quick smoke-test, set max_frames_per_video to a small number (e.g., 30).
    process_all_videos('input_videos', show_window=True, max_frames_per_video=None)