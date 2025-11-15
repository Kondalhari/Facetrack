import cv2
import json
import sys
import logging
from ultralytics import YOLO
import database
import face_embedder
import state_tracker

def main():
    # 1. Load Configuration
    try:
        with open('config.json') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("FATAL: config.json not found.")
        sys.exit(1)

    # 2. Connect to Database
    db_conn = database.get_db_connection(config)
    if db_conn is None:
        print("FATAL: Could not connect to database. Check config and run init_db.py.")
        sys.exit(1)
    
    # 3. Load AI Models
    print(f"Loading YOLO detector: {config.get('yolo_model_path')}")
    detector = YOLO(config.get('yolo_model_path'))

    # Configure embedder (allow CPU by default; set use_gpu=True in config to use GPU)
    embedder = face_embedder.FaceEmbedder(use_gpu=config.get('use_gpu', False))

    # 4. Initialize State Tracker
    tracker = state_tracker.VisitorTracker(config)

    # 5. Open Video Source
    video_source = config.get('video_source')
    if not video_source:
        print("FATAL: 'video_source' not set in config.json")
        sys.exit(1)
        
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"FATAL: Could not open video source: {video_source}")
        sys.exit(1)

    print(f"--- Processing video stream: {video_source} ---")
    
    frame_count = 0
    frame_skip = config.get('frame_skip', 1)

    # 6. Main Processing Loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video
            
        # 6.1. Frame Skipping Logic
        if frame_count % frame_skip!= 0:
            frame_count += 1
            continue
        
        frame_count += 1

        # 6.2. Run Detection & Tracking (YOLO + ByteTrack)
        # We specify `classes=0` assuming 'face' is class 0 in this model.
        # `persist=True` tells the tracker to remember tracks between frames.
        # `tracker="bytetrack.yaml"` explicitly selects ByteTrack.
        try:
            results = detector.track(frame, 
                                     persist=True, 
                                     tracker="bytetrack.yaml", 
                                     classes=0,
                                     verbose=False) # Set to True for more debug info
        except Exception as e:
            print(f"Detector error: {e}")
            continue
        
        # Normalize results handling (ultralytics may return a Results object or a list)
        boxes = None
        if hasattr(results, 'boxes'):
            boxes = results.boxes
        elif isinstance(results, (list, tuple)) and len(results) > 0 and hasattr(results[0], 'boxes'):
            boxes = results[0].boxes

        # 6.3. Update State Tracker (if any boxes/tracks were returned)
        if boxes is not None:
            try:
                tracker.update_frame(frame, boxes, embedder, db_conn)
            except Exception as e:
                print(f"Tracker update error: {e}")

        # 6.4. Visualization (for your demo)
        try:
            if hasattr(results, 'plot'):
                annotated_frame = results.plot()
            elif isinstance(results, (list, tuple)):
                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame
        except Exception:
            annotated_frame = frame
        
        # Add a title with the current unique visitor count (from the DB)
        try:
            with db_conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM Visitors;")
                count_row = cur.fetchone()
                count = count_row[0] if count_row else 0
            cv2.putText(annotated_frame, f"Unique Visitors Registered: {count}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception:
            pass # DB connection might be busy

        cv2.imshow("Intelligent Face Tracker", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    # 7. Cleanup
    cap.release()
    cv2.destroyAllWindows()
    db_conn.close()
    print("Processing finished.")

if __name__ == "__main__":
    main()