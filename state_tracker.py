import time
import os
import datetime
import cv2
import numpy as np
import logging
import database  # Our database module

# Configure the system-wide event logger
logging.basicConfig(
    filename='logs/events.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class VisitorTracker:
    """
    Manages the state of tracked visitors to ensure robust, "exactly one"
    entry/exit logging per visit.
    """
    def __init__(self, config):
        self.similarity_threshold = config.get('similarity_threshold', 0.6)
        self.exit_timeout = config.get('exit_timeout_seconds', 3.0)
        self.entry_log_dir = config.get('entry_log_dir', 'logs/entries')
        
        # State Dictionaries:
        
        # 1. {track_id (int): {'visitor_id': UUID, 'last_crop': np.array}}
        #    Maps a *transient* ByteTrack ID to our *persistent* visitor_ID
        #    and stores the last known image crop for exit logging.
        self.active_tracks = {}

        # 2. {visitor_id (UUID): {'timestamp': float, 'last_crop': np.array}}
        #    A buffer for visitors who have disappeared from frame.
        #    If they don't reappear within self.exit_timeout, they are
        #    logged as an 'exit'.
        self.pending_exit = {}
        
        # 3. {visitor_id (UUID)}
        #    A set of visitors who have already logged an 'entry' event
        #    for their *current visit*. This prevents duplicate entry logs.
        self.logged_entry_this_visit = set()
        
        # Ensure log directories exist
        os.makedirs(self.entry_log_dir, exist_ok=True)
        logging.info("VisitorTracker initialized.")

    def _log_system_event(self, message):
        """Helper to log to both console and file."""
        print(message)
        logging.info(message)

    def _save_cropped_face(self, crop_img, visitor_id, event_type):
        """
        Saves a cropped face image to the filesystem.
        Logs to a dated folder structure as required.[12, 13, 1]
        """
        try:
            today_str = datetime.datetime.now().strftime('%Y-%m-%d')
            today_dir = os.path.join(self.entry_log_dir, today_str)
            os.makedirs(today_dir, exist_ok=True)
            
            timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{visitor_id}_{event_type}_{timestamp_str}.jpg"
            filepath = os.path.join(today_dir, filename)
            
            cv2.imwrite(filepath, crop_img)
            return filepath
        except Exception as e:
            self._log_system_event(f"ERROR: Failed to save image for {visitor_id}: {e}")
            return None

    def update_frame(self, frame, tracks, embedder, db_conn):
        """
        Main logic loop. Processes all tracks from a single frame.
        'tracks' is the results.boxes object from Ultralytics.
        """
        
        current_track_ids = set()
        current_visitor_ids_in_frame = set()

        if tracks.id is None:
            # No tracks in this frame
            pass
        else:
            track_ids = tracks.id.int().cpu().tolist()
            bboxes = tracks.xyxy.int().cpu().tolist()
            
            # --- LOOP 1: Identify all tracks in the current frame ---
            for track_id, bbox in zip(track_ids, bboxes):
                current_track_ids.add(track_id)
                x1, y1, x2, y2 = bbox
                crop_img = frame[y1:y2, x1:x2]

                visitor_id = None
                
                if track_id in self.active_tracks:
                    # 1.1: This is a known track
                    visitor_id = self.active_tracks[track_id]['visitor_id']
                else:
                    # 1.2: This is a new track. Get embedding.
                    embedding = embedder.get_embedding(crop_img)
                    
                    if embedding is None:
                        continue # Bad crop, skip this track for now

                    # 1.3: Check if this face is already in our DB
                    visitor_id, sim = database.find_visitor(db_conn, embedding, self.similarity_threshold)
                    
                    if visitor_id is None:
                        # 1.4: New Unique Visitor. Register them.
                        visitor_id = database.register_new_visitor(db_conn, embedding)
                        if visitor_id:
                            self._log_system_event(f"AUTO-REGISTER: New unique visitor detected: {visitor_id}")
                        else:
                            continue # Failed to register, skip
                    else:
                        self._log_system_event(f"RE-ID: Recognized returning visitor {visitor_id} (Sim: {sim:.2f})")

                    # 1.5: Add this new track_id to our active state
                    self.active_tracks[track_id] = {
                        'visitor_id': visitor_id,
                        'last_crop': crop_img
                    }

                if visitor_id:
                    current_visitor_ids_in_frame.add(visitor_id)
                    # Always update the last_crop image for this active track
                    self.active_tracks[track_id]['last_crop'] = crop_img


        # --- LOOP 2: Handle Entry/Re-appearance Logic ---
        for visitor_id in current_visitor_ids_in_frame:
            # 2.1: If this person was in the 'pending_exit' buffer,
            #      they just re-appeared. Cancel their exit.
            if visitor_id in self.pending_exit:
                self.pending_exit.pop(visitor_id)

            # 2.2: If this person is not in the 'logged_entry' set,
            #      it's their first appearance this visit. Log ENTRY.
            if visitor_id not in self.logged_entry_this_visit:
                # Find their crop image (must be in active_tracks)
                crop_to_log = None
                for track_data in self.active_tracks.values():
                    if track_data['visitor_id'] == visitor_id:
                        crop_to_log = track_data['last_crop']
                        break
                
                if crop_to_log is not None:
                    img_path = self._save_cropped_face(crop_to_log, visitor_id, 'entry')
                    if img_path:
                        database.log_event(db_conn, visitor_id, 'entry', img_path)
                        self._log_system_event(f"EVENT: 'ENTRY' logged for {visitor_id}")
                        self.logged_entry_this_visit.add(visitor_id)


        # --- LOOP 3: Handle Disappearances (Start Exit Timer) ---
        lost_track_ids = set(self.active_tracks.keys()) - current_track_ids
        
        for track_id in lost_track_ids:
            track_data = self.active_tracks.pop(track_id)
            visitor_id = track_data['visitor_id']
            last_crop = track_data['last_crop']
            
            # Is this visitor *still* in the frame under a *different* track_id?
            is_still_visible = visitor_id in current_visitor_ids_in_frame
            
            if not is_still_visible:
                # This visitor is truly gone. Start their exit timer.
                self.pending_exit[visitor_id] = {
                    'timestamp': time.time(),
                    'last_crop': last_crop
                }
                
        # --- LOOP 4: Process Final Exits (Check Timeout Buffer) ---
        current_time = time.time()
        
        # Use list() to allow modifying dict during iteration
        for visitor_id, exit_data in list(self.pending_exit.items()):
            time_disappeared = current_time - exit_data['timestamp']
            
            if time_disappeared > self.exit_timeout:
                # 4.1: Timeout exceeded. Log 'EXIT'.
                last_crop = exit_data['last_crop']
                img_path = self._save_cropped_face(last_crop, visitor_id, 'exit')
                
                if img_path:
                    database.log_event(db_conn, visitor_id, 'exit', img_path)
                    self._log_system_event(f"EVENT: 'EXIT' logged for {visitor_id} (disappeared for {time_disappeared:.2f}s)")
                
                # 4.2: Remove from pending AND from logged_entry_this_visit
                #      This "resets" them, allowing a new 'entry' log
                #      if they return later.
                self.pending_exit.pop(visitor_id)
                if visitor_id in self.logged_entry_this_visit:
                    self.logged_entry_this_visit.remove(visitor_id)