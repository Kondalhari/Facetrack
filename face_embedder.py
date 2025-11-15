import insightface
import numpy as np
import cv2


class FaceEmbedder:
    """
    Wrapper class for the InsightFace FaceAnalysis model.
    """
    def __init__(self, model_root='models', use_gpu=False, model_name='buffalo_l'):
        print("Loading InsightFace model... This may take a moment.")
        # Build FaceAnalysis app
        try:
            self.app = insightface.app.FaceAnalysis(name=model_name, root=model_root)
            # ctx_id = 0 for GPU; -1 for CPU. Default to CPU unless use_gpu=True
            ctx_id = 0 if use_gpu else -1
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            print("InsightFace model loaded.")
        except Exception as e:
            print(f"Failed to initialize InsightFace: {e}")
            raise

    def get_embedding(self, cropped_face_img):
        """
        Generates a 512-dimension embedding for a single cropped face image.

        """
        try:
            # FaceAnalysis.get returns a list of Face objects
            faces = self.app.get(cropped_face_img)

            if faces and len(faces) > 0:
                # Return the embedding of the first face
                first_face = faces[0]
                return first_face.embedding
            else:
                return None
        except Exception as e:
            print(f"Error during embedding generation: {e}")
            return None