import cv2
import glob
import math
import torch
import logging
import argparse
import numpy as np
from pathlib import Path
from queue import Queue
from threading import Thread
from collections import deque
from typing import List, Tuple, Optional
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

class _FaceRecognition:
    """
    A class for face recognition using the InsightFace library.

    This class:
    - Loads existing face embeddings from a database folder.
    - Detects faces and generates embeddings for images provided as numpy arrays.
    - Compares the generated embedding against known embeddings using cosine similarity.
    - If no match is found (based on a threshold), it can save the new face into the database
      (but only when explicitly instructed, not by default).
    """

    def __init__(self, 
                 db_dir: str = "/workspace/database/face_db",
                 model_name: str = "buffalo_l",
                 recognition_threshold: float = 0.55):
        """
        Initialize the FaceRecognition class.

        Args:
            db_dir (str): Directory path for face database.
            model_name (str): InsightFace model name.
            recognition_threshold (float): Threshold for considering a face as known.
        """
        self.db_dir = Path(db_dir)
        self.recognition_threshold = recognition_threshold
        self.model_name = model_name

        # Ensure database directory exists
        self._ensure_db_directory()

        # Initialize face analysis model
        self.app = self._initialize_face_analysis()

        # Load database embeddings
        self.known_ids, self.known_embeddings = self._load_database()
        self.model_points = self._get_3d_model_points()
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        self.face_img_queue = Queue(maxsize=15)
        self.face_id_queue = deque(maxlen=15)
        self.save_img_queue = deque(maxlen=15)
        self.face_embedding_queue = deque(maxlen=15)

        face_recognition_thread = Thread(
            target=self._face_recognition_on_queue,
            daemon=True
        )

        face_recognition_thread.start()

    def add2face_img_queue(self, image):
        self.face_img_queue.put(image)

    def _ensure_db_directory(self):
        """Ensure that the database directory exists."""
        if not self.db_dir.exists():
            self.db_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_face_analysis(self) -> FaceAnalysis:
        """Initialize the InsightFace FaceAnalysis object and load models."""
        providers = [('CUDAExecutionProvider', {"device_id": 0}), 'CPUExecutionProvider'] \
            if torch.cuda.is_available() else ['CPUExecutionProvider']
        app = FaceAnalysis(name=self.model_name, providers=providers)
        app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_thresh=0.7)
        return app

    def _get_3d_model_points(self):
        """
        Defines a generic 3D model of the 5 facial keypoints for head pose estimation.
        Order: Right Eye, Left Eye, Nose Tip, Right Mouth Corner, Left Mouth Corner
        """
        model_points = np.array([
            [-30.0,  30.0, -30.0],  # Right eye
            [ 30.0,  30.0, -30.0],  # Left eye
            [  0.0,   0.0,   0.0],  # Nose tip
            [-25.0, -30.0, -30.0],  # Right mouth corner
            [ 25.0, -30.0, -30.0]   # Left mouth corner
        ], dtype=np.float32)
        return model_points


    def _get_camera_matrix(self, img_shape):
        """
        Builds camera intrinsic matrix using SoftBank Pepper OV5640 specs:
        - Resolution: 640x480
        - Horizontal FOV: 56.3°
        - Vertical FOV:   43.7°
        """
        h, w = img_shape[:2]
        # Convert FOVs to radians
        hfov = np.deg2rad(56.3)
        vfov = np.deg2rad(43.7)
        # Compute focal lengths in pixels
        f_x = (w / 2) / np.tan(hfov / 2)
        f_y = (h / 2) / np.tan(vfov / 2)
        # Optical center
        c_x = w / 2
        c_y = h / 2

        camera_matrix = np.array([
            [f_x,   0, c_x],
            [  0, f_y, c_y],
            [  0,   0,   1]
        ], dtype=np.float32)
        return camera_matrix

    def _rotation_matrix_to_euler_angles(self, R):
        """
        Converts rotation matrix to Euler angles (pitch, yaw, roll) in degrees using ZYX convention.
        """
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.degrees([x, y, z])

    def _is_side_face(self, face, cam_matrix):
        x1, y1, x2, y2 = face.bbox.astype(int)
        kps = face.kps.astype(np.float32)
        image_points = kps
        success, rvec, tvec = cv2.solvePnP(
            self.model_points, image_points, cam_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success:
            print("Side facePnp not working")


        R, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = self._rotation_matrix_to_euler_angles(R)

        if abs(yaw) > 45.0:
            return True
        else:
            return False

    def _load_database(self) -> Tuple[List[str], np.ndarray]:
        """
        Load the known face embeddings and IDs from the database directory.

        Returns:
            Tuple[List[str], np.ndarray]: A list of face IDs and a numpy array of their embeddings.
        """
        embedding_files = glob.glob(str(self.db_dir / "*.npy"))
        known_ids = []
        known_embeddings = []

        for ef in embedding_files:
            face_id = Path(ef).stem  # filename without extension
            emb = np.load(ef)
            known_ids.append(face_id)
            known_embeddings.append(emb)

        if known_embeddings:
            known_embeddings = np.vstack(known_embeddings)
        else:
            known_embeddings = np.array([])

        return known_ids, known_embeddings

    def _get_face_area(self, face):
        x1, y1, x2, y2 = [int(i) for i in face.bbox]
        return abs((x2 - x1) * (y2 - y1))

    def _get_embedding(self, img: np.ndarray, skip_validation: bool = False) -> np.ndarray:
        """
        Given an image array, detect the face, and generate a face embedding.
        if its a valid face embedding

        Args:
            img (np.ndarray): The image array.
            skip_validation (bool): If True, skip face area and side-face checks
                                    (still requires a face to be detected).

        Returns:
            np.ndarray: The face embedding vector of shape (1, embedding_dim).
        """
        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError("No face detected in the given image.")

        cam_matrix = self._get_camera_matrix(img.shape)

        reason = ""
        for face in faces:
            embedding = face.embedding

            if not skip_validation:
                area = self._get_face_area(face)
                is_side_face = self._is_side_face(face, cam_matrix)

                # Check if Area is valid
                if area < 4500:
                    reason = "Area too small, {}".format(area)
                    continue

                if is_side_face:
                    reason = "Side face was detected"
                    continue

            embedding = embedding.reshape(1, -1)  # shape: (1, embedding_dim)
            return embedding

        print("Face not recognised because ", reason)
        raise ValueError("The face detected were invalid")

    def _match_face(self, embedding: np.ndarray) -> Optional[str]:
        """
        Match the given embedding against known embeddings.

        Args:
            embedding (np.ndarray): The face embedding to match.

        Returns:
            Optional[str]: The matched face ID if found, otherwise None.
        """
        if self.known_embeddings.size == 0:
            return None

        # Compute cosine similarity
        sim = cosine_similarity(embedding, self.known_embeddings)
        best_match_idx = np.argmax(sim)
        best_score = sim[0, best_match_idx]

        # Threshold check (closer to 1 is more similar)
        # We interpret "recognition_threshold" as the maximum distance from 1 
        # allowed. i.e. if best_score >= (1 - threshold) => recognized
        if best_score >= (1 - self.recognition_threshold):
            return self.known_ids[best_match_idx]
        else:
            return None

    def _save_new_face(self, embedding: np.ndarray, img: np.ndarray, save_img=True) -> str:
        """
        Save a new face to the database (embedding and optionally the face image).

        Returns:
            str: The ID of the newly saved face.
        """
        # Generate a new unique ID
        new_id = self._generate_new_face_id()

        # Save embedding to disk
        np.save(self.db_dir / f"{new_id}.npy", embedding)

        # Update known faces in memory
        self.known_ids.append(new_id)
        if self.known_embeddings.size == 0:
            self.known_embeddings = embedding
        else:
            self.known_embeddings = np.vstack([self.known_embeddings, embedding])
        
        # Optionally save the face image
        if save_img:
            out_path = self.db_dir / f"{new_id}.png"
            cv2.imwrite(str(out_path), img)

        return new_id

    def _generate_new_face_id(self) -> str:
        """
        Generate a unique face ID by counting existing IDs.

        Returns:
            str: A new face ID, for example 'face_1', 'face_2', etc.
        """
        current_ids = [int(x.replace("face_", "")) 
                       for x in self.known_ids if x.startswith("face_")]
        next_id_num = (max(current_ids) + 1) if current_ids else 1
        return f"face_{next_id_num}"

    ############################################################################
    #               Key Changes: Separate "recognize" vs. "enroll"             #
    ############################################################################
    def recognize_face_no_enroll(self, img: np.ndarray, skip_validation: bool = False) -> Tuple[Optional[str], np.ndarray]:
        """
        Attempt to recognize the face in the image but DO NOT enroll new faces.
        This method returns the recognized ID (or None if unknown) AND the embedding.

        Args:
            img (np.ndarray): The image array.
            skip_validation (bool): If True, skip face area and side-face checks.

        Returns:
            (face_id, embedding)
        """
        embedding = self._get_embedding(img, skip_validation=skip_validation)
        face_id = self._match_face(embedding)
        return face_id, embedding

    def enroll_face(self, embedding: np.ndarray, img: np.ndarray) -> str:
        """
        Explicitly enroll a new face into the database using the given embedding and image.

        Returns:
            str: the newly generated face ID
        """
        return self._save_new_face(embedding, img, save_img=True)

    def _face_recognition_on_queue(self):
        while True:
            img = self.face_img_queue.get()

            try:
                recognized_id, emb = self.recognize_face_no_enroll(img)
            except ValueError as e:
                self.face_id_queue.append(None)
                self.face_embedding_queue.append(None)
                self.save_img_queue.append(None)
                continue

            self.face_id_queue.append(recognized_id)
            self.face_embedding_queue.append(emb)
            self.save_img_queue.append(img)

    ############################################################################
    #            Modified method that does the voting over 10 frames           #
    ############################################################################

    def get_most_frequent_face_id(self) -> Optional[str]:
        """
        Process up to the last 10 images in the queue, attempt to recognize each face
        WITHOUT immediately enrolling any new face. If the final "winner" is None or
        if None-count is strictly greater than the most frequent recognized ID count,
        then enroll as a new face.

        Returns:
            Optional[str]: The most frequent recognized face ID or a newly enrolled ID.
        """
        recent_images = list(self.save_img_queue)[-10:]
        face_ids = list(self.face_id_queue)[-10:]
        embeddings = list(self.face_embedding_queue)[-10:]

        if len(face_ids) == 0:
            return None

        # 2) Count how often each ID appears, plus how many are None
        from collections import Counter
        counter = Counter(face_ids)  # e.g., {face_1: 3, face_2: 1, None: 6}
        none_count = counter[None]
        del counter[None]  # remove None from the recognized-IDs dict if it exists

        if len(counter) == 0:
            # There were no recognized IDs at all => everything was None
            most_frequent_id = None
            max_freq = 0
        else:
            # Identify the recognized ID with the highest frequency
            most_frequent_id, max_freq = max(counter.items(), key=lambda x: x[1])  # e.g. ("face_1", 3)

        # 3) If the number of None is strictly greater than the max recognized frequency,
        #    treat the result as "new/unknown" face.
        if none_count > max_freq:
            logging.info(f"None-count ({none_count}) > recognized-count ({max_freq}). "
                        "Treating as new face.")
            most_frequent_id = None

        # 4) If the final voted ID is None => truly unknown => we enroll as new face
        if most_frequent_id is None:
            # Enroll a new face from one of the unknown instances (e.g., the last unknown)
            for i in reversed(range(len(face_ids))):
                if face_ids[i] is None and embeddings[i] is not None:
                    new_face_id = self.enroll_face(embeddings[i], recent_images[i])
                    logging.info(f"No known face found in the last 10 images. "
                                f"Enrolled new face: {new_face_id}")
                    return new_face_id
            # If we never found a valid unknown embedding, just return None
            return None

        # 5) Otherwise, return the recognized majority face
        logging.info(f"Most frequent recognized ID is: {most_frequent_id} "
                    f"(count={max_freq}, none_count={none_count})")
        return most_frequent_id

    def recognize_face_relaxed(self, img: np.ndarray) -> Optional[str]:
        """
        One-shot face recognition with relaxed validation (no area/side-face checks).
        Still requires a face to be detected. If recognized returns face_id,
        if unknown enrolls as new face, if no face returns None.

        Args:
            img (np.ndarray): The image array.

        Returns:
            Optional[str]: The face ID or None if no face detected.
        """
        try:
            face_id, embedding = self.recognize_face_no_enroll(img, skip_validation=True)
            if face_id is not None:
                return face_id
            # Unknown face — enroll it
            new_id = self.enroll_face(embedding, img)
            logging.info(f"Relaxed recognition: enrolled new face {new_id}")
            return new_id
        except ValueError:
            return None

    def get_face_box(self, img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the face in the image and return the bounding box.

        Args:
            img (np.ndarray): The input image array (e.g., from cv2.imread).

        Returns:
            Optional[Tuple[int, int, int, int]]: Bounding box (x, y, w, h) if a face is detected, else None.
        """
        bboxes, _ = self.app.det_model.detect(img, max_num=0, metric='default')
        if bboxes.shape[0] == 0:
            # logging.warning("No person found")
            return None

        first_bbox = bboxes[0, 0:4].astype(int)
        return tuple(first_bbox)

def main():
    parser = argparse.ArgumentParser(description="Face recognition embedding extractor")
    parser.add_argument("--img_path", type=str, help="Path to the input image")
    parser.add_argument("--output_dir", type=str, default="./output_embeddings", help="Directory to save the embedding")
    args = parser.parse_args()

    img_path = Path(args.img_path)
    if not img_path.exists():
        logging.error(f"Image file not found: {img_path}")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        logging.error("Failed to read image.")
        return

    recognizer = _FaceRecognition()
    import time
    start_time = time.time()
    try:
        embedding = recognizer._get_embedding(img)
    except ValueError as e:
        logging.error(f"Face detection failed: {e}")
        return
    end_time = time.time()
    print(f"Batch prediction took {(end_time - start_time) * 1000:.2f} ms")
    exit()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / f"{img_path.stem}_embedding.npy"
    np.save(embedding_path, embedding)

    print(f"Embedding saved at: {embedding_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

