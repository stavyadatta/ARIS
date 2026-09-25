from PIL import Image
from queue import Queue
from threading import Thread
from collections import Counter, deque
from transformers import pipeline, Pipeline

class _ClipClassification():
    """
    Clip classification from huggingface
    """

    def __init__(self, checkpoint="openai/clip-vit-large-patch14") -> None:
        pass
        # self.detector: Pipeline = pipeline(model=checkpoint, task="zero-shot-image-classification")
        self.img_queue = Queue(maxsize=15)
        # self.face_class_queue = deque(maxlen=15)
        #
        # face_class_thread = Thread(
        #     target=self._face_class_on_queue,
        #     daemon=True
        # )
        # face_class_thread.start()

    def add2clip_img_queue(self, image):
        self.img_queue.put(image)

    def _convert_cv2pil(self, recent_images):
        pil_imgs = []
        for cv_img in recent_images:
            pil_img = Image.fromarray(cv_img)
            pil_imgs.append(pil_img)

        return pil_imgs

    def _get_dominant_label(self, predictions_iterator):
        labels = []

        for _, preds in enumerate(predictions_iterator):
            # Get the top prediction for this image
            top_pred = max(preds, key=lambda x: x['score'])

            # Use the label only if confidence is at least 0.5
            if top_pred['score'] >= 0.5:
                labels.append(top_pred['label'])
            else:
                labels.append("no_face")

        # Count the labels and find the most common one
        most_common_label, _ = Counter(labels).most_common(1)[0]
        return most_common_label

    def _face_class_on_queue(self):
        candidate_labels = ["front_face", "slight_side_face", "side_face", "no_face"]
        while True:
            img = self.img_queue.get()
            pil_img = Image.fromarray(img)
            try:
                predictions = self.detector(pil_img, candidate_labels=candidate_labels)
                pred = predictions[0]
            except Exception as e:
                print('This is the error in clip_classification', e)
                self.face_class_queue.append("no_face")
                continue

            score = pred.get("score")
            label = pred.get("label")

            if score > 0.5:
                self.face_class_queue.append(label)
            else:
                self.face_class_queue.append("no_face")

    def get_most_face_class(self):
        last_10_classes = list(self.face_class_queue)[-10:]
        counts = Counter(last_10_classes)
        most_common_class, _ = counts.most_common(1)[0]
        return most_common_class
