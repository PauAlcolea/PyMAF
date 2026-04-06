import os
from typing import Dict, List, Optional, Union

import numpy as np
try:
    from multi_person_tracker import Sort  # type: ignore
except Exception:
    Sort = None
from ultralytics import YOLO


def _bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


class _SimpleSort:
    """Minimal IoU tracker fallback when multi_person_tracker is unavailable."""

    def __init__(self, iou_thr: float = 0.3, max_age: int = 5) -> None:
        self.iou_thr = float(iou_thr)
        self.max_age = int(max_age)
        self.next_id = 1
        self.frame_idx = -1
        self.tracks = {}  # id -> {"bbox": np.ndarray(4), "last": int}

    def update(self, detections: np.ndarray) -> np.ndarray:
        self.frame_idx += 1
        dets = np.asarray(detections, dtype=np.float32)
        if dets.size == 0:
            self._expire_tracks()
            return np.empty((0, 5), dtype=np.float32)

        boxes = dets[:, :4]
        assigned_tracks = set()
        out_rows = []

        for box in boxes:
            best_id = None
            best_iou = 0.0
            for tid, t in self.tracks.items():
                if tid in assigned_tracks:
                    continue
                iou = _bbox_iou_xyxy(box, t["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_id is None or best_iou < self.iou_thr:
                best_id = self.next_id
                self.next_id += 1

            assigned_tracks.add(best_id)
            self.tracks[best_id] = {"bbox": box.copy(), "last": self.frame_idx}
            out_rows.append([box[0], box[1], box[2], box[3], float(best_id)])

        self._expire_tracks()
        return np.asarray(out_rows, dtype=np.float32)

    def _expire_tracks(self) -> None:
        alive = {}
        for tid, t in self.tracks.items():
            if (self.frame_idx - int(t["last"])) <= self.max_age:
                alive[tid] = t
        self.tracks = alive


class MPT8:
    def __init__(
            self,
            model_type: str = 'yolov8n.pt',
            output_format: str = 'dict',
    ) -> None:
        self.model = YOLO(model_type)
        self.extensions = set(['jpg', 'jpeg', 'png'])
        self.output_format = output_format

    def __call__(self, image_folder: str) -> Optional[Union[Dict, List]]:
        image_paths = sorted(
            [
                os.path.join(image_folder, filename)
                for filename in os.listdir(image_folder)
                if self._check_extension(filename)
            ]
        )
        tracker = Sort() if Sort is not None else _SimpleSort()
        trackers = []
        # predictions = self.model(image_paths)  # too much of ram! 6.5GB vs <2GB
        for image_path in image_paths:
            # infere yolo model
            prediction = self.model(image_path)[0].boxes.boxes.cpu().numpy()
            # filter non-human objects, remove class dim
            prediction = prediction[prediction[:, 5] == 0][:, :5]
            # track objects
            if prediction.shape[0] > 0:
                track_trackers = tracker.update(prediction)
            else:
                track_trackers = np.empty((0, 5))
            trackers.append(track_trackers)
        if self.output_format == 'dict':
            result = self._prepare_output_tracks(trackers)
        elif self.output_format == 'list':
            result = trackers
        else:
            raise ValueError(
                'output_format should be either "dict" or "list", while set '
                f'to {self.output_format}'
            )
        return result

    def _check_extension(self, filename: str) -> bool:
        return filename.split('.')[-1].lower() in self.extensions

    def _prepare_output_tracks(self, trackers):
        '''
        Put results into a dictionary consists of detected people
        :param trackers (ndarray): input tracklets of shape Nx5
            [x1,y1,x2,y2,track_id]
        :return: dict: of people. each key represent single person with
            detected bboxes and frame_ids

        *borrowed from repo: mkocabas/multi-person-tracker
        file: multi-person-tracker/multi_person_tracker/mpt.py
        '''
        people = dict()

        for frame_idx, tracks in enumerate(trackers):
            for d in tracks:
                person_id = int(d[4])

                w, h = d[2] - d[0], d[3] - d[1]
                c_x, c_y = d[0] + w/2, d[1] + h/2
                w = h = np.where(w / h > 1, w, h)
                bbox = np.array([c_x, c_y, w, h])

                if person_id in people.keys():
                    people[person_id]['bbox'].append(bbox)
                    people[person_id]['frames'].append(frame_idx)
                else:
                    people[person_id] = {
                        'bbox' : [],
                        'frames' : [],
                    }
                    people[person_id]['bbox'].append(bbox)
                    people[person_id]['frames'].append(frame_idx)
        for k in people.keys():
            people[k]['bbox'] = (
                np
                .array(people[k]['bbox'])
                .reshape((len(people[k]['bbox']), 4))
            )
            people[k]['frames'] = np.array(people[k]['frames'])

        return people
