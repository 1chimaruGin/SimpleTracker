import cv2
import sys
import torch
import argparse
import numpy as np
from typing import List, Tuple
from norfair import Detection, Tracker

from boxes import non_max_suppression, scale_coords
from helper import ReadVideo, check_img_size, euclidean_distance

sys.path.insert(0, "yolov5")
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.plots import Annotator


class Detector:
    def __init__(
        self,
        model: str,
        imgsz: int = 640,
        classes: List[int] = [0],
        conf_thresh: float = 0.1,
        iou_thresh: float = 0.45,
        half: bool = False,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.classes = classes if classes else [0]

        model = DetectMultiBackend(model, device=self.device, dnn=False)
        
        stride, names, pt, jit, onnx, engine = (
            model.stride,
            model.names,
            model.pt,
            model.jit,
            model.onnx,
            model.engine,
        )
        self.model = model
        self.imgsz = check_img_size(imgsz, s=stride)  # check image size
        self.stride = stride
        self.pt = pt

        half &= (
            pt or jit or onnx or engine
        ) and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
        self.half = half
        if pt or jit:
            model.model.half() if self.half else model.model.float()

    def prepare(self, img: np.ndarray) -> torch.Tensor:
        im = torch.from_numpy(img).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        return im

    def post_process(self, pred: torch.Tensor, im: torch.Tensor, im0: torch.Tensor) -> torch.Tensor:
        pred = non_max_suppression(
            pred, self.conf_thresh, self.iou_thresh, classes=self.classes
        )
        output = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det.to('cpu').numpy()):
                    output.append((*xyxy, conf, cls))
        return output

    @torch.no_grad()
    def predict(
        self, img: np.ndarray, im0: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float, int]]:
        im = self.prepare(img)
        pred = self.model(im)
        return self.post_process(pred, im, im0)

def to_nofair(
        det: torch.Tensor, frameid: int
    ) -> List[Tuple[str, float, float, float, float]]:
    result = []
    for x_min, y_min, x_max, y_max, score, cls in det:
        xc, yc = (x_min + x_max) / 2, (y_min + y_max) / 2
        w, h = x_max - x_min, y_max - y_min
        result.append(
            Detection(
                points=np.array([xc, yc]),
                scores=np.array([score]),
                data=np.array([w, h, frameid]),
            )
        )
    return result

def track(
    source: str,
    model: str = "yolov5m6.pt",
    imgsz: int = 1280,
    classes: List[int] = [0],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    half: bool = False,
    save: bool = True,
) -> None:
    detector = Detector(
        model, imgsz=imgsz, classes=classes, conf_thresh=conf_thresh, iou_thresh=iou_thresh, half=half
    )
    stride = detector.stride

    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=30,
        initialization_delay=1,
    )

    dataset = ReadVideo(source, stride=stride)
    vid_path, vid_writer = [None], [None]
    for i, (img, im0) in enumerate(dataset):
        det = detector.predict(img, im0)
        tracked_objects = tracker.update(detections=to_nofair(det, frameid=i))
        annotator = Annotator(im0, line_width=3)
        for tobj in tracked_objects:
            bbox_width, bbox_height, _ = tobj.last_detection.data
          
            xc, yc = tobj.estimate[0]
            x_min, y_min = int(round(xc - bbox_width / 2)), int(round(yc - bbox_height / 2))
            xyxy = [x_min, y_min, x_min + bbox_width, y_min + bbox_height]
            annotator.box_label(xyxy, label=f'{tobj.id}' , color=(0, 255, 0))

        im0 = annotator.result()
        cv2.imshow("Tracking", im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='Source Video')
    parser.add_argument('--model', type=str, default='weights/yolov5s.pt',help='Model')
    parser.add_argument('--imgsz', type=int, default=1280, help='Image Size')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='Confidence Threshold')
    parser.add_argument('--iou_thresh', type=float, default=0.45, help='IoU Threshold')
    parser.add_argument('--half', action='store_true', help='Use half precision')

    args = parser.parse_args()

    if not args.source:
        print("Please specify a source video (e.g --source videos/test.mp4)")
        exit(1)

    track(args.source, args.model, args.imgsz, args.classes, args.conf_thresh, args.iou_thresh, args.half)
