import json
import _init_paths
from typing import List, Dict
from BoundingBoxes import BoundingBoxes
from BoundingBox import BoundingBox
from utils import BBType
from Evaluator import Evaluator


def _convert_dict_to_bboxes(bbox_list: List[Dict], bb_type: BBType) -> BoundingBoxes:
    ret = BoundingBoxes()

    for elem in bbox_list:
        image_name = elem["file"]
        image_size = (elem["width"], elem["height"])

        bboxes_elem = elem["bboxes"]

        for b in bboxes_elem:
            conf = b.get("conf")
            bbx = BoundingBox(
                imageName=image_name,
                imgSize=image_size,
                classId=b["class"],
                x=b["xmin"],
                y=b["ymin"],
                w=(b["xmax"] - b["xmin"]),
                h=(b["ymax"] - b["ymin"]),
                bbType=bb_type,
                classConfidence=conf
            )
            ret.addBoundingBox(bbx)
    return ret


def calc_accuracy_metrics(gt_json: str, dt_json: str) -> List[Dict]:
    gt_json = gt_json.replace("\n", "")
    dt_json = dt_json.replace("\n", "")

    gt_dict = json.loads(gt_json) # type: List[Dict]
    dt_dict = json.loads(dt_json) # type: List[Dict]

    gt_bboxes = _convert_dict_to_bboxes(gt_dict, BBType.GroundTruth)
    dt_bboxes = _convert_dict_to_bboxes(dt_dict, BBType.Detected)

    all_boxes = BoundingBoxes()
    all_boxes._boundingBoxes.extend(gt_bboxes._boundingBoxes)
    all_boxes._boundingBoxes.extend(dt_bboxes._boundingBoxes)

    eval = Evaluator()
    ret = eval.GetPascalVOCMetrics(all_boxes)
    return ret

def calc_mean_average_precision(gt_json: str, dt_json: str) -> float:
    all_metrics = calc_accuracy_metrics(gt_json, dt_json)
    valid_classes = 0
    ap_sum = 0
    for metrics_per_class in all_metrics:
        ap = metrics_per_class['AP']
        totalPositives = metrics_per_class['total positives']

        if totalPositives > 0:
            valid_classes += 1
            ap_sum += ap
    mAP = ap_sum / valid_classes
    return mAP


if __name__ == "__main__":
    gt_str = open("gt_bbox_format.json").read().replace("\n", "")
    dt_str = open("dt_bbox_format.json").read().replace("\n", "")
    ret = calc_mean_average_precision(gt_str, dt_str)
    print(ret)