import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import cv2

from typing import *

def draw_bounding_box(
    image: Union[torch.Tensor, np.ndarray],
    boxes: Union[torch.Tensor, np.ndarray, List, Tuple],
    labels: List[str],
    colors: List[Tuple[int]],
    l: int = 30,
    t: int = 3,
    alpha: float = 0.2,
) -> np.ndarray:

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu()
        image = image.numpy()

    _image = image.copy()
    _image = (_image - _image.min())/_image.ptp()
    _image = (_image * 255.0)

    output_image = _image.copy()
    overlay = _image.copy()

    fontsize = 0.0022 * (image.shape[0] * image.shape[1]) ** 0.5
    font_thickness = int(0.004 * (image.shape[0] * image.shape[1]) ** 0.5)
    line_thickness = int(0.005 * (image.shape[0] * image.shape[1]) ** 0.5)
    border_length = line_thickness * 9

    for bbox, label, color in zip(boxes.long().cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes, labels, colors):
        x, y, x1, y1 = bbox
        w = x1 - x
        h = y1 - y

        # color = (0, 160, 255)

        cv2.rectangle(overlay, bbox, color, -1)
        
        output_image = cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0)

        cv2.rectangle(output_image, bbox, color, max(1, int(line_thickness / 3)), )
        
        if w > int(border_length) and w > int(border_length):
            cv2.line(output_image, (x, y), (x + border_length, y), color, line_thickness)
            cv2.line(output_image, (x, y), (x, y + border_length), color, line_thickness)

            cv2.line(output_image, (x1, y), (x1 - border_length, y), color, line_thickness)
            cv2.line(output_image, (x1, y), (x1, y +border_length), color, line_thickness)

            cv2.line(output_image, (x, y1), (x + border_length, y1), color, line_thickness)
            cv2.line(output_image, (x, y1), (x, y1 - border_length), color, line_thickness)

            cv2.line(output_image, (x1, y1), (x1 - border_length, y1), color, line_thickness)
            cv2.line(output_image, (x1, y1), (x1, y1 - border_length), color, line_thickness)

        cv2.putText(output_image, label, (bbox[0] - 2 * int(font_thickness), bbox[1] - int(font_thickness * 2)), 1, fontsize, color, font_thickness)

    return output_image