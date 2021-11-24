import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

from typing import *

color_list = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), 
             (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), 
            (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), 
            (0, 0, 128), (128, 128, 128)]

def draw_bounding_box(
    image: Union[torch.Tensor, np.ndarray],
    boxes: Union[torch.Tensor, np.ndarray, List, Tuple],
    labels: List[str],
    colors: List[Tuple[int]],
    l: int = 30,
    t: int = 3,
    alpha: float = 0.2,
    text_color = None,
) -> np.ndarray:

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu()
        image = image.numpy()

    _image = image.copy()
    _image = (_image - _image.min())/_image.ptp()
    _image = (_image * 255.0).astype(np.uint8)

    output_image = _image.copy()
    overlay = _image.copy()

    img_h, img_w, _ = _image.shape

    fontsize = 0.0030 * (image.shape[0] * image.shape[1]) ** 0.5
    font_thickness = int(0.004 * (image.shape[0] * image.shape[1]) ** 0.5)
    line_thickness = int(0.005 * (image.shape[0] * image.shape[1]) ** 0.5)
    border_length = line_thickness * 9

    boxes = boxes.long().cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes

    for bbox, label, color in zip(boxes, labels, colors):
        # print(bbox)
        x, y, x1, y1 = bbox
        x *= img_w
        x1 *= img_w
        y1 *= img_h
        y *= img_h
        x, y, x1, y1 = int(x), int(y), int(x1), int(y1)
        
        w = x1 - x
        h = y1 - y

        cv2.rectangle(overlay, (x,y), (x1, y1), color, -1)
        
        output_image = cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0)

        cv2.rectangle(output_image, (x,y), (x1, y1), color, max(1, int(line_thickness / 3)), )
        
        if w > int(border_length) or h > int(border_length):
            cv2.line(output_image, (x, y), (x + border_length, y), color, line_thickness)
            cv2.line(output_image, (x, y), (x, y + border_length), color, line_thickness)

            cv2.line(output_image, (x1, y), (x1 - border_length, y), color, line_thickness)
            cv2.line(output_image, (x1, y), (x1, y +border_length), color, line_thickness)

            cv2.line(output_image, (x, y1), (x + border_length, y1), color, line_thickness)
            cv2.line(output_image, (x, y1), (x, y1 - border_length), color, line_thickness)

            cv2.line(output_image, (x1, y1), (x1 - border_length, y1), color, line_thickness)
            cv2.line(output_image, (x1, y1), (x1, y1 - border_length), color, line_thickness)

        x_text = x - 2 * int(font_thickness)
        y_text = y - int(font_thickness * 2)
        x_text = max(x_text, 0)
        y_text = max(y_text, int(10*fontsize))
        cv2.putText(output_image, label, (x_text, y_text), 1, fontsize, color if text_color is None else text_color, font_thickness)

    return output_image

def plot_hmap(hmap, label=0):
    hmap = hmap.squeeze(0).cpu()
    plt.imshow(hmap[0])
    plt.xticks([])
    plt.yticks([])
    plt.title(label)
    plt.show()

def plot_image(image):
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()

# def save_image()