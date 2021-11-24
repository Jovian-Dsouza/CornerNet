import torch
import numpy as np

from utils.utils import non_max_suppression
from utils.keypoint import _decode
from utils.visualizations import draw_bounding_box, color_list

def post_process(images, output, 
                label_dict,
                prob_threshold=0.5, 
                iou_threshold=0.5, 
                ae_threshold = 0.5, # paireing threshold 
                topk = 100):
    '''
    Takes the input images and model output and draws the predictions
    Args:
        images : Torch.Tensor , (-1, c, h, w)
        output : Torch.Tensor
        label_dict : list, note: first entry in list is assumed to background
    Returns:
        list of output images : np.array of (h, w, c), len(list) == batch_size
    '''
    hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br = output
    batch_size, num_classes, hm_height, hm_width = hmap_tl.shape

    dets = _decode(*output, ae_threshold=ae_threshold, K=topk, kernel=3, num_dets=topk)
    dets = dets.reshape(dets.shape[0], -1, 8).cpu()
    # dets => (batch_size, 100, 8) => bboxes[0-4], scores[4], scores_tl[5], scores_br[6], classes[7]

    # Rescale bounding box between 0 and 1
    dets[..., 0:4:2] /= hm_width
    dets[..., 1:4:2] /= hm_height

    # for each batch
    output_images = []
    for batch_idx in range(batch_size):
        dets_nms = non_max_suppression(dets[batch_idx], prob_threshold, iou_threshold)
        if len(dets_nms) > 0:
            ilabels = dets_nms[:, -1]
            boxs = np.array(dets_nms[:, :4])
            labels = [label_dict[int(ilabel)+1] for ilabel in ilabels]
            colors = [color_list[int(ilabel)] for ilabel in ilabels]            
            # print(labels)
            output_images.append(draw_bounding_box(images[batch_idx], boxs, labels, colors=colors))
        else:
            image = images[batch_idx]
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).cpu()
                image = image.numpy()
            image = (image - image.min())/image.ptp()
            image = (image * 255.0).astype(np.uint8)
            output_images.append(image)
    return output_images