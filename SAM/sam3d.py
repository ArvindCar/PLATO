from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from torchvision.ops import box_convert
import torch
import supervision as sv
import open3d as o3d
import robomail.vision as vis

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

class get_bbox():
    def __init__(self):
        self.model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights_get_bbox/groundingdino_swint_ogc.pth")
    
    def __run__(self,
                IMAGE_PATH:str,
                prompt:str, 
                BOX_TRESHOLD = 0.35,
                TEXT_TRESHOLD = 0.25):
        """
        Responsible for getting the bounding box coordinates of the respective text prompt
        """
        
        image_source, image = load_image(IMAGE_PATH)
        boxes, logits, phrases = predict(
        model=self.model,
        image=image,
        caption=prompt,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
        )
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        #cv2.imwrite("annotated_image.jpg", annotated_frame)
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes = boxes, in_fmt = "cxcywh", out_fmt = "xyxy").numpy()
        detections = sv.Detections(xyxy= xyxy)
        box_coordinates = detections.xyxy[0]

        return box_coordinates
    
class SAM():
    def __init__(self):
        self.sam_checkpoint = "sam_vit_b_01ec64.pth" 
        self.model_type = 'vit_b'
        device = 'cuda:0'
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint).to(device=device)
        self.mask_predictor = SamPredictor(self.sam)
    
    def get_mask(self, IMAGE_PATH, bbox_coordinates, Saveimgat):
        image = cv2.imread(IMAGE_PATH)
        mask_predictor = self.mask_predictor
        mask_predictor.set_image(image)
        bbox_of_prompt = np.array(bbox_coordinates)
        masks, _, _ = mask_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox_of_prompt,
            multimask_output=False,
        )
        mask = masks.squeeze()
        mask_uint8 = (mask * 255).astype(np.uint8)
        cv2.imwrite(Saveimgat, mask_uint8)
        print('Completed')

def main():
    #get the segmented part in the image
    # need to decide on the box-threshold we need to set
    get_associated_bbox = get_bbox()
    bbox = get_associated_bbox.__run__('try_images/cimage2.png', 'cap')
    
    #given bounding box and the image, predicts the mask in the image
    segment_class = SAM()
    segment_class.get_mask('try_images/cimage2.png', bbox)

if __name__ == '__main__':
    #main()
    cloud = o3d.io.read_point_cloud('pointcloud.ply')
    o3d.visualization.draw_geometries([cloud]) 