import torch
from torch2trt import TRTModule
import cv2
import torchvision.transforms as transforms
import PIL.Image
import numpy as np
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import json
import trt_pose.coco

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)


topology = trt_pose.coco.coco_category_to_topology(human_pose)
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)



def preprocess(image):
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    image = transforms.functional.to_tensor(image).cuda()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


if __name__ == "__main__":
    OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
    image = cv2.imread('demo.jpg')
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, image = cap.read()
        pil = PIL.Image.fromarray(image[:, :, ::-1]).resize((224, 224))
        input_tensor = preprocess(pil)
        cmap, paf = model_trt(input_tensor)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)
        draw_objects(image, counts, objects, peaks)
        cv2.imshow('DEMO', image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
