'''Vision encoder backbone for ALFRED VLN BERT'''

import argparse
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import get_detection_dataset_dicts
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils
from detectron2.modeling import build_model
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.structures import Boxes
from detectron2.utils.events import EventStorage

from vision import (
    add_alfred_config,
    AlfredMappper,
    register_alfred_instances,
)


class MaskRCNN(object):
    '''
    pretrained MaskRCNN from torchvision
    '''

    def __init__(self, args, eval=True, share_memory=False, min_size=224):

        if not args.vis_model_path:
            self.vis_model = models.detection.maskrcnn_resnet50_fpn(pretrained=True, min_size=min_size)
            #  # get the number of input features for the classifier
            # in_features = maskrcnn.roi_heads.box_predictor.cls_score.in_features
            # # replace the pre-trained head with a new one
            # maskrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, args.num_classes)

            # # now get the number of input features for the mask classifier
            # in_features_mask = maskrcnn.roi_heads.mask_predictor.conv5_mask.in_channels
            # hidden_layer = args.hidden_size
            # # and replace the mask predictor with a new one
            # maskrcnn.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
            #                                                         hidden_layer,
            #                                                         args.num_classes)
            raise Exception
        else:
            # Load ALFRED-Detectron Config
            cfg = get_cfg()
            add_alfred_config(cfg)
            cfg.merge_from_file(args.vis_model_config)
            cfg.MODEL.WEIGHTS = args.vis_model_path
            cfg.freeze()
            self.cfg = cfg

            # Load data mapper
            self.mapper = AlfredMappper(cfg, is_train=False)
            
            # Load model from pretrained weights
            model = build_model(cfg)
            checkpointer = DetectionCheckpointer(model)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            self.vis_model = model
            

        if args.gpu:
            self.vis_model = self.model.to(torch.device('cuda'))

        if eval:
            self.vis_model.eval()


    def extract(self, batched_images):
        '''
        Extracts image feature from the Resnet backbone and
        object features from roi heads
        images: list of [C, H, W]
        returns: image_features, mask_features, masks, labels, reachability
        '''
        #images, _ = self.model.transform(images)

        images = self.vis_model.preprocess_image(batched_images)
        gt_instances = None

        pred_out = self.vis_model(batched_images)
        #print(pred_out)

        # Resnet & fpn image feature
        features = self.vis_model.backbone(images.tensor)
        image_features = self.vis_model.backbone.bottom_up(images.tensor)["res5"]

        with EventStorage() as storage:
            proposals, proposal_losses = self.vis_model.proposal_generator(images, features, gt_instances)
        
        with EventStorage() as storage:  
            box_features = [features[f] for f in self.vis_model.roi_heads.box_in_features]
            box_features = self.vis_model.roi_heads.box_pooler(box_features, [x.proposal_boxes for x in proposals])
            box_features = self.vis_model.roi_heads.box_head(box_features) # box features here
            box_predictions = self.vis_model.roi_heads.box_predictor(box_features)
            pred_instances, _ = self.vis_model.roi_heads.box_predictor.inference(box_predictions, proposals)


        with EventStorage() as storage:
            mask_features = [features[f] for f in self.vis_model.roi_heads.mask_in_features]
            boxes = [x.proposal_boxes if self.vis_model.training else x.pred_boxes for x in pred_instances]
            mask_features = self.vis_model.roi_heads.mask_pooler(mask_features, boxes) #mask features here

            # to get mask output
            pred_instances = self.vis_model.roi_heads.mask_head(mask_features, pred_instances)
            pred_instances = self.vis_model.roi_heads._forward_reachable(features, pred_instances)
            pred_instances = self.vis_model._postprocess(pred_instances, batched_images, images.image_sizes)

        return image_features, mask_features, pred_instances


class VisionEncoder(object):

    def __init__(self, args, eval=False, pretrained=False):
        self.model_type = args.vis_model
        self.gpu = args.gpu

        if self.model_type == "maskrcnn_reachable":
            self.model = MaskRCNN(args)
        elif self.model_type == "maskrcnn":
            self.model = MaskRCNN(args)
    
    def featurize(self, images, batch=32):
        '''
        Input: list of images as numpy array
        Output: image features, mask features, mask predictions, reachability predictions
        '''
        
        # if self.model_type == "maskrcnn_reachable":
        #     batched_input = [self.model.mapper({"image": img}) for img in images]
        batched_input = self.parse_inputs(images)

        image_feature, mask_features, preds = self.model.extract(batched_input)
        return image_feature, mask_features, preds
    
    def parse_inputs(self, images):
        batched_input = []
        for img in images:
            height, width = img.shape[:2]
            aug = T.ResizeShortestEdge(
                [self.model.cfg.INPUT.MIN_SIZE_TEST, self.model.cfg.INPUT.MIN_SIZE_TEST], self.model.cfg.INPUT.MAX_SIZE_TEST
            )
            image = aug.get_transform(img).apply_image(img)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            img_input = {"image": image, "height": height, "width": width}
            batched_input.append(img_input)
        
        return batched_input


def parse_object_reachable(output, id_cat_map, score_threshold=0.5, dist_threshold=0.5):
    '''
    Takes in prediction for batch and parses all objects and corresponding reachability prediction
    dist threshold is a cutoff for reachability prediction
    1: reachable
    0: unreachable
    '''
    parses = []
    for instance in output:
        objects_reachables = []
        pred = instance['instances']
        object_ids = pred.pred_classes.tolist()
        scores = pred.scores.tolist()
        reachables = pred.pred_reachables.tolist()
        for obj_id, score, dist in zip(object_ids, scores, reachables):
            obj_id = int(obj_id) + 1
            score = float(score)
            dist = float(dist)
            if score > score_threshold:
                if dist > dist_threshold:
                    objects_reachables.append((id_obj_map[obj_id], 1))
                else:
                    objects_reachables.append((id_obj_map[obj_id], 0))
        parses.append(objects_reachables)
    
    return parses

def parse_vision_features(image_features, mask_features, preds):
    '''
    Parses batched Detectron2 output
    '''

    # Reshape features to be [batch_size, feature_size]
    img_features = image_features.view(image_features.shape[0], -1)
    obj_features = mask_features.view(mask_features.shape[0], -1)

    obj_masks = []
    obj_seq_lens = []
    obj_cat_preds = []
    obj_dist_preds = []
    for instance in preds:
        pred = instance['instances']
        obj_masks.append(pred.pred_masks)
        obj_seq_lens.append(len(pred.pred_classes))
        obj_cat_preds.append(pred.pred_classes)
        obj_dist_preds.append(pred.pred_reachables)

    return img_features, obj_features, obj_masks, obj_seq_lens, obj_cat_preds, obj_dist_preds



if __name__ == '__main__':
    '''
    Debugging purpose only
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--vis_model', type=str, default='maskrcnn_reachable')
    parser.add_argument('--vis_model_path', type=str, default="../data/detectron/model_final_alfred_full.pth")
    parser.add_argument('--vis_model_config', type=str, default="vision/vision_config/alfred_reachable.yaml")
    parser.add_argument('--data_path', type=str, default="storage/data/alfred/json_feat_2.1.0")
    parser.add_argument('--eval-only', default=True, action='store_true')

    args = parser.parse_args()

    

    # Dummy data
    image_name1 = "../data/img_dummy/000000002.png"
    image_name2 = "../data/img_dummy/000000080.png"
    image_name3 = "../data/img_dummy/dummy_test.jpeg"
    image_name4 = "../data/img_dummy/000000099.png"
    image1 = utils.read_image(image_name1)
    image2 = utils.read_image(image_name2)
    image3 = utils.read_image(image_name3)
    image4 = utils.read_image(image_name4)


    # Open obj id to category name mapping
    import json
    with open("vision/vision_config/alfred_categories_mapping.json") as f:
        cat_map = json.load(f)
    id_obj_map = dict()
    for cat in cat_map:
        id_obj_map[cat['id']] = cat['name']
    
    # Run model and extract object-reachable predictions
    model = VisionEncoder(args)
    image_feature, mask_features, preds = model.featurize([image4])
    out = parse_object_reachable(preds, id_obj_map)
    print(preds)
    print(out)
    

    img_features, obj_features, obj_masks, obj_seq_len, obj_cat_pred, obj_dist_pred = parse_vision_features(image_feature, mask_features, preds)

    #print(img_features.shape, obj_features.shape, obj_masks, obj_seq_len, obj_cat_pred, obj_dist_pred)

    # Visualization code
    img1 = cv2.imread(image_name4)[:, :, ::-1]
    plt.imshow(img1)
    for box in preds[0]["instances"].pred_boxes:
        x1, y1, x2, y2 = box.cpu()
        w, h = x2 - x1, y2 - y1
        rec = Rectangle((x1, y1), width=w, height=h, fill=False, color="r", lw=2)
        plt.gca().add_patch(rec)
    plt.show()

    # img2 = cv2.imread(image_name2)[:, :, ::-1]
    # plt.imshow(img2)

    # for box in preds[1]["instances"].pred_boxes:
    #     x1, y1, x2, y2 = box.cpu()
    #     w, h = x2 - x1, y2 - y1
    #     rec = Rectangle((x1, y1), width=w, height=h, fill=False, color="r", lw=2)
    #     plt.gca().add_patch(rec)
    # plt.show()

    
