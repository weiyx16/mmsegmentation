# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from ..diffdist.functional import all_gather
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy
from ...utils import get_root_logger

import random
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel

def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    out_list = all_gather(out_list, x)  # if use dist.all_gather, you can't get gradient over each

    return torch.cat(out_list, dim=0)

def dist_collect_list(x, device):
    x = torch.tensor(x).to(device)
    x = dist_collect(x)
    return x.cpu().tolist()

def varsize_dist_collect(tensor: torch.Tensor):
    # ref: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/comm.py
    # ref: https://discuss.pytorch.org/t/how-to-concatenate-different-size-tensors-from-distributed-processes/44819/4
    tensor = tensor.contiguous()

    size_tens = dist_collect_list([tensor.shape[0]], tensor.device)
    max_size = max(size_tens)

    padded = torch.empty(max_size, *tensor.shape[1:],
                         dtype=tensor.dtype,
                         device=tensor.device)
    padded[:tensor.shape[0]] = tensor

    ag = dist_collect(padded)

    slices = []
    for i, sz in enumerate(size_tens):
        start_idx = i * max_size
        end_idx = start_idx + sz

        if end_idx > start_idx:
            slices.append(ag[start_idx:end_idx])

    ret = torch.cat(slices, dim=0)

    return ret.to(tensor)

class LModel(nn.Module):
    def __init__(self):
        super(LModel, self).__init__()
        # https://huggingface.co/roberta-base; config: https://huggingface.co/roberta-base/resolve/main/config.json
        # downloaded to /root/.cache/huggingface/transformers
        configuration = RobertaConfig.from_pretrained('roberta-base')
        configuration = configuration.__dict__.copy()
        configuration.update({'return_dict': False})
        configuration.update({'gradient_checkpointing': False})
        configuration.pop('model_type')
        configuration = RobertaConfig(**configuration)
        self.backbone = RobertaModel.from_pretrained('roberta-base', config=configuration, add_pooling_layer=False)     
        hidden_size, proj_size = 768, 256
        self.projector = nn.Linear(hidden_size, proj_size, bias=False)

    def _output_avg_pool(self, sequence_output, attention_mask):
        '''
        # This version will take padding part into calculation
        # [bs, h]
        # sequence_output_txt = F.adaptive_avg_pool1d(sequence_output_txt.transpose(1,2), 1).transpose(1,2)
        # sequence_output_img = F.adaptive_avg_pool1d(sequence_output_img.transpose(1,2), 1).transpose(1,2)
        # mask format: [1: attend / 0: ignore]
        '''
        # [bs, 1, 1]
        seq_len = attention_mask.squeeze().sum(-1, keepdim=True).unsqueeze(-1)
        # [bs, sq_len, 1]
        attention_mask = attention_mask.squeeze().unsqueeze(-1)
        # [bs, 1, h]
        pooled_output = (sequence_output * attention_mask).sum(1, keepdim=True) / seq_len
        return pooled_output.squeeze()

    def forward(self, sentence):
        latents = self.backbone(**sentence, return_dict=False)[0]
        latents = self._output_avg_pool(latents, sentence['attention_mask'])
        latents = F.linear(input=latents.float(),
                weight=self.projector.weight.float(),
            )            
        return latents

class LanDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for LanDecodeHead.
    The key defference from BaseDecodeHead is using language model as classifier.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict | Sequence[dict]): Config of decode loss.
            The `loss_name` is property of corresponding loss function which
            could be shown in training log. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
             e.g. dict(type='CrossEntropyLoss'),
             [dict(type='CrossEntropyLoss', loss_name='loss_ce'),
              dict(type='DiceLoss', loss_name='loss_dice')]
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255.
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super(LanDecodeHead, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        # self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.language_model = LModel()
        self.visual_proj = nn.Linear(channels, 256, bias=False) if channels != 256 else nn.Identity()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1. / 0.05)), requires_grad=False)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.templates = [  'a bad photo of a {}.',  'a photo of many {}.',  'a sculpture of a {}.',  'a photo of the hard to see {}.',  'a low resolution photo of the {}.',  'a rendering of a {}.',  'graffiti of a {}.',  'a bad photo of the {}.',  'a cropped photo of the {}.',  'a tattoo of a {}.',  'the embroidered {}.',  'a photo of a hard to see {}.',  'a bright photo of a {}.',  'a photo of a clean {}.',  'a photo of a dirty {}.',  'a dark photo of the {}.',  'a drawing of a {}.',  'a photo of my {}.',  'the plastic {}.',  'a photo of the cool {}.',  'a close-up photo of a {}.',  'a black and white photo of the {}.',  'a painting of the {}.',  'a painting of a {}.',  'a pixelated photo of the {}.',  'a sculpture of the {}.',  'a bright photo of the {}.',  'a cropped photo of a {}.',  'a plastic {}.',  'a photo of the dirty {}.',  'a jpeg corrupted photo of a {}.',  'a blurry photo of the {}.',  'a photo of the {}.',  'a good photo of the {}.',  'a rendering of the {}.',  'a {} in a video game.',  'a photo of one {}.',  'a doodle of a {}.',  'a close-up photo of the {}.',  'a photo of a {}.',  'the origami {}.',  'the {} in a video game.',  'a sketch of a {}.',  'a doodle of the {}.',  'a origami {}.',  'a low resolution photo of a {}.',  'the toy {}.',  'a rendition of the {}.',  'a photo of the clean {}.',  'a photo of a large {}.',  'a rendition of a {}.',  'a photo of a nice {}.',  'a photo of a weird {}.',  'a blurry photo of a {}.',  'a cartoon {}.',  'art of a {}.',  'a sketch of the {}.',  'a embroidered {}.',  'a pixelated photo of a {}.',  'itap of the {}.',  'a jpeg corrupted photo of the {}.',  'a good photo of a {}.',  'a plushie {}.',  'a photo of the nice {}.',  'a photo of the small {}.',  'a photo of the weird {}.',  'the cartoon {}.',  'art of the {}.',  'a drawing of the {}.',  'a photo of the large {}.',  'a black and white photo of a {}.',  'the plushie {}.',  'a dark photo of a {}.',  'itap of a {}.',  'graffiti of the {}.',  'a toy {}.',  'itap of my {}.',  'a photo of a cool {}.',  'a photo of a small {}.',  'a tattoo of the {}.']
        # https://gist.github.com/willprice/f19da185c9c5f32847134b87c1960769
        if num_classes == 150:
            # ade20k
            self.classnames = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag']
        elif num_classes == 21:
            # voc12
            self.classnames = ['background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'sedan', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']
        elif num_classes == 19:
            # cityscapes
            self.classnames = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
        global_rank = int(dist.get_rank())
        total_gpu = int(dist.get_world_size())
        per_gpu_len = len(self.classnames) // total_gpu + 1
        self.full_classnames = self.classnames
        self.classnames = self.classnames[global_rank * per_gpu_len : min((global_rank+1) * per_gpu_len, len(self.classnames))]

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self, pretrained):
        """Initiate the parameters from scratch."""
        ### LOAD Pretrain Weight
        if pretrained is not None:
            self.pretrained = pretrained
            print(" Successfully initial the head !!! ")
            assert isinstance(self.pretrained, str), 'give path to pretrained vl model'
            logger = get_root_logger()
            logger.info(f'load language head from: {self.pretrained}')
            checkpoint = torch.load(self.pretrained, map_location='cpu')
            state_dict = checkpoint['model']

            new_state_dict = {}
            new_state_dict_proj = {}
            for k, v in state_dict.items():
                if k.startswith('sentence_model.'):
                    new_state_dict[k.replace('sentence_model.', '')] = v.cpu()
                if k.startswith('visual_model.projector.'):
                    new_state_dict_proj[k.replace('visual_model.projector.', '')] = v.cpu()
            state_dict = new_state_dict
            msg = self.language_model.load_state_dict(state_dict, strict=False)
            # self.visual_proj.load_state_dict(new_state_dict_proj, strict=True)

            logger.info(msg)
            logger.info(f"=> loaded successfully '{self.pretrained} to head'")
            del checkpoint
            torch.cuda.empty_cache()

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.is_test = False
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        self.is_test = True
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            # bs * channel * size * size
            feat = self.dropout(feat)
        # output = self.conv_seg(feat)
        bs, channel, h, w = feat.size()
        feat = feat.permute(0,2,3,1).contiguous().view(bs, -1, channel)
        # N * pixels * channel
        feat = self.visual_proj(feat)
        ## inference class weight:
        if self.is_test:
            if hasattr(self, 'class_head_weight_gathered_prompt'):
                class_head_weight_gathered = self.class_head_weight_gathered_prompt
            else:
                print(" Inference Ensembled Feature with language head. ")
                zeroshot_weights = []
                for idx, classname in enumerate(self.full_classnames):
                    texts = []
                    for template in self.templates:
                        _texts = template.format(classname)
                        texts.append(_texts)
                    prompted_imagenet_classhead_input = self.tokenizer(texts, padding=True, truncation=True, max_length=16, return_tensors='pt')
                    prompted_imagenet_classhead_input = {k:v.to(feat.device, non_blocking=True) for k, v in prompted_imagenet_classhead_input.items()}
                    with torch.no_grad():
                        class_head_weight = self.language_model(prompted_imagenet_classhead_input)
                        class_head_weight /= class_head_weight.norm(dim=-1, keepdim=True)
                        class_embedding = class_head_weight.mean(dim=0)
                        class_embedding /= class_embedding.norm()
                    zeroshot_weights.append(class_embedding)
                zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(feat.device).t()
                self.class_head_weight_gathered_prompt = zeroshot_weights
                class_head_weight_gathered = self.class_head_weight_gathered_prompt
        else:
            sentence = []
            for sent in self.classnames:
                prompt = random.choice(self.templates)
                sentence.append(prompt.format(sent))
            prompted_imagenet_classhead_input = self.tokenizer(sentence, padding=True, truncation=True, max_length=16, return_tensors='pt')
            prompted_imagenet_classhead_input = {k:v.to(feat.device, non_blocking=True) for k, v in prompted_imagenet_classhead_input.items()}
            # with torch.no_grad():
            class_head_weight = self.language_model(prompted_imagenet_classhead_input)
            # class_head_weight_gathered = SyncFunction.apply(class_head_weight) 
            class_head_weight_gathered = varsize_dist_collect(class_head_weight)
        
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        feat, class_head_weight_gathered, = \
            map(lambda t: F.normalize(t, p = 2, dim = -1) if t is not None else t, (feat, class_head_weight_gathered))
        # [N, pixels, num_classes]
        output = logit_scale * feat @ class_head_weight_gathered.t()
        output = output.permute(0,2,1).contiguous().view(bs, -1, h, w)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss
