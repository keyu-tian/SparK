#!/usr/bin/python3

# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import json
import logging
import os
import time
from collections import OrderedDict, defaultdict
from functools import partial
from pprint import pformat

import numpy as np
import torch
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch, PeriodicWriter
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.layers import get_norm
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.events import EventWriter

from lr_decay import get_default_optimizer_params, lr_factor_func


# [modification] for better logging
def _ex_repr(self):
    d = vars(self)
    ex = ', '.join(f'{k}={v}' for k, v in d.items() if not k.startswith('__') and k not in [
        'trainer', 'before_train', 'after_train', 'before_step', 'after_step', 'state_dict',
        '_model', '_data_loader', 'logger',
    ])
    return f'{type(self).__name__}({ex})'
hooks.HookBase.__repr__ = _ex_repr
EventWriter.__repr__ = _ex_repr


# [modification] add norm
@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    """
    As described in the MOCO paper, there is an extra BN layer
    following the res5 stage.
    """
    
    def _build_res5_block(self, cfg):
        seq, out_channels = super()._build_res5_block(cfg)
        norm = cfg.MODEL.RESNETS.NORM
        norm = get_norm(norm, out_channels)
        seq.add_module("norm", norm)
        return seq, out_channels


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """
    
    # [modification] override the `build_optimizer` for using Adam and layer-wise lr decay
    lr_decay_ratio: float = 1.0
    @classmethod
    def build_optimizer(cls, cfg, model):
        is_resnet50 = int(cfg.MODEL.RESNETS.DEPTH) == 50
        if comm.is_main_process():
            dbg = defaultdict(list)
            for module_name, module in model.named_modules():
                for module_param_name, value in module.named_parameters(recurse=False):
                    if not value.requires_grad:
                        continue
                    lrf = lr_factor_func(f"{module_name}.{module_param_name}", is_resnet50=is_resnet50, dec=cls.lr_decay_ratio, debug=True)
                    dbg[lrf].append(f"{module_name}.{module_param_name}")
            for k in sorted(dbg.keys()):
                print(f'[{k}] {sorted(dbg[k])}')
            print()
        
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            lr_factor_func=partial(lr_factor_func, is_resnet50=is_resnet50, dec=cls.lr_decay_ratio, debug=False)
        )
        
        opt_clz = {
            'sgd': partial(torch.optim.SGD, momentum=cfg.SOLVER.MOMENTUM, nesterov=cfg.SOLVER.NESTEROV),
            'adamw': torch.optim.AdamW,
            'adam': torch.optim.AdamW,
        }[cfg.SOLVER.OPTIMIZER.lower()]
        return maybe_add_gradient_clipping(cfg, opt_clz)(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)
    
    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # [modification] we add these two new keys
    cfg.SOLVER.OPTIMIZER, cfg.SOLVER.LR_DECAY = 'sgd', 1.0  # by default using SGD and no lr_decay
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # [modification] for implementing lr decay and for logging
    Trainer.lr_decay_ratio = cfg.SOLVER.LR_DECAY
    
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    
    # [modification] just skip some warnings
    import warnings
    comm.synchronize()
    warnings.filterwarnings('ignore', category=UserWarning)
    _ = np.arange(3, dtype=np.int).astype(np.bool)
    _ = np.array(torch.ones(3, dtype=torch.int32).numpy(), dtype=np.int)
    _ = np.array(torch.ones(3, dtype=torch.int64).numpy(), dtype=np.int)
    _ = np.array(torch.ones(3, dtype=torch.long).numpy(), dtype=np.int)
    _ = torch.rand(100) // 5
    _ = torch.meshgrid(torch.ones(1))
    warnings.resetwarnings()
    comm.synchronize()
    
    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    for h in trainer._hooks:
        if isinstance(h, PeriodicWriter):
            h._period = 1000  # [modification] less logging
    
    # [modification] we add some hooks for logging
    is_local_master = comm.get_rank() % args.num_gpus == 0
    if comm.is_main_process():
        print(f'[default hooks] {pformat(trainer._hooks, indent=2, width=300)}')
    ex_hooks = [
        hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model)) if cfg.TEST.AUG.ENABLED else None,
        LogHook(cfg.TEST.EVAL_PERIOD, args.config_file, cfg.OUTPUT_DIR, is_local_master) if comm.is_main_process() else None,
    ]
    trainer.register_hooks(ex_hooks)
    if comm.is_main_process():
        print(f'[extra hooks] {pformat(ex_hooks, indent=2, width=300)}')
    
    return trainer.train()


# [modification] we add a hook for logging results to `cfg.OUTPUT_DIR/d2_coco_log.txt`
class LogHook(hooks.HookBase):
    def __init__(self, eval_period, config_file, output_dir, is_local_master):
        self.eval_period = eval_period
        self.log_period = eval_period // 4
        self.log = {}
        
        self.is_master = comm.is_main_process()
        self.is_local_master = is_local_master
        
        self.config_file = config_file
        self.out_dir = output_dir
        self.log_txt_name = os.path.join(self.out_dir, 'd2_coco_log.txt')
    
    def __write_to_log_file(self, d):
        if self.is_local_master:
            self.log.update(d)
            with open(self.log_txt_name, 'w') as fp:
                json.dump(self.log, fp)
                fp.write('\n')
    
    def update_and_write_to_local_log(self):
        stat = self.trainer.storage.latest()
        self.log['boxAP'], self.log['bAP50'], self.log['bAP75'] = stat['bbox/AP'][0], stat['bbox/AP50'][0], stat['bbox/AP75'][0]
        self.log['mskAP'], self.log['mAP50'], self.log['mAP75'] = stat['segm/AP'][0], stat['segm/AP50'][0], stat['segm/AP75'][0]
        self.log['bAP-l'], self.log['bAP-m'], self.log['bAP-s'] = stat['bbox/APl'][0], stat['bbox/APm'][0], stat['bbox/APs'][0]
        self.log['mAP-l'], self.log['mAP-m'], self.log['mAP-s'] = stat['segm/APl'][0], stat['segm/APm'][0], stat['segm/APs'][0]
        all_ap = sorted([(v[0], k.split('AP-')[-1].strip()) for k, v in stat.items() if k.startswith('bbox/AP-')])
        all_ap = [tu[1] for tu in all_ap]
        self.log['easy'] = ' | '.join(all_ap[-7:])
        self.log['hard'] = ' | '.join(all_ap[:7])
        for k in self.log.keys():
            if 'AP' in k:
                self.log[k] = round(self.log[k], 3)
        self.__write_to_log_file({})
    
    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self.eval_period > 0 and next_iter % self.eval_period == 0:
            self.update_and_write_to_local_log()
        
        if self.log_period > 0 and next_iter % self.log_period == 0:
            stat = self.trainer.storage.latest()
            remain_secs = round(stat['eta_seconds'][0])
            d = {
                'cfg': self.config_file,
                'rema': str(datetime.timedelta(seconds=remain_secs)), 'fini': time.strftime("%m-%d %H:%M", time.localtime(time.time() + remain_secs)),
                'cur_iter': f'{next_iter}/{self.trainer.max_iter}',
            }
            self.__write_to_log_file(d)
    
    def after_train(self):
        self.update_and_write_to_local_log()
        last_boxAP, last_mskAP = round(self.log['boxAP'], 3), round(self.log['mskAP'], 3)
        self.__write_to_log_file({
            'rema': '-', 'fini': time.strftime("%m-%d %H:%M", time.localtime(time.time() - 120)),
            'last_boxAP': last_boxAP,
            'last_mskAP': last_mskAP,
        })
        time.sleep(5)
        if self.is_master:
            print(f'\n[finished] ========== last_boxAP={last_boxAP}, last_mskAP={last_mskAP} ==========\n')


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)
