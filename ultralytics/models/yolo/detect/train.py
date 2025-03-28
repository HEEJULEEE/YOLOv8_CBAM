# Ultralytics 🚀 AGPL-3.0 License
import math
import random
from copy import copy
import torch

import numpy as np
import torch.nn as nn

from ultralytics.data import build_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.data.dataset import FusionDataset  # 커스텀 FusionDataset import

torch.cuda.empty_cache()

class DetectionTrainer(BaseTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        """
        img_path: RGB image directory
        self.args.thermal_path: Thermal image directory (추가로 self.args에서 받아옴)
        self.args.weight_csv: per-image weight csv (추가)
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return FusionDataset(
            img_path=img_path,
            thermal_path=self.args.thermal_path,
            weight_csv=self.args.weight_csv,
            imgsz=self.args.imgsz,
            augment=mode == "train",
            half = self.args.half,
            hyp=self.args,
            rect=mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("rect=True and shuffle=True are incompatible. Setting shuffle=False.")
            shuffle = False

        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def preprocess_batch(self, batch):
        """ Normalize and send to device. """
        for key in ["img_rgb", "img_thermal"]:
            if key in batch:
                batch[key] = batch[key].to(self.device, non_blocking=True)
        return batch

    def set_model_attributes(self):
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg='/home/heeju064/Yolo_CBAM/ultralytics/ultralytics/cfg/models/v8/yolov8_fusion.yaml', nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/box", f"{prefix}/cls", f"{prefix}/dfl"]
        if isinstance(loss_items, (list, tuple, torch.Tensor)) and len(loss_items) == 3:
            return dict(zip(keys, [round(float(x), 5) for x in loss_items]))
        else:
            return {k: 0.0 for k in keys}  # or return keys

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % ("Epoch", "GPU_mem", *self.loss_names, "Instances", "Size")

    def plot_training_samples(self, batch, ni):
        plot_images(
            images=batch["img_rgb"],  # RGB 이미지만 시각화
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=["fusion"] * len(batch["batch_idx"]),
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
        
    def plot_metrics(self): 
        plot_results(file=self.csv, on_plot=self.on_plot)

    def plot_training_labels(self):
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)
        
    def train_step(self, batch):
        batch = self.preprocess_batch(batch)
        preds = self.model(batch)  # batch dict 그대로 전달
        loss, loss_items = self.criterion(preds, batch)
        return loss, loss_items

def train(cfg):
    from ultralytics.cfg import get_cfg
    cfg = get_cfg('/home/heeju064/Yolo_CBAM/ultralytics/ultralytics/cfg/models/v8/yolov8_fusion.yaml')
    cfg['fusion'] = True
    
    cfg.thermal_path = '/home/heeju064/Yolo_CBAM/ultralytics/ultralytics/data/thermal/images'
    cfg.weight_csv = '/home/heeju064/Yolo_CBAM/ultralytics/ultralytics/data/fire_prompt_hj_results.csv'
    trainer = DetectionTrainer(overrides=cfg)
    trainer.train()


if __name__ == "__main__":
    train("/home/heeju064/Yolo_CBAM/ultralytics/ultralytics/cfg/fusion.yaml")  # ← 여기에 yaml 경로 넣기
