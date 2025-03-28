from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect.train import DetectionTrainer

def main():
    cfg = get_cfg("/Users/heejulee/Desktop/FLIR/Yolo_CBAM/ultralytics/ultralytics/cfg/fusion.yaml")  # 위에서 만든 설정 파일
    trainer = DetectionTrainer(overrides=cfg)
    trainer.train()

if __name__ == "__main__":
    main()