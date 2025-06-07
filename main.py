import time
import wandb
from ultralytics import YOLO

if __name__ == '__main__':
    project_name = "OBB-EC"
    name = time.strftime("%Y%m%d-%H%M%S")
    model = "./ultralytics/cfg/models/v8/yolov8n-obb-EC.yaml"
    data = "./ultralytics/cfg/datasets/dota8-EC.yaml"

    model = YOLO(model, task="obb")

    # Train the model
    train_results = model.train(
        project=project_name,
        name=name,
        data=data,  # path to dataset YAML
        imgsz=640,  # training image size
        device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )
