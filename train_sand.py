from ultralytics import YOLO
import torch
from datetime import datetime

def get_datetime():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    device = 'cpu'
    if (torch.cuda.is_available()):
        device = 'cuda'
    
    print(f"Using device: {device}")
    
    model = YOLO('yolo11n.pt')
    model.to(device)
    
    data_path = "data.yaml"
    epochs = 5
    workers = 8
    batch_size = 16
    img_size = 640
    save_period = epochs // 2
    
    dt = get_datetime()
    
    res = model.train(
        data=data_path, 
        imgsz=img_size,
        epochs=epochs, 
        device=device,
        save_period=save_period,
        name=f'trained_model_{dt}',
        # project="detect"
        # batch=batch_size,
        # workers=workers,
        # amp=True,
    )
    
    
    
    
    
if __name__ == '__main__':
    main()