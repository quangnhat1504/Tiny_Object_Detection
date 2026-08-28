from PIL import Image
from pathlib import Path
from ultralytics import YOLO

img = Image.open('data/test/images/bb_V0005_I0001640_jpg.rf.012502ae69b3c26f7fea0a4181dd7485.jpg')
print('Size:', img.size)

yolo = YOLO('runs/yolo_baseline/yolov8n/weights/best.pt')
r = yolo(img, conf=0.15, iou=0.45, verbose=False)
if r[0].boxes and len(r[0].boxes) > 0:
    boxes = r[0].boxes.xyxy
    scores = r[0].boxes.conf
    labels = r[0].boxes.cls.long()
    print(f'YOLO detections: {len(boxes)} boxes')
    print(f'Score range: min={scores.min():.3f} max={scores.max():.3f}')
    grey_zone = ((scores >= 0.20) & (scores <= 0.55)).sum().item()
    print(f'In grey zone [0.20,0.55]: {grey_zone}')
    
    areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
    tiny = (areas < 256).sum().item()
    print(f'Tiny boxes (<256 px^2): {tiny}')
else:
    print('YOLO: NO DETECTIONS')
