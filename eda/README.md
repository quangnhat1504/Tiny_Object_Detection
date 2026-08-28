# EDA - Dataset TinyPerson

Bo script phan tich tham do (EDA) cho dataset TinyPerson (YOLO polygon format,
export tu Roboflow). Sinh thong ke tong quan, bieu do truc quan va bao cao Markdown.

## Noi dung
- `eda_tinyperson.py` - script chinh: quet `data/`, parse nhan, tinh thong ke.
- `plots.py` - cac ham ve bieu do (matplotlib).
- `report.py` - sinh bao cao `REPORT.md` tu so lieu thuc te.
- `requirements.txt` - thu vien can them.

## Cach chay
```bash
# 1. Cai thu vien con thieu (chi can matplotlib)
.venv/Scripts/python.exe -m pip install matplotlib

# 2. Chay EDA
.venv/Scripts/python.exe eda/eda_tinyperson.py
```

## Ket qua sinh ra
- `eda/REPORT.md` - bao cao tong hop (kem bieu do).
- `eda/figures/*.png` - 9 bieu do truc quan.
- `eda/summary.json` - toan bo so lieu dang JSON.
- `eda/instances.csv` - 1 dong / object (de phan tich sau).
- `eda/images.csv` - 1 dong / anh.

## Ghi chu ky thuat
- Nhan o dinh dang **polygon** (`class x1 y1 x2 y2 ...`), duoc chuyen thanh
  bounding box (min/max cua cac dinh) de tinh kich thuoc.
- Kich thuoc object do bang `sqrt(dien tich bbox)` theo pixel tuyet doi,
  nguong tiny < 20px, small < 32px theo huong cua benchmark TinyPerson.
