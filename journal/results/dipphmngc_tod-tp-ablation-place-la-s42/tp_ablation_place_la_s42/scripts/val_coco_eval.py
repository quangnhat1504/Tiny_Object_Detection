"""COCO eval on validation — 1 model at a time, save to file."""
from pathlib import Path
import torch, json, sys
sys.path.insert(0, '.')
from common.config import DEVICE, seed_all, SEED
from common.dataset import YOLOTinyDataset, collate_fn
from common.metrics import get_metric_fn
from common.model import build_model
from common.eval_utils import evaluate
from torch.utils.data import DataLoader

seed_all(SEED)
vd = Path('data/valid')
ds = YOLOTinyDataset(img_dir=vd/'images', lbl_dir=vd/'labels', is_train=False)
vl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
print(f'Val: {len(ds)} tiles')

checkpoints = [
    ('NWD',           'runs/nwd__la_loss__seed42/best.pt',                       'nwd'),
    ('IGWD',          'runs/igwd__la_loss__seed42/best.pt',                      'igwd'),
    ('ALW',           'runs/alw_full__la_loss__seed42/best.pt',                  'alw_original'),
    ('SA-ALW full',   'runs/sa_alw_full__la_loss__seed42/best.pt',              'sa_alw_full'),
    ('SA-ALW beta',   'runs/sa_alw_beta_only__la_loss__seed42/best.pt',         'sa_alw_beta_only'),
    ('SA-ALW wpos',   'runs/sa_alw_pos_only__la_loss__seed42/best.pt',          'sa_alw_pos_only'),
    ('IGWD+aniso',    'runs/igwd_anisotropic_s__la_loss__seed42/best.pt',       'igwd_anisotropic_s'),
    ('IGWD+log',      'runs/igwd_log_shape__la_loss__seed42/best.pt',            'igwd_log_shape'),
    ('IoU full',      'runs/frcnn_standard__full__seed42/best.pt',              'iou'),
    ('IoU patches',   'runs/frcnn_standard__patches__seed42/best.pt',           'iou'),
]

out = {}
for label, cp_str, mn in checkpoints:
    cp = Path(cp_str)
    if not cp.exists():
        print(f'{label}: SKIP (no ckpt)')
        continue
    print(f'{label}...', end=' ', flush=True)
    ck = torch.load(cp, map_location='cpu', weights_only=False)
    mfn = None if mn == 'iou' else get_metric_fn(mn)
    pl = 'everywhere' if mn == 'iou' else 'la_loss'
    model = build_model(metric_fn=mfn, placement=pl).to(DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    r = evaluate(model, vl, DEVICE, measure_fps_flag=False)
    row = {
        'AP': round(r.get('coco_AP', 0), 4),
        'AP50': round(r.get('coco_AP50', 0), 4),
        'AP75': round(r.get('coco_AP75', 0), 4),
        'AP_S': round(r.get('coco_AP_small', 0), 4),
        'AP_M': round(r.get('coco_AP_medium', 0), 4),
        'AP_L': round(r.get('coco_AP_large', 0), 4),
        'AR100': round(r.get('coco_AR100', 0), 4),
    }
    out[label] = row
    print(f'AP={row["AP"]} AP50={row["AP50"]} AP75={row["AP75"]} AP_S={row["AP_S"]} AP_M={row["AP_M"]} AP_L={row["AP_L"]} AR100={row["AR100"]}')
    del model; torch.cuda.empty_cache()

with open('runs/val_coco.json', 'w') as f:
    json.dump(out, f, indent=2)
print('\nDone! -> runs/val_coco.json')
