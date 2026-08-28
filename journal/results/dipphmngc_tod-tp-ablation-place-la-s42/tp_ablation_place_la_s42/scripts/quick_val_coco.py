"""Quick COCO eval on validation set for all checkpoints."""
from pathlib import Path
import torch, json, sys
sys.path.insert(0, '.')

from common.config import DEVICE, seed_all, SEED
from common.dataset import YOLOTinyDataset, collate_fn
from common.metrics import get_metric_fn
from common.model import build_model
from common.eval_utils import evaluate
from torch.utils.data import DataLoader

def main():
    seed_all(SEED)
    print(f'Device: {DEVICE}')

    vd = Path('data/valid')
    ds = YOLOTinyDataset(img_dir=vd/'images', lbl_dir=vd/'labels', is_train=False)
    val_loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f'Val tiles: {len(ds)}')

    checkpoints = [
        ('NWD',     'runs/nwd__la_loss__seed42/best.pt',         'nwd'),
        ('IGWD',    'runs/igwd__la_loss__seed42/best.pt',        'igwd'),
        ('ALW',     'runs/alw_full__la_loss__seed42/best.pt',    'alw_original'),
        ('SA-ALW full',     'runs/sa_alw_full__la_loss__seed42/best.pt',     'sa_alw_full'),
        ('SA-ALW beta',     'runs/sa_alw_beta_only__la_loss__seed42/best.pt', 'sa_alw_beta_only'),
        ('SA-ALW wpos',     'runs/sa_alw_pos_only__la_loss__seed42/best.pt',  'sa_alw_pos_only'),
        ('IGWD+aniso', 'runs/igwd_anisotropic_s__la_loss__seed42/best.pt', 'igwd_anisotropic_s'),
        ('IGWD+log',   'runs/igwd_log_shape__la_loss__seed42/best.pt',       'igwd_log_shape'),
        ('FRCNN full', 'runs/frcnn_standard__full__seed42/best.pt',           'iou'),
        ('FRCNN patch','runs/frcnn_standard__patches__seed42/best.pt',        'iou'),
    ]

    results = {}
    for label, ckpt_path_str, metric_name in checkpoints:
        ckpt_path = Path(ckpt_path_str)
        if not ckpt_path.exists():
            print(f'SKIP {label}: no checkpoint')
            continue

        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        ep = ck.get('epoch', '?')
        cfg = ck.get('config', {})
        metric_name = cfg.get('metric', metric_name)

        metric_fn = None if metric_name == 'iou' else get_metric_fn(metric_name)
        placement = 'everywhere' if metric_name == 'iou' else cfg.get('placement', 'la_loss')
        model = build_model(
            metric_fn=metric_fn,
            placement=placement,
            reliability_thr=cfg.get('reliability_thr', 16.0),
            box_loss_type=cfg.get('box_loss', 'metric'),
            use_quality_score=bool(cfg.get('quality_score', False)),
            quality_loss_weight=float(cfg.get('quality_loss_weight', 0.0) or 0.0),
            use_quality_focal=bool(cfg.get('quality_focal', False)),
            quality_focal_beta=float(cfg.get('quality_focal_beta', 2.0)),
            use_rank_sort=bool(cfg.get('rank_sort', False)),
            rank_sort_delta=float(cfg.get('rank_sort_delta', 0.5)),
            use_double_head=bool(cfg.get('double_head', False)),
            double_head_reg_roi_scale=float(
                cfg.get('double_head_reg_roi_scale', 1.3)),
            double_head_num_convs=int(cfg.get('double_head_num_convs', 4)),
            cbl_alpha=float(cfg.get('cbl_alpha', 5.0)),
            cbl_num_bins=int(cfg.get('cbl_num_bins', 6)),
            cbl_grid_beta=float(cfg.get('cbl_grid_beta', 1.0)),
            cbl_um_weight=float(cfg.get('cbl_um_weight', 1.0)),
        ).to(DEVICE)
        model.load_state_dict(ck['model'])
        model.eval()

        metrics = evaluate(model, val_loader, DEVICE, measure_fps_flag=False)

        row = {
            'label': label, 'epoch': ep,
            'AP': round(metrics.get('coco_AP', 0), 4),
            'AP50': round(metrics.get('coco_AP50', 0), 4),
            'AP75': round(metrics.get('coco_AP75', 0), 4),
            'AP_S': round(metrics.get('coco_AP_small', 0), 4),
            'AP_M': round(metrics.get('coco_AP_medium', 0), 4),
            'AP_L': round(metrics.get('coco_AP_large', 0), 4),
            'AR100': round(metrics.get('coco_AR100', 0), 4),
        }
        results[label] = row
        print(f'{label:<16} ep={ep} AP={row["AP"]:.4f} AP50={row["AP50"]:.4f} AP75={row["AP75"]:.4f} AP_S={row["AP_S"]:.4f} AP_M={row["AP_M"]:.4f} AP_L={row["AP_L"]:.4f} AR100={row["AR100"]:.4f}')

    with open('runs/val_coco_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to runs/val_coco_results.json')

if __name__ == '__main__':
    main()
