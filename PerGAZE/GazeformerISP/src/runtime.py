import sys
from itertools import islice
from tqdm import tqdm
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.dataset import PerGAZE, collate_func
from dataset.schema import make_file_manifest, read_records
from models.sampling import sample_scanpaths
from utils.evaluation import Metrics, summarize
from geometry import GEOMETRY
from parallel import resolve_devices, wrap_model

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def setup(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    if args.device == 'auto' and args.gpu_ids and torch.cuda.is_available():
        device = torch.device('cuda', args.gpu_ids[0])
    device, args.gpu_ids = resolve_devices(device, args.gpu_ids, torch.cuda.device_count())
    train_records = read_records(args.train_file)
    validation_records = read_records(args.val_file)
    records = train_records + validation_records
    manifest = make_file_manifest(train_records, validation_records)
    manifest['sources'] = {
        'train': {'path': str(args.train_file.resolve()), 'sha256': hashlib.sha256(args.train_file.read_bytes()).hexdigest()},
        'validation': {'path': str(args.val_file.resolve()), 'sha256': hashlib.sha256(args.val_file.read_bytes()).hexdigest()},
    }
    manifest['geometry'] = GEOMETRY.copy()
    from dataset.text import add_text_manifest
    add_text_manifest(records, manifest)
    if args.text_embeddings:
        with np.load(args.text_embeddings, allow_pickle=False) as archive:
            args.text_dim = archive['vectors'].shape[1]
        manifest['text_embeddings_sha256'] = hashlib.sha256(args.text_embeddings.read_bytes()).hexdigest()
    return (records, manifest, device)

def dataset(args, records, manifest, split):
    cls, extra = (PerGAZE, {})
    from dataset.text import GazeformerPerGAZE
    cls = GazeformerPerGAZE
    extra = {'text_embeddings': args.text_embeddings, 'max_text_length': args.max_text_length}
    return cls(records, manifest, args.img_dir, args.att_dir, split=split, max_length=args.max_length, blur_sigma=args.blur_sigma, **extra)

def loader(args, data, shuffle=False, evaluation=False):
    if not len(data):
        raise ValueError('The requested split is empty')
    return DataLoader(data, batch_size=args.test_batch if evaluation else args.batch, shuffle=shuffle, num_workers=args.workers, collate_fn=collate_func, pin_memory=torch.cuda.is_available())

def model(args, manifest, device, pretrained=None):
    from models.gazeformer.model import GazeformerISP
    network = GazeformerISP(args, manifest, pretrained=args.pretrained if pretrained is None else pretrained)
    return wrap_model(network, device, args.gpu_ids)

def move(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}

def forward(model, batch):
    return model(batch['images'], batch['subjects'], batch['attention_maps'], batch['task_tokens'], batch['task_mask'], batch.get('task_embeddings'))

def evaluate(model, batches, args, device, limit=0, description="Evaluation"):
    model.eval()
    metrics, rows = (Metrics(), [])
    total = min(len(batches), limit) if limit else len(batches)
    with torch.no_grad(), tqdm(total=total, desc=description, unit="batch",
                               dynamic_ncols=True, mininterval=1, file=sys.stdout) as progress:
        for batch in islice(batches, total):
            moved = move(batch, device)
            prediction = forward(model, moved)
            for repeat in range(args.eval_repeat_num):
                paths, *_ = sample_scanpaths(prediction, args.min_length, greedy=args.greedy)
                for info, target, path in zip(batch['metadata'], batch['fix_vectors'], paths):
                    rows.append({**info, 'repeat': repeat, 'scanpath': path.tolist(), 'metrics': metrics.pair(target, path)})
            progress.set_postfix(samples=len(rows), refresh=False)
            progress.update(1)
    return (summarize(rows), rows)

def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=True)

def restore_config(args, checkpoint, manifest):
    saved_manifest = checkpoint['manifest']
    if saved_manifest.get('geometry') != GEOMETRY:
        raise ValueError('Checkpoint uses a different image/scanpath geometry; retrain with the 1024x768 image and 512x352 scanpath configuration')
    if saved_manifest.get('split_mode') != 'explicit_files':
        raise ValueError('Checkpoint uses the previous automatic split; start a new run for train.json/test_seen.json')
    for split in ('train', 'validation'):
        if saved_manifest['sources'][split]['sha256'] != manifest['sources'][split]['sha256']:
            raise ValueError(f'Checkpoint {split} JSON differs from the supplied file')
    if checkpoint['config'].get('model', 'chenlstm') != args.model:
        raise ValueError('Checkpoint architecture does not match --model')
    for key in ('max_length', 'min_length', 'embedding_dim', 'action_map_num', 'dropout', 'blur_sigma'):
        setattr(args, key, checkpoint['config'][key])
    for key in ('hidden_dim', 'nhead', 'num_encoder', 'num_decoder', 'encoder_dropout', 'decoder_dropout', 'text_dim', 'max_text_length', 'train_backbone', 'backbone_weights'):
        setattr(args, key, checkpoint['config'][key])
    saved_path = checkpoint['config'].get('text_embeddings')
    if saved_path and args.text_embeddings is None:
        args.text_embeddings = Path(saved_path)
    if bool(saved_path) != bool(args.text_embeddings):
        raise ValueError('Checkpoint text backend does not match --text_embeddings')
    if args.text_embeddings and hashlib.sha256(args.text_embeddings.read_bytes()).hexdigest() != saved_manifest['text_embeddings_sha256']:
        raise ValueError('Task embedding archive differs from checkpoint')
    return saved_manifest
