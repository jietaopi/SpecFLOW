"""
Stage 1：Encoder + WB Fuser + Decoder 联合训练
用法: python stage1.py config.yaml --logdir ./logs/stage1
"""

import argparse
import csv
import os
import shutil
from glob import glob

import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from easydict import EasyDict
from tqdm import tqdm

from data_provider.stage1_dm import Stage1DM
from models.model import ADMCModel
from utils.misc import *
from utils.common import sample_known_mask, sample_known_mask_test

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 光谱补全训练")
    parser.add_argument("config", type=str, help="yaml 配置文件路径，或断点续训时的 log 目录")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--logdir", type=str, default="./logs/stage1")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume_iter", type=int, default=None, help="断点续训时指定从哪个 iter 的 ckpt 恢复，None 则取 latest")
    parser.add_argument("--save_to_csv", type=str, default=None, help="test 模式下将预测光谱追加写入的 csv 路径")
    parser.add_argument("--missing_modalities", type=str, default="1", help="test 模式下指定缺失模态的下标，逗号分隔，如 '0,2'")
    parser.add_argument("--use_diffusion_infer", action="store_true", help="test 模式下使用 Flow Matching 采样")
    return parser.parse_args()

def main(args):
    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, "*.yaml"))[0]
        resume_from = os.path.realpath(args.config)
    else:
        config_path = os.path.realpath(args.config)
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    with open(config_path, "r", encoding="utf-8") as f:
        config = EasyDict(yaml.safe_load(f))
    stage = int(config.stage)
    train_modules = list(config.train_modules)
    log_keys = list(config.log_keys)
    seed_all(config.seed)

    if resume:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag="resume")
        os.symlink(os.path.realpath(resume_from), os.path.join(log_dir, os.path.basename(resume_from.rstrip("/"))))
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        src_models_dir = os.path.join(os.path.dirname(__file__), "models")
        shutil.copytree(src_models_dir, os.path.join(log_dir, "models"))

    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger  = get_logger("train", log_dir)
    writer  = SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    model = ADMCModel(config.model, config.modality_names, config.spec_len, args.device)
    model.set_stage(stage)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[Stage {stage}] 参数: 冻结={frozen:,}  可训练={trainable:,}")
    model.to(args.device)

    logger.info("Loading datasets...")
    dm           = Stage1DM(config)
    train_loader = dm.train_loader()
    train_iter   = iter(train_loader)
    val_loader   = dm.val_loader()

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)

    start_iter = 1
    best_val_loss = float("inf")
    if resume:
        ckpt_path, start_iter = get_checkpoint_path(os.path.join(resume_from, "checkpoints"), iteration=args.resume_iter)
        logger.info(f"[Stage {stage}] Resuming from {ckpt_path}, iter={start_iter}")
        ckpt = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("avg_val_loss", float("inf")))

    kl_warmup_steps = getattr(config, "kl_warmup_steps", 1000)
    kl_weight_max   = config.model.loss_weight.kl_weight

    def train(current_step):
        nonlocal train_iter
        model.eval()
        for mod_name in train_modules:
            getattr(model, mod_name).train()

        train_iter, batch = next_batch(train_iter, train_loader, args.device)
        batch_size = batch["enc_batch"][config.modality_names[0]].size(0)
        known_mask = sample_known_mask(
            batch_size=batch_size, num_modalities=len(config.modality_names),
            drop_prob=config.drop_prob, device=args.device
        )
        kl_weight = min(current_step / kl_warmup_steps, 1.0) * kl_weight_max

        out = model(batch, known_mask, kl_weight=kl_weight, use_diffusion=False, use_decoder=True)

        optim.zero_grad(set_to_none=True)
        out["loss"].backward()
        optim.step()

        if current_step % config.log_freq == 0:
            loss_str = "  ".join(f"{k}={out[k].item():.5f}" for k in log_keys if k in out)
            cos_str = f"  cos={out['cos_sims'].tolist()}" if "cos_sims" in out else ""
            logger.info(f"[stage{stage} train] Iter {current_step:06d}/{config.max_iters} " + loss_str + cos_str)

        for k in log_keys:
            if k in out: writer.add_scalar(f"train/{k}", out[k].item(), current_step)
        if "cos_sims" in out:
            cos_vals = out["cos_sims"].detach().cpu().float()
            for m_idx, modality in enumerate(config.modality_names):
                if m_idx < cos_vals.numel(): writer.add_scalar(f"train/cos_sim/{modality}", cos_vals[m_idx].item(), current_step)
        writer.flush()

    def validate(current_step):
        sums = {k: 0.0 for k in log_keys}
        sum_cos = [0.0] * len(config.modality_names)
        n = 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Stage {stage}] Val"):
                batch = move_to_device(batch, args.device)
                batch_size = batch["enc_batch"][config.modality_names[0]].size(0)
                known_mask = sample_known_mask(
                    batch_size=batch_size, num_modalities=len(config.modality_names),
                    drop_prob=config.drop_prob, device=args.device
                )
                out = model(batch, known_mask, use_diffusion=False, use_decoder=True)
                for k in log_keys:
                    if k in out: sums[k] += out[k].item()
                if "cos_sims" in out:
                    for idx, v in enumerate(out["cos_sims"].tolist()): sum_cos[idx] += v
                n += 1

        avgs = {k: sums[k] / n for k in log_keys}
        avg_cos = [v / n for v in sum_cos]
        loss_str = "  ".join(f"{k}={avgs[k]:.5f}" for k in log_keys)
        logger.info(f"[stage{stage} val]   Iter {current_step:06d}/{config.max_iters} " + loss_str + (f"  cos={avg_cos}" if "cos_sims" in out else ""))
        for k in log_keys: writer.add_scalar(f"val/{k}", avgs[k], current_step)
        writer.flush()
        return avgs.get("loss", 0.0)

    try:
        for current_step in range(start_iter, config.max_iters + 1):
            train(current_step)
            if current_step % config.val_freq == 0 or current_step == config.max_iters:
                avg_val_loss = validate(current_step)
                is_best = avg_val_loss < best_val_loss
                if is_best:
                    best_val_loss = avg_val_loss
                ckpt_data = {
                    "config": config, "model": model.state_dict(), "optim": optim.state_dict(),
                    "iteration": current_step, "avg_val_loss": avg_val_loss, "best_val_loss": best_val_loss, "stage": stage,
                }
                torch.save(ckpt_data, os.path.join(ckpt_dir, "latest.pt"))
                if is_best:
                    torch.save(ckpt_data, os.path.join(ckpt_dir, "best.pt"))
                    logger.info(f"[Stage {stage}] New best at iter {current_step}: val_loss={avg_val_loss:.6f}")
    except KeyboardInterrupt:
        logger.info("Terminating...")

    writer.close()
    logger.info(f"Done. log_dir={log_dir}")

def test(args):
    config_path = glob(os.path.join(args.config, "*.yaml"))[0]
    with open(config_path, "r", encoding="utf-8") as f:
        config = EasyDict(yaml.safe_load(f))
    stage = int(config.stage)
    seed_all(config.seed)

    log_dir = get_new_log_dir(args.logdir, prefix=f"{os.path.splitext(os.path.basename(config_path))[0]}_test")
    logger  = get_logger("test", log_dir)
    writer  = SummaryWriter(log_dir)

    missing_modalities = parse_missing_modalities(args.missing_modalities)
    M = len(config.modality_names)
    logger.info(f"[Stage {stage} Test] missing={missing_modalities} use_diffusion_infer={args.use_diffusion_infer}")

    model = ADMCModel(config.model, config.modality_names, config.spec_len, args.device).to(args.device)
    ckpt = torch.load(os.path.join(os.path.realpath(args.config), "checkpoints/best.pt"), map_location=args.device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    logger.info("Loading datasets...")
    test_loader = Stage1DM(config).test_loader()
    stats_csv = os.path.join(log_dir, "test_stats.csv")
    fieldnames = ["batch_idx", "known_mask_pattern", "spec_missing_loss"] + [f"cos_sim_{name}" for name in config.modality_names]

    sum_spec, sum_cos = 0.0, [0.0] * M
    with open(stats_csv, "w", encoding="utf-8", newline="") as f_csv:
        stats_writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        stats_writer.writeheader()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc=f"[Stage {stage}] Test")):
                batch = move_to_device(batch, args.device)
                batch_size = batch["enc_batch"][config.modality_names[0]].size(0)
                known_mask = sample_known_mask_test(batch_size=batch_size, num_modalities=M, missing_modalities=missing_modalities, device=args.device)
                spec_pred, spec_missing_loss, cos_sims, _ = model.predict_spectra(batch, known_mask=known_mask, use_diffusion=args.use_diffusion_infer, return_debug=True)
                
                sum_spec += spec_missing_loss.item()
                cos_list = cos_sims.mean(dim=0).tolist()
                for idx in range(M): sum_cos[idx] += cos_list[idx]

                row = {"batch_idx": i, "known_mask_pattern": repr(known_mask[0].tolist()), "spec_missing_loss": f"{spec_missing_loss.item():.6f}"}
                for m_idx, name in enumerate(config.modality_names): row[f"cos_sim_{name}"] = f"{cos_list[m_idx]:.6f}"
                stats_writer.writerow(row)

                if args.save_to_csv:
                    append_spec_pred_to_csv(args.save_to_csv, smiles=batch.get("smiles", []), spec_pred=spec_pred, modality_names=config.modality_names, cos_sims=cos_sims)

    avg_spec = sum_spec / len(test_loader)
    avg_cos  = [v / len(test_loader) for v in sum_cos]
    logger.info(f"[Stage {stage} Test Result] spec_missing_loss={avg_spec:.4f}")
    writer.add_scalar(f"test/stage{stage}/spec_missing_loss", avg_spec)
    for i, name in enumerate(config.modality_names): writer.add_scalar(f"test/stage{stage}/cos_sim/{name}", avg_cos[i])
    writer.close()

if __name__ == "__main__":
    args = parse_args()
    main(args) if args.mode == "train" else test(args)