# SpecFLOW

SpecFLOW is a codebase for multimodal spectral completion training and inference, organized in three stages:
Stage 1 trains Encoder + WB Fuser + Decoder, Stage 2 fine-tunes with Latent Diffusion on Stage 1 weights, and Stage 3 freezes the Encoder and further tunes the Decoder.

## Requirements

- Python 3.8+
- PyTorch

## Training and Testing

### Dataset Download
[QM9S Dataset](https://figshare.com/articles/dataset/QM9S_dataset/24235333)

### Stage 1

```bash
python stage1.py /path/to/config.yaml --logdir ./logs/stage1
```

Test:

```bash
python stage1.py /path/to/logdir --mode test \
	--missing_modalities "1" \
	--use_diffusion_infer \
	--save_to_csv ./predictions_stage1.csv
```

### Stage 2

```bash
python stage2.py /path/to/config.yaml \
	--pretrained ./logs/stage1/checkpoints/best.pt \
	--logdir ./logs/stage2
```

Test:

```bash
python stage2.py /path/to/logdir --mode test \
	--missing_modalities "1" \
	--use_diffusion_infer \
	--save_to_csv ./predictions_stage2.csv
```

### Stage 3

```bash
python stage3.py /path/to/config.yaml \
	--pretrained ./logs/stage2/checkpoints/best.pt \
	--logdir ./logs/stage3
```

Test:

```bash
python stage3.py /path/to/logdir --mode test \
	--missing_modalities "1" \
	--use_diffusion_infer \
	--save_to_csv ./predictions_stage3.csv
```

