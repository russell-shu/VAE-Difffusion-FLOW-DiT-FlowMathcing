# Stage 12 · REPA + 完整管线 + 轻量评估

本阶段把前面所有模块接成一条链，并加上论文 4.1 节末尾的 **REPA**
（Representation Alignment）与最小客观指标。

## 训练目标

1. **CFM（与 stage 9–11 相同）**：在 mask 内预测速度场，损失为
   `(1-m) ⊙ ||v_pred - (z1-z0)||²` 的均值。
2. **REPA（Yu 2024）**：取 DiT 第 `repa.dit_layer` 个 block 输出 token
   序列（0-based，默认 7 即「第 8 层」），经线性层投影后与冻结
   HuBERT / mHuBERT 的隐藏状态做 L1。HuBERT 在 **GT 波形** 上提取特征，
   不参与反传。

总损失：`loss = loss_fm + repa.weight * loss_repa`。

若本机未安装 `transformers` 或无法下载 HuBERT 权重，脚本会打印警告并
**自动关闭 REPA**，仅训练 CFM（仍可用于打通流程）。

## 推理

与 stage 11 相同：`EulerTTSSolver` + mismatch 修正 + 无条件分支去掉
prompt 噪声 + **APG** 引导，最后 `WavVAE.decode`。

## 评估（`mini_audiodit/eval/metrics.py`）

默认始终可算：

- `wave_l1`：时域 L1
- `stft_l1`：多分辨率 STFT 对数幅值 L1（与 Wav-VAE 里 STFT 损失同族）

若安装了 `pesq` / `pystoi`，会额外尝试 PESQ、STOI。论文中的 Whisper
WER、WavLM SIM、UTMOS、DNSMOS 依赖更重，可在该文件中自行接入口。

## 运行

```bash
python scripts/stage12_full_pipeline.py --config configs/stage12_full_pipeline.yaml
```

产物：`runs/stage12/full.pt`、`sample_*.wav`、`train.log`。

## 论文锚点

- 4.1 节 REPA + mHuBERT（Boito 2024）
- 4.3–4.4 推理：Euler、mismatch、APG
- 3–4 节：Wav-VAE 潜空间 + CFM + DiT
