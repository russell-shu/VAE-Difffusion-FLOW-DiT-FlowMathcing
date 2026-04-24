LongCat-AudioDiT 系统学习与复现规划（v2 · 音频对齐版）

一、总体策略

你原规划用 MNIST 走完全程，优点是快，但缺点是学完仍不会处理变长 1D 音频、文本条件、masked-prompt、对抗式 VAE 等论文核心。

新规划采用双轨推进：

Track A (理论快速 toy, MNIST, 阶段 1-4):
  VAE -> DDPM(对照) -> Rectified Flow + Euler -> CFG

Track B (音频复现 LJSpeech/LibriTTS-R, 阶段 5-12):
  Audio I/O -> mini-AE -> Wav-VAE(KL+GAN+multi-STFT)
       |
       v
  DiT (AdaLN/RoPE/QK-Norm) + CFM on latent (uncond)
       |
       v
  UMT5 Cross-Attn -> Masked Conditioning
       |
       v
  Mismatch fix -> APG -> REPA -> full pipeline

Track A 只是打理论底，不追求生成质量，每阶段 1-2 天过完。
Track B 是论文主线，每阶段必须能复现对应章节的现象/消融。

二、与原规划的 5 个核心差异





阶段 2 (DDPM) 降级为理论对照组：论文不用 DDPM，但要懂 Ho et al. 2020 和 Albergo 2025 给出的 DDPM/FM 统一性。只写 ε-prediction + ancestral sampling 的最小版，不展开 noise schedule 调优。



切换到音频的时机提前到阶段 5：原规划在阶段 9 才整合，太晚。音频的 1D 序列、变长、对抗训练特性必须早接触。



Wav-VAE 拆成 "mini-AE + 完整 Wav-VAE" 两阶段：论文的 Wav-VAE 同时包含 KL、multi-res STFT loss、multi-scale STFT discriminator、feature matching、Snake activation、Oobleck block、non-parametric shortcut（见论文 3.1-3.2 节），一次性实现极易出错。



新增"条件注入机制"专项 (阶段 7-8)：AdaLN、Cross-Attention、RoPE、QK-Norm、Global AdaLN、Long-skip 都是论文明确要求的，原规划没拆。



明确复现论文独有贡献：





Masked conditioning (阶段 9, 论文式 4)



Training-inference mismatch 修正 (阶段 10, 论文式 7)



CFG unconditional 分支同步 drop z_t^ctx (论文 4.3 末尾 corollary)



APG 的 velocity-to-sample domain 投影 + reverse momentum (阶段 11, 论文式 9-10)



REPA 用 mHuBERT 对齐第 8 层 (阶段 12, 论文 4.1 节末尾)

三、Mini 化原则（单卡/少卡可跑）





数据：LJSpeech (~24h) 或 LibriTTS-R train-clean-100 子集



采样率：24 kHz（对齐论文），或 16 kHz（更省）



片段长度：3-6 秒



Wav-VAE：10-30M 参数（论文 157M）



DiT：30-80M 参数（论文 1B/3.5B）



每阶段只训练到"现象出现"即可，不追 SOTA

四、各阶段设计规范

每阶段按以下结构：





目标：学到什么概念/能力



论文锚点：对应论文哪一节 / 公式 / 参考文献



代码任务：具体实现哪些模块



验证点：必须观察到的现象（失败则不进入下阶段）

阶段 0 · 环境 + 音频 I/O 骨架





目标：统一训练框架（dataset / model / trainer / config / sampler），音频加载/可视化



论文锚点：5.1 节实验设置



代码任务：PyTorch Lightning 或裸 PyTorch 均可；torchaudio 读写；波形图 + mel-spec 调试可视化；LJSpeech Dataset



验证点：能加载、播放、画图；框架能跑一个 dummy MLP 训练回路

阶段 1 · VAE on MNIST





目标：理解 reparameterization、KL 项、后验崩塌



论文锚点：Kingma & Welling 2013；论文 3.1 节的 z = μ + σ ⊙ ε



代码任务：encoder 输出 mean/logvar、decoder 重建、loss = recon + β·KL



验证点：latent 可视化接近 N(0,I)；从先验采样能生成数字

阶段 2 · DDPM 对照组 (MNIST, 教学用)





目标：懂 forward q(x_t|x_0) 闭式解、反向 ε-prediction、为什么不方便



论文锚点：Ho 2020；Sohl-Dickstein 2015；Albergo 2025（FM/DDPM 统一）



代码任务：最小 DDPM，简单 2D UNet，linear/cosine schedule；ancestral sampling



验证点：能采样出数字；明确写下 DDPM 和 rectified flow 的数学对应关系

阶段 3 · Rectified Flow + Euler (MNIST, 论文核心数学)





目标：真正对齐论文式 (3)(4)



论文锚点：Lipman 2022、Liu 2022a (Rectified Flow)、论文式 (3) z_t = (1-t)z_0 + t z_1



代码任务：





训练：采 z_0 ~ N(0,I)、z_1 ~ data、t ~ U(0,1)，loss = MSE(v_pred, z_1 - z_0)



采样：Euler，从 z_0 积分到 z_1，支持 NFE=8/16/50



验证点：NFE 越大质量越好；对比阶段 2 的 DDPM 采样，写一段笔记说明为什么 FM 数学更简洁

阶段 4 · Conditional + CFG (MNIST)





目标：懂 conditional dropout 训练 + guidance scale 推理



论文锚点：Ho & Salimans 2021；论文式 (8)



代码任务：label embedding 条件；训练 10% drop 条件；推理时 v_cfg = v + α(v - v_u)



验证点：α=1→4→10 的视觉变化（小α模糊多样，大α尖锐但失真）

阶段 5 · Mini 1D Audio AE





目标：从图像思维切换到 1D 波形；无 KL、无 GAN 的纯重建基线



论文锚点：论文 3.1 节（encoder/decoder 骨架）；Evans 2024 (Oobleck)；Ziyin 2020 (Snake)



代码任务：





1D conv encoder/decoder，weight-norm



Oobleck block：dilated residual unit + Snake activation



Non-parametric shortcut（space-to-channel reshape + 通道平均），见论文 3.1 节



纯 L1 time-domain + multi-res STFT recon loss



验证点：重建音频能听；downsample ratio R 按论文配置（11.72 Hz ≈ 24000/2048 倍压缩）

阶段 6 · Wav-VAE 完整版





目标：真正复现论文 Wav-VAE



论文锚点：论文 3.2 节；Kumar 2023 (DAC multi-scale mel loss)、Zeghidour 2021 (multi-res STFT)、Kong 2020 (HiFi-GAN feature matching)



代码任务：在阶段 5 上加：





VAE bottleneck (μ + σ ⊙ ε + KL loss)



Multi-scale mel-spectrogram loss



Multi-scale STFT discriminator



Adversarial loss + feature matching loss



Warmup（先纯重建，再启用 adv/fm）



验证点：PESQ ≈ 3.0+、STOI ≈ 0.96+ 在 LibriTTS test-clean 子集；latent 近似 N(0,I)

阶段 7 · DiT 结构 + CFM on latent (unconditional)





目标：搭建论文主干，先不加文本条件



论文锚点：论文 4.1 节；Peebles & Xie 2023；Perez 2018 (AdaLN)、Su 2024 (RoPE)、Henry 2020 (QK-Norm)、Chen 2024b (Global AdaLN)



代码任务：





Self-Attention + AdaLN(t) 注入时间步



RoPE + QK-Norm (RMSNorm)



Long-skip connection（input 加到最后一层）



Global AdaLN（所有层共享 AdaLN 投影）



Patch/token embedding on Wav-VAE latent



CFM loss + Euler 采样



验证点：能在 Wav-VAE latent 上做 unconditional 生成，decoder 出来是"类语音"的噪声

阶段 8 · 多语言文本编码 + Cross-Attention





目标：完成文本条件注入



论文锚点：论文 4.2 节；Chung 2023 (UMT5)、Woo 2023 (ConvNeXt V2)；论文式 (5)



代码任务：





加载 google/umt5-base，抽 last_hidden_state + raw_word_embedding（embedding 层）



q = LN(last) + LN(raw_emb)



ConvNeXt V2 refinement 模块



DiT 里每层加 Cross-Attention 到 q



验证点：给定文本能生成对应内容的语音（质量差没关系，关键是对齐出现）

阶段 9 · Masked Conditioning (零样本声音克隆)





目标：复现 VoiceBox-style 的 prompt masking



论文锚点：论文式 (4)；Le et al. 2024 (VoiceBox)



代码任务：





训练时随机 mask 连续 span，z_ctx = z_1 ⊙ (1-m)



Loss 只算 masked 区域：(1-m) ⊙ ‖(z_1-z_0) - v_θ‖²



训练 10% drop z_ctx 和 q（为 CFG 做准备）



验证点：能给 prompt 音频 + 新文本，生成保持说话人音色的语音

阶段 10 · Training-Inference Mismatch 修正（论文独有贡献 1）





目标：理解并复现论文 4.3 节



论文锚点：论文 4.3 节，式 (6)(7)



代码任务：Euler 采样每步：





计算 v(z_t, t, z_ctx, q)，做 z_{t+Δt} = z_t + v·Δt



每步强制覆盖 z_t^ctx ← t·z_ctx + (1-t)·z_0^ctx（式 7）



Unconditional 分支的 z_t^u = concat(∅, z_t^gen)（4.3 末尾 corollary，同步 drop 噪声 prompt）



验证点：消融对比（论文 Table 4）—— 修正后 SIM/UTMOS/DNSMOS 显著上升

阶段 11 · CFG → APG 替换（论文独有贡献 2）





目标：理解并复现论文 4.4 节



论文锚点：论文 4.4 节，式 (8)(9)(10)；Sadat 2024 (APG)；Kynkäänniemi 2024（饱和现象）



代码任务：





先实现标准 CFG（式 8）



APG：





velocity → sample domain: μ_t = z_t + (1-t)v_t



Δμ_t = μ_t - μ_t^u



正交分解：Δμ∥ = <Δμ, μ>/<μ,μ> · μ，Δμ⊥ = Δμ - Δμ∥



μ_APG = μ_t + α·Δμ⊥ + η·Δμ∥（默认 η=0.5）



反向回 velocity: v_APG = (μ_APG - z_t)/(1-t)



Reverse momentum: Δμ_t ← Δμ_t + β·Δμ_{t-1}（默认 β=-0.3）



验证点：CFG 高 α 出现的金属音/尖锐伪影在 APG 下消失；UTMOS/DNSMOS 上升

阶段 12 · REPA + 整合 + 评估





目标：完成最后一块拼图并跑通 end-to-end



论文锚点：论文 4.1 节末尾；Yu 2024 (REPA)、Boito 2024 (mHuBERT)



代码任务：





加载 mhubert-147，L1 对齐 DiT 第 8 层输出与 mHuBERT 特征



完整 pipeline：text → UMT5 → DiT(CFM) → Euler+mismatch fix+APG → Wav-VAE decoder → waveform



评估：CER (Whisper-large-v3)、SIM (WavLM fine-tuned)、UTMOS、DNSMOS



验证点：能做 zero-shot voice cloning；消融实验结果方向与论文 Table 3、4 一致

五、预计产出





一个可训练/可推理的 mini-LongCat-AudioDiT 仓库



每个阶段一份笔记（md），记录：理论要点 + 实验现象 + 与论文对照



论文 Table 3（Wav-VAE vs Mel-VAE）和 Table 4（mismatch/APG 消融）的小规模复现结果



对 RQ1/RQ2/RQ3 三个研究问题的自己的回答

六、需要你在开工前再明确两件事（可放心开干但影响工程选型）





数据规模：是只用 LJSpeech 单说话人（简单、零样本无法真实验证）还是 LibriTTS-R 多说话人子集（能做 zero-shot 克隆验证，磁盘更大）



采样率：24 kHz 对齐论文但序列长；16 kHz 更省，评估工具也够用。二选一会在阶段 0/5 固定下来。

进入执行阶段时再选即可，规划本身不受影响。