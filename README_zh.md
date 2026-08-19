# ComfyUI-Raon-OpenTTS

**[English](./README.md)** | **中文**

适用于 [KRAFTON Raon-OpenTTS](https://github.com/krafton-ai/Raon-OpenTTS) 的原生 ComfyUI 节点 —— 开放权重、开放数据的零样本声音克隆(F5-TTS 风格 CFM/DiT + HiFi-GAN,16 kHz 英语)。

支持将官方检查点重打包为三种 safetensors,包括通过 [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) 量化内核执行的 **INT8 ConvRot** 版本(权重始终以 INT8 驻留,加载时不会反量化为 bf16)。

**预转换权重:** [drbaph/Raon-OpenTTS-comfyui](https://huggingface.co/drbaph/Raon-OpenTTS-comfyui) —— 文件缺失时加载器会自动从该仓库下载。

**上游:** [KRAFTON/Raon-OpenTTS-1B](https://huggingface.co/KRAFTON/Raon-OpenTTS-1B) · [KRAFTON/Raon-OpenTTS-0.3B](https://huggingface.co/KRAFTON/Raon-OpenTTS-0.3B) · [arXiv:2605.20830](https://arxiv.org/abs/2605.20830)

| 版本 | 1B 文件 | 0.3B 文件 | 说明 |
|---|---|---|---|
| fp32 | 4.17 GB | 1.35 GB | EMA 权重的无损提取(参考用) |
| bf16 | 2.08 GB | 0.68 GB | 推荐的半精度运行时版本 |
| int8-convrot | 1.41 GB | 0.50 GB | 65.9%(1B)/ 54.6%(0.3B)参数量化,与 bf16 相比 mel 偏差约 1% |

## 节点

- **Raon OpenTTS Load Model** —— 从 `ComfyUI/models/raon_opentts` 选择检查点文件名(自动搜索两个模型文件夹;仅列出 int8-convrot 和 bf16;fp32 仅作转换参考,不列出),可选 dtype(auto/bf16/fp32)、设备、注意力后端(auto/sdpa/flash_attention/sageattention)。注册到 ComfyUI/AIMDO 内存管理。开启 `download_if_missing` 时,只会从 [drbaph/Raon-OpenTTS-comfyui](https://huggingface.co/drbaph/Raon-OpenTTS-comfyui) 下载所选的那一个版本(及其 config/vocab 和 HiFi-GAN 声码器),不会下载整个仓库。
- **Raon OpenTTS Generate (Voice Clone)** —— 零样本克隆:`text` + `ref_audio` + `ref_text`,官方默认值(steps 32、cfg 2.0、sway -1.0、基于 VAD 的时长估计、目标 RMS 0.1、150 ms 交叉淡化)。种子默认 42;0 = 随机;每个文本块递增。控制台有实时 tqdm 进度条,同时更新 ComfyUI 进度条。文本分块控制:`do_split`(开 = 长文本自动分块;关 = 始终单块)和 `max_chars`(0 = 根据参考语速自动计算块大小 —— `ref_bytes/ref_seconds × (22 - ref_seconds)`;任何正数值则强制固定预算,分块结果与说话人无关)。
- **Raon Whisper Transcribe** —— 为 `ref_text` 转写参考音频(优先复用 `ComfyUI/models/audio_encoders` 中已有的本地 Whisper 模型;缺失时才下载)。

## 模型目录结构

```text
ComfyUI/models/raon_opentts/
  Raon-OpenTTS-1B/
    config.yaml
    vocab.txt
    Raon-OpenTTS-1B-fp32.safetensors
    Raon-OpenTTS-1B-bf16.safetensors
    Raon-OpenTTS-1B-int8-convrot.safetensors
  Raon-OpenTTS-0.3B/
    config.yaml
    vocab.txt
    Raon-OpenTTS-0.3B-fp32.safetensors
    Raon-OpenTTS-0.3B-bf16.safetensors
    Raon-OpenTTS-0.3B-int8-convrot.safetensors
  tts-hifigan-libritts-16kHz/generator.ckpt   (自动下载)
```

## 复现检查点

需要官方训练检查点(KRAFTON HF 仓库中的 `model_520000.pt` / `model_225000.pt`)、`config.yaml` 和 `vocab.txt`,以及克隆到 `upstream/` 的 [Comfy-Org/comfy-model-tools](https://github.com/Comfy-Org/comfy-model-tools)。

```powershell
$PY = "C:\path\to\ComfyUI\venv\Scripts\python.exe"
$M  = "C:\path\to\ComfyUI\models\raon_opentts\Raon-OpenTTS-1B"

# 1) 干净的纯推理提取(EMA 权重,fp32)
& $PY tools/convert_raon_to_safetensors.py --ckpt "$M/model_520000.pt" `
    --config "$M/config.yaml" --vocab "$M/vocab.txt" `
    --out "$M/Raon-OpenTTS-1B-fp32.safetensors" --dtype keep

# 2) 验证提取是否逐位一致(全部 448 个张量,torch.equal)
& $PY tools/validate_raon_safetensors.py --safetensors "$M/Raon-OpenTTS-1B-fp32.safetensors" `
    --config "$M/config.yaml" --source "$M/model_520000.pt"

# 3) bf16 运行时版本(inv_freq 保持 fp32)
& $PY tools/convert_raon_to_safetensors.py --ckpt "$M/model_520000.pt" `
    --config "$M/config.yaml" --vocab "$M/vocab.txt" `
    --out "$M/Raon-OpenTTS-1B-bf16.safetensors" --dtype bf16

# 4) INT8 ConvRot(先 dry-run,再生成并验证)
& $PY tools/quantize_raon_int8_convrot.py --src "$M/Raon-OpenTTS-1B-fp32.safetensors" --dry-run
& $PY tools/quantize_raon_int8_convrot.py --src "$M/Raon-OpenTTS-1B-fp32.safetensors"
& $PY tools/validate_raon_int8_convrot.py "$M/Raon-OpenTTS-1B-int8-convrot.safetensors" --expect-blocks 28

# 5) 内核单元测试 + 端到端对比(在 ComfyUI 根目录下运行)
& $PY tools/test_int8_kernels.py "$M/Raon-OpenTTS-1B-int8-convrot.safetensors" "$M/Raon-OpenTTS-1B-fp32.safetensors"
& $PY tools/test_e2e_int8_vs_fp32.py 1B
```

0.3B 使用 `model_225000.pt` 和 `--expect-blocks 22`。

## 量化方案(V1,保守)

量化范围:每个 DiT 块中恰好 6 个重复的块级 GEMM —— `attn.to_q/to_k/to_v/to_out.0`、`ff.ff.0.0`(上行)、`ff.ff.2`(下行)。

- 1B(dim 1408,heads 24,inner 1536,ff 5632):168 层 = 112x GS64(K=1408)+ 56x GS256(K=1536/5632)。注意 1408 % 256 != 0,所以 GS64 是让 1408 宽的主干能用上 ConvRot 的关键。
- 0.3B(dim 1024,inner 1024,ff 2048):132 层,全部 GS256。
- 保持全精度:词元嵌入、ConvNeXt 文本块、逐块 AdaLN 调制、时间嵌入 MLP、输入投影、卷积位置编码、最终 AdaLN + proj_out、归一化层、`inv_freq`。
- 标准 absmax 缩放(不使用 MSE 裁剪)。逐行 fp32 scale,逐层 JSON 标记(`int8_tensorwise` + `convrot` + `convrot_groupsize`)。

## 词表说明(5559 vs 5555)

发布的 `vocab.txt` 有 5559 个词元;两个检查点的嵌入表都是 5555(含 filler 共 5556 行)。因此模型按 `text_num_embeds=5555` 构建(以检查点为准),排序文件尾部 4 个最高码位词元(U+FDFA `ﷺ`、U+FDFB `ﷻ`、U+FFFD `�`、U+1F3B5 `🎵`)会在运行时从映射中丢弃(与其他未知字符一样映射到索引 0)。由于词表文件是有序的,且多出的 4 个词元恰好是码位最高的 4 个,所有可用字符的索引完全一致;生成音频的 Whisper 转写已确认英文输出正常。针对训练语料做形式化验证并不现实(数据集约 19 TB),因此请把这 4 个码位视为不支持。

## 其他说明

- 注意力:`auto` 在安装了 `flash_attn`(FA2)且为半精度计算时使用 FA2,否则使用 torch SDPA。官方模型默认 SDPA;FA2 在 bf16 舍入范围内数值等价,长文本生成时更快。sageattention 作为 SDPA 补丁运行。
- ODE 时间调度在 fp32 下计算后一次性转换到计算 dtype(上游直接用计算 dtype 计算;在 bf16 下相邻时间步会塌缩成相同值,torchdiffeq 会拒绝)。
- INT8 速度:在 RTX 5090 上、这些 GEMM 尺寸下,INT8 ConvRot 目前约为 bf16/fp32 的速度,但显存占用最低(1B 推理峰值约 1.7 GB,对比 bf16 2.4 GB / fp32 4.5 GB)。

## 引用

```bibtex
@article{kim2026raonopentts,
  title     = {Raon-OpenTTS: Open Models and Data for Robust Text-to-Speech},
  author    = {Kim, Semin and Chung, Seungjun and Moon, Taehong and Lee, Sangheon and Ahn, Minyoung and Lee, Keon and Kim, Nam Soo and Cho, Jaewoong and Schmidt, Ludwig and Lee, Kangwook and Park, Dongmin},
  journal   = {arXiv preprint arXiv:2605.20830},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.20830}
}
```

## 许可证

代码:MIT。模型权重(包括 [drbaph/Raon-OpenTTS-comfyui](https://huggingface.co/drbaph/Raon-OpenTTS-comfyui) 上的转换检查点)采用 [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/),与上游 KRAFTON 发布一致。模型代码来自 [krafton-ai/Raon-OpenTTS](https://github.com/krafton-ai/Raon-OpenTTS)(Apache-2.0),其本身基于 [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS);HiFi-GAN 声码器改编自 speechbrain(Apache-2.0)。
