# Qwen Image 2.1 Speedup

Sampling accelerator for Qwen Image 2.1 in ComfyUI.

## Node

- Display name: `Qwen Image 2.1 Speedup`
- Class ID: `QwenImage21Speedup`
- Category: `model/patch`

## 一体化加速采样器

- 显示名称：`Qwen Image 2.1 加速采样器`
- 节点 ID：`QwenImage21FastSampler`
- 类别：`采样/Qwen Image 2.1`
- 直接连接 Qwen-Image 2.1 的模型、正负条件与潜变量，输出潜变量接 VAE 解码。
- 图像编辑时直接使用 `TextEncodeQwenImage21` 输出的潜变量；节点不会中途缩放，避免编辑参考位置偏移。
- 默认是 40 步、CFG 1、Euler / Simple、`均衡`档。`关闭`档通过同一节点走原生 KSampler，可用于固定种子的速度和画质对照。
- `保真`、`均衡`、`快速`档依次增加缓存复用和画质近似；这些档位不是少步数模型，不保证每个工作流有相同提速。
- 可以在模型输入前接 ComfyUI 原生 `Qwen Image 2.1 Cache`；它管理文字/参考图的 KV 缓存，本节点复用的是逐步的模型残差，两者作用不同。
- 控制台分别记录跳过的模型前向次数和本节点采样耗时。不要把“跳过百分比”直接当成整张图的提速百分比。

旧 `QwenImage21Speedup` 模型补丁节点继续保留，旧工作流无需迁移。同一条链路只需使用上述两种加速入口之一；若都连接，一体化采样器会覆盖上游补丁档位。

### CFG>1 与原生 KV 缓存的已知限制

- CFG 输入允许大于 1，但参数允许设置不代表所有缓存组合都已验证。下方历史速度测试使用的是 CFG=1。
- 部分 ComfyUI 版本的 Qwen 2.1 原生多条件 KV 缓存在选择缓存槽位时会报张量尺寸不匹配。CFG>1 会执行正负两路计算，可能触发这一宿主错误。本节点当前没有集成该原生缓存的修复。
- 遇到这类错误，可在模型与采样器之间连接原生 `Qwen Image 2.1 Cache`，保持节点正常执行并设置 `device=off`。旁路、删除这个节点并不等于关闭模型内置的 KV 缓存，其缺省策略仍是 `auto`。
- 采样器的 `均衡/保真/快速` 和 `残差缓存位置=自动/显存/内存` 控制的是另一层残差缓存，不能关闭原生 KV 缓存。显式关闭原生 KV 后，本节点的残差加速仍保留，但会失去原生 KV 提供的那部分提速。
- 双采两个阶段实际启用时，可以选择 `原生 KV 缓存模式=安全关闭`，它会在两个采样模型副本上显式设置 `device=off`。单采或关闭模型二时，应通过上游原生 Cache 节点显式设置；使用不同模型时要分别处理两路模型。
- 当前发布没有新增 CFG=4 专用误差控制，也不承诺 CFG=4 的提速倍率、无损画质或消除所有运行错误。请用固定种子与同 CFG 的关闭加速结果对照。

## Qwen Image 2.1 专用双采

- `Qwen Image 2.1 双采（百分比）` / `QwenImage21DualPercentSampler`
- `Qwen Image 2.1 双采（SNR）` / `QwenImage21DualSNRSampler`
- 两个节点有相同的模型一/模型二连接、CFG、计划步数、采样器、调度器和各自的加速档位。模型二可接另一份 Qwen Image 2.1 模型或同模型的另一组 LoRA。只接受相同 Qwen 2.1 潜空间，不会混接其他架构。
- 一采与二采在同一个物理 sigma 处交接：一采只加一次起始噪声，未放大时二采接续同一带噪潜变量，**不会把两个完整 KSampler 串联**。模型二关闭或三项模型二输入全未连接时，一采运行自己的完整步数，交接设置不参与。只连部分模型二输入会明确报错。
- 默认一、二采各计划 40 步，Euler / Simple、CFG 1；模型一 `均衡`，模型二 `保真`；百分比节点默认 50%，SNR 节点默认 6 dB。实际执行步数由两套 sigma 时间表的交点决定，控制台记录每阶段步数和耗时。
- 双采默认在采样模型**副本**上关闭 Qwen 原生 KV 缓存，规避本机既有的多参考图交接错误；仍可选择沿用上游 KV 缓存设置。逐步残差加速独立生效。默认显存模式为 `省显存`，采样前清旧模型、各阶段后卸载；`自动`在同模型同 LoRA 时尽量复用。

### 内部放大

- `最终宽度=0` 且 `最终高度=0`：**完全不放大**，输出保持输入潜变量的尺寸；只填一边时按输入长宽比计算另一边。
- 文生图双采：一采在输入小尺寸完成到干净潜变量，再用 `bislerp`（可改）放大，二采在目标尺寸重新加受控噪声继续精修。默认 `重绘强度=1.0` 使用交接 sigma；`0.8` 使用其 80%，并非从纯噪声重新生成。放大前至少完成 10 个一采计划步，必要时自动推迟交接。
- 图像编辑双采：接**两个** `Text Encode Qwen Image 2.1` 节点，使用同一批参考图和对应提示词。模型一的正负条件及输入潜变量来自低分辨率编码；模型二的正负条件来自按最终宽高重新编码的高分辨率编码。节点检查两阶段第一参考图的 latent 尺寸是否分别匹配输入尺寸和目标尺寸；不匹配就报错，不偷偷放大低分辨率参考特征。编码节点需要连接 VAE，以产生参考潜变量。
- 编辑放大优先保持默认 `重绘强度=1.0`。降低它虽会减少二采起始噪声，但不保证更像原图；本机茶壶参考编辑测试中，`0.8` 出现明显双边/黏连。对姿态和轮廓要求严格时，应与目标尺寸直接编辑单采做固定种子对照。
- 模型二关闭时，若填写最终宽高，纯文生图会先缩放空潜变量再由模型一采样；带参考图的编辑应在编码节点先设置目标尺寸，节点不会内部放大。
- `第一采结果`是 latent：普通交接时仍带噪声，不能当成完整预览图；启用中途放大时它已收敛为可解码的低分辨率 latent。
- 暂不支持带噪声遮罩的内部放大。编辑放大仍可能使姿态、边缘和细节发生变化；需要固定种子对照，不保证与直接大图单采逐像素一致。

### 本机单次验证（2026-09-30）

独立 ComfyUI 0.37.0 实例、RTX 5080、Qwen-Image 2.1 INT8、512×512、40 步、固定种子 42、Euler / Simple、CFG 1、原生 KV 缓存 `auto/default`：

| 工作流 | 热态关闭加速 | 均衡加速 | 实际模型前向复用 |
| --- | ---: | ---: | ---: |
| 文生图采样 | 5.094 秒 | 2.392 秒 | 22 / 40 |
| 单参考图编辑采样 | 7.384 秒 | 2.980 秒 | 22 / 40 |

编辑测试图中的茶壶主体和桌面位置均保持；重复运行相同编辑条件时，两张加速结果的解码像素完全一致。加速与无加速结果并非逐像素相同，这是近似缓存的预期现象。上述数字只是该配置下的采样器耗时，不代表其他分辨率、模型精度、LoRA 或整张图一定有相同倍率。

## What it does

Step-level residual caching, compatible with the model's built-in prefix K/V
cache (text + reference image K/V computed once per sampling run):

- **Skip gate (TeaCache-style)**: while the accumulated relative drift of the
  timestep embedding stays under `cache_threshold`, the whole transformer
  forward is skipped. The first/last stretch of the schedule
  (`cache_start_percent` / `cache_end_percent`) and `max_consecutive_skips`
  bound the approximation. CFG positive/negative streams are cached
  independently.
- **Residual forecast (TaylorCache-style)**: the residual applied on a cached
  step is extrapolated in sigma space from the measured residual history —
  `first` uses the last two residuals (secant), `second` adds the quadratic
  term from the last three (clamped to the linear term's magnitude), `off`
  replays the latest residual verbatim.
- **Adaptive mode**: with `target_error` > 0, the actual error of each skip
  run is measured against the next real forward and the effective threshold
  is adjusted (0.7x-1.3x per correction) to hold the error near the target.
  Three starved steps raise the threshold by 1.5x so the controller can
  always re-probe upward.

## Recommended chain

```
UNETLoader → Qwen Image 2.1 Speedup → KSampler
```

Add the official `Qwen Image 2.1 Cache` node after this one when VRAM is
tight (int8 prefix K/V). Nothing else is required: the mixed-granularity
attention and the prefix K/V reuse from the Qwen Image 2.1 architecture are
already built into ComfyUI.

## Tuning

Designed and calibrated for 40-step runs; the defaults are the measured
sweet spot (adaptive `target_error` 0.05, first-order forecast, ~50% of
forwards skipped at ~1.7x model speedup).

- **Adaptive mode (default, `target_error = 0.05`)**: after every skip run
  the replay error is measured against the next real forward and the
  effective threshold adjusted (0.7x-1.3x per correction). 0.06-0.08 is
  faster with slightly lower fidelity. `cache_threshold` (0.30) is the
  controller's starting point.
- **Fixed mode** (`target_error = 0`): measured drift is ~0.13 per step at
  40 steps, so the threshold is roughly 0.13 x the skip run length: 0.3
  skips ~2 steps per refresh, 0.5 ~3-4, 0.8 ~6.
- Keep `cache_start_percent` >= 0.15: measured replay error peaks right at
  the schedule start.
- `forecast` `first` is the measured sweet spot: at equal skip rate it cuts
  the replay error roughly in half vs `off`. `second` showed no further gain
  in testing and costs extra time; it stays available for comparison.
- The node logs the skip rate at the end of each run; with `debug_log` (or
  adaptive mode) it also reports the measured replay error mean/max and the
  final effective threshold.
