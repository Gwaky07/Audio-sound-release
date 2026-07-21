# Audio-sound

`Audio-sound` 是一个面向中文口播、课程讲解、旁白和访谈音频的 Windows 本地音频清理工作流仓库。

它的目标不是做一个轻量预览，而是让 Codex 或人工操作者能直接得到可交付的修音成品：降噪、去呼吸/口水音、处理停顿残留、修复删词接缝，并输出可复核的报告与频谱证据。



## 效果展示

> 说明：仓库中不提交客户原始音视频或成品音视频。下面的图是公开安全的波形示意，用来说明本仓库重点解决的“边界残留 / 接缝杂音 / 刺音爆点”问题。真实项目的成品音视频建议放在飞书文档或 GitHub Releases 的 Assets 中。

![修音前后对比示意](docs/assets/before-after-waveform.svg)

真实处理证据示例：下图来自一次已完成修音任务的频谱/波形复核图，已裁掉文件说明，仅保留“处理前/处理后”的声学证据。

![真实频谱与波形修复对比](docs/assets/spectrum-before-after.jpg)

典型处理效果：
- 删除重复字、口误、废句后，避免残留半个音节。
- 对接缝使用短交叉淡化或安全停顿，减少“突然接上下一句”的卡顿感。
- 对刺耳瞬态、爆点、齿音进行窄窗口压制，尽量不伤正常人声。
- 导出后做完整二次复核，而不是只检查被修改的时间点。

## 适用范围

当前仓库主要适合：

- 中文课程讲解
- 中文口播 / 配音 / 旁白
- 中等噪声环境下的采访音轨
- 需要批量处理并保留报告的音频
- 需要删除口误、重复字、废句并修复接缝的视频/音频

## 仓库结构

- `.codex/skills/audio-sound/`：仓库级 Codex skill，记录默认处理规则和验收标准。
- `audio_sound/`：核心 Python 包，包含配置、流程、删词剪辑和 CLI 逻辑。
- `scripts/audio_cleanup.py`：底层清理、检查、安装和维护入口。
- `scripts/audio_skill_workflow.py`：推荐的最终成品处理入口。
- `scripts/remove_spoken_segments.py`：物理删词、删句、视频同步剪辑入口。
- `presets/`：处理预设。
- `docs/`：架构、调参和参考说明。
- `tests/`：单元测试。
- `release/`：仅用于说明发布包位置；正式压缩包放在 GitHub Releases。

## 环境要求

必需：

- Python 3.10+
- `ffmpeg`
- `ffprobe`

推荐：

- `torch`
- `torchaudio`
- `librosa`
- `intervaltree`
- `deepfilternet`

可选外部资产：

- 本地 `Respiro-en` 仓库
- 本地 `respiro-en.pt` 权重

注意：`tools/`、`ffmpeg/`、`.env`、`output/`、`scratch/` 都是本地运行资产，不进入 Git 仓库。

## 快速开始

检查运行环境：

```bash
python scripts/audio_cleanup.py doctor
```

Windows 下初始化本地环境：

```bat
setup.cmd
```

列出可用预设：

```bash
python scripts/audio_cleanup.py list-presets
```

检查单个音频文件：

```bash
python scripts/audio_cleanup.py inspect "D:/audio/raw-voice.wav"
```

## 推荐成品流程

默认最终交付请使用：

```bash
python scripts/audio_skill_workflow.py run "D:/audio/raw-voice.wav"
```

该流程会：

1. 提取源音频为 WAV。
2. 检查运行环境和可用依赖。
3. 执行呼吸、口水音、噪声和响度处理。
4. 输出最终 WAV 和 MP3。
5. 生成报告、指标和频谱证据。
6. 将最终成品放入 `output/修音成品/`。

最终命名规则：

```text
修音版_<原音频文件名>.wav
修音版_<原音频文件名>.mp3
```

如果同名文件已存在，则自动递增为 `_01`、`_02`。

## 删词与接缝修复

当目标是删除一句话、口误、重复字或指定时间段时，不要只做降噪，应使用物理删词脚本：

```bash
python scripts/remove_spoken_segments.py run "D:/video/第二段.mp4" --cut "0.62,1.82" --removed-phrase "啊说德语啊"
```

多个片段可以重复传入 `--cut`：

```bash
python scripts/remove_spoken_segments.py run "D:/video/第六段.mp4" --cut "0.00,1.80" --cut "2.10,2.34" --removed-phrase "删除口误"
```

默认删词规则：

- `--boundary-search-ms 80`：在边界附近寻找更低能量位置，避免保留残留音素。
- `--crossfade-ms 12`：短交叉淡化，只用于防爆点，不用于掩盖残留。
- 视频输入会同步补偿音频交叉淡化，避免多次删除后音画漂移。

如果删完后仍然像“突然接上下一句/一个字”，不要继续拉长交叉淡化，改用语音安全接缝：

```bash
python scripts/remove_spoken_segments.py run "D:/video/第二段.mp4" --cut "0.62,1.82" --removed-phrase "啊说德语啊" --seam-pause-ms 80
```

`--seam-pause-ms 60–100` 会让前句淡出、短暂停顿、下一句淡入，并在视频上冻结上一帧同等时长。

## 常用预设

- `fast`：快速一遍处理，动态处理较轻。
- `safe`：默认中文口播安全预设。
- `review`：清理后额外输出可疑静音/残留候选。
- `voice-isolate`：使用同文件噪声窗口做更有针对性的非人声残留压制。

示例：

```bash
python scripts/audio_cleanup.py clean "D:/audio/raw-voice.wav" --preset voice-isolate --noise-window 143.089208:144.093687
```

## 输出目录

默认输出分两类：

- `output/修音成品/`
  - 面向用户的最终 WAV/MP3/MP4。
  - 文件名遵守 `修音版_<原文件名>` 规则。
- `output/skill-<timestamp>_<source>/`
  - 流程报告、指标和频谱证据。
  - 中间音频默认清理，避免误交付。

排查时如需保留中间音频：

```bash
python scripts/audio_skill_workflow.py run "D:/audio/raw-voice.wav" --keep-intermediate-audio
```

## Codex 使用规则

本仓库包含项目 skill：

```text
.codex/skills/audio-sound/SKILL.md
```

在本仓库中，如果用户要求“处理音频”“修一下音频”“剪掉这句”“边界有残留”，Codex 应默认理解为最终可交付任务，而不是简单预览。

关键规则：

- 默认成品入口是 `python scripts/audio_skill_workflow.py run ...`。
- 删词、删句、重复字和接缝问题必须使用 `scripts/remove_spoken_segments.py`。
- 时间码只是候选，必须结合波形、频谱和局部听感确认边界。
- ASR 只能辅助定位，不能单独判定“重复字已解决”。
- 导出后必须二次完整复核，不只检查修改过的窗口。
- 旧错误成品不要作为新剪辑源，只能作为缺陷参考。

## 测试

运行测试：

```bash
pytest -q
```

或：

```bash
python -m unittest discover tests
```

运行 doctor：

```bash
python -m audio_sound.cli doctor
```

## 发布包

干净源码压缩包应放在 GitHub Releases 中，不提交到代码目录。

推荐生成方式：

```bash
git archive --format=zip --output tmp/Audio-sound-release-source.zip HEAD
```

该压缩包只包含 Git 已跟踪源码，不包含 `.env`、`tools/`、`ffmpeg/`、`output/`、`scratch/` 等本地资产。

## 当前不做的事情

- 图形界面剪辑器
- 完整 ASR 工作台
- 说话人分离
- 与 Adobe Audition 完全一致的处理链

这些能力可以后续扩展；当前仓库优先保证中文口播音频清理和删词接缝修复的稳定自动化。
