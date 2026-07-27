# Audio-sound 发布包快速开始

GitHub 的 clean-source 分享包只包含仓库源码、Codex skill 和配置文件，不包含 Python 虚拟环境、`ffmpeg/`、`tools/`、模型权重或任何用户音频。

在接收端 Windows 机器上：

1. 安装 Python 3.11 或 3.10。Python 3.12 及以上版本不受支持。
2. 安装系统级 FFmpeg，确保 `ffmpeg` 和 `ffprobe` 可从 `PATH` 调用；也可以手动放入本目录的 `ffmpeg\bin`。
3. 在本目录打开 PowerShell 或 Command Prompt。
4. 运行 `setup.cmd`。脚本优先使用 Python 3.11，其次使用 3.10，并创建仓库专用 `.venv`。
5. 运行 `check_runtime.cmd` 或 `doctor.cmd`。
6. 使用以下命令处理音频：

```bat
run_audio_workflow.cmd "D:\audio\raw-voice.wav"
```

处理完成后，从以下目录获取最终音频：

```text
output\修音成品\
```

最终文件名为 `修音版_<原音频文件名>.wav` 和 `修音版_<原音频文件名>.mp3`。同一源文件再次处理时依次添加 `_01`、`_02`。内部阶段音频默认清理，用户不需要从多个中间 WAV 中选择。

仅在排查问题时使用以下参数保留内部阶段音频：

```bat
run_audio_workflow.cmd "D:\audio\raw-voice.wav" --keep-intermediate-audio
```

辅助脚本会优先把本地 `ffmpeg\bin` 加入 `PATH`，但 clean-source 包不会内置 FFmpeg；未手动放入时必须使用系统安装版本。

不安装可选模型时，确定性的 `final` 修音链仍可运行。需要启用 Respiro 候选时，执行：

```bat
.\.venv\Scripts\python.exe scripts\audio_cleanup.py setup-respiro
```

该命令会在本机准备 `tools\Respiro-en` 和 `tools\respiro-en.pt`；这些资产不会进入分享包。`doctor` 会分别报告 Python 版本、FFmpeg、Respiro 资产、Respiro 依赖、Respiro 权重加载和 DeepFilterNet 导入状态。模型不可用时会记录真实能力状态，不会阻止无模型确定性链运行。

默认 `auto` 不会无条件串行使用模型。它根据输入诊断生成固定安全候选：

- `respiro-breath-safe`：真实 Respiro 检测成功后，只做有限局部 duck。
- `deepfilter-denoise-safe`：仅在稳定底噪和可用噪声窗成立时运行，关闭 post-filter。
- `model-combined-review`：两个单模型候选分别通过后才生成，并且要求双 ASR。

报告中的 `model_applied`、`model_succeeded`、`fallback_used`、`benefit_score` 和 `harm_score` 是模型是否真实参与和能否交付的依据。fallback、无收益或伤害守门失败时，流程会自动回退 `natural`。
