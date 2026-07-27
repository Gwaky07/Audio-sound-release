# Audio-sound 发布包快速开始

发布包包含仓库源码、Codex skill、ffmpeg、Respiro-en 和 `respiro-en.pt`，不包含 Python 虚拟环境。

在接收端 Windows 机器上：

1. 安装 Python 3.11 或 3.10。Python 3.12 及以上版本不受封装模型支持。
2. 在本目录打开 PowerShell 或 Command Prompt。
3. 运行 `setup.cmd`。脚本优先使用 Python 3.11，其次使用 3.10，并创建仓库专用 `.venv`。
4. 运行 `check_runtime.cmd` 或 `doctor.cmd`。
5. 使用以下命令处理音频：

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

辅助脚本会把发布包的 `ffmpeg\bin` 加入 `PATH`，接收端不需要安装系统级 ffmpeg。

发布包 `.env` 指向：

- `tools\Respiro-en`
- `tools\respiro-en.pt`

`doctor` 会分别报告 Python 版本、Respiro 资产、Respiro 依赖、Respiro 权重加载和 DeepFilterNet 导入状态。任一项不可用时，重新运行 `setup.cmd` 并检查 Python 版本。

默认 `auto` 不会无条件串行使用模型。它根据输入诊断生成固定安全候选：

- `respiro-breath-safe`：真实 Respiro 检测成功后，只做有限局部 duck。
- `deepfilter-denoise-safe`：仅在稳定底噪和可用噪声窗成立时运行，关闭 post-filter。
- `model-combined-review`：两个单模型候选分别通过后才生成，并且要求双 ASR。

报告中的 `model_applied`、`model_succeeded`、`fallback_used`、`benefit_score` 和 `harm_score` 是模型是否真实参与和能否交付的依据。fallback、无收益或伤害守门失败时，流程会自动回退 `natural`。
