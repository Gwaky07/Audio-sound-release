# Audio-sound v0.4.0

发布日期：2026-07-27

## 主要更新

- Windows CLI 统一启用 UTF-8 标准输出，中文路径和报告在 Codex、PowerShell 与重定向场景中更稳定。
- 新增 `pre_speech_soft_noise_boosted` 发布守门，阻止响度处理把片头轻噪声异常抬高。
- `final` 停顿清理同时使用源音和处理后音频的 FFmpeg 静音证据，并继续经过源语音保护，避免用无证据的片头硬封印误删短字头。
- 包版本统一升级到 `0.4.0`，并锁定已验证的 `torch==2.2.2` / `torchaudio==2.2.2` 安装组合。
- 发布文档明确 clean-source 包不内置 FFmpeg、模型仓库、模型权重、虚拟环境或用户媒体。
- CI 使用真实 `final` 工作流报告执行独立同源对比与 `audio-verify-delivery --repair-intent`，不再改写报告伪造 PASS。

## 分享包

GitHub Release 提供：

- `Audio-sound-release-v0.4.0-clean-source.zip`
- `Audio-sound-release-v0.4.0-clean-source.zip.sha256`

分享包从发布标签通过 `git archive` 生成，只包含 Git 跟踪的源码、Codex skill、配置、预设、文档、测试和启动脚本。

## 运行要求

- Windows
- Python 3.10 或 3.11
- 可从 `PATH` 调用的 FFmpeg / FFprobe，或手动放入 `ffmpeg\bin`
- 可选模型资产按 `RELEASE_QUICK_START.md` 在接收端单独准备

## 验证边界

- 发布门禁覆盖 Python 3.10 / 3.11 全量测试、隔离 wheel 安装、无模型确定性 `final` 链、格式保持、独立 pair guard 与 repair-intent 交付验证。
- Respiro 和 DeepFilterNet 只有在接收端 `doctor` 显示真实 ready 时才会作为候选；clean-source 包本身不包含模型资产。
- 自动化验证不等同于感知听觉确认。对真实用户音频交付时仍需按仓库契约完成最终试听与固定刻度频谱复核。
