# Audio-sound v0.5.0

> 已由 `v0.5.1` 取代。`v0.5.0` 的片头封印会把首个语音前的整段区域作为单一处理窗口并加入内部授权排除，不建议继续分发或作为新安装版本使用。

发布日期：2026-07-27

## 主要更新

- 恢复证据驱动的片头轻噪声封印：以源音首个持续活跃语音起点为依据，在母带前后分别处理，防止 `loudnorm` 将不可闻气口抬成轰鸣。
- 使用与 `source_active_hard_mute` 相同的 `-35 dBFS` 活跃语音阈值，并保留约 45 ms hold pad，避免无证据硬静音或吞掉保留字头。
- 保留独立的 `pre_speech_soft_noise_boosted` 发布守门，且该检查直接比较真实成品，不允许授权排除窗口洗绿。
- 报告分别记录 `leading_pre_speech_seal` 与 `leading_pre_speech_seal_post`，便于核对母带前后实际封印状态和窗口。
- 新增片头软噪声封印回归测试，并继续满足 `process_media_file` 编排器不超过 600 行的架构门禁。

## 分享包

GitHub Release 提供：

- `Audio-sound-release-v0.5.0-clean-source.zip`
- `Audio-sound-release-v0.5.0-clean-source.zip.sha256`

分享包从发布标签通过 `git archive` 生成，只包含 Git 跟踪的源码、Codex skill、配置、预设、文档、测试和启动脚本。

## 验证边界

- 自动测试验证片头封印、守门检出能力、编排器架构约束和既有质量契约。
- 对真实用户音频仍须从原音重跑并完成独立 pair guard、`audio-verify-delivery --repair-intent` 与最终听感确认。
