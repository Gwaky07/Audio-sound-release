# Audio-sound v0.5.1

发布日期：2026-07-27

## 重要修复

- 取代 `v0.5.0` 的整段片头封印：首个源语音起点只用于限定检测范围，不再授权把 `0 → 语音起点` 整段压到静音地板。
- 只在母带后比较不可变原音和真实成品，且同时满足“源音安静、成品达到可闻电平、增益超过阈值”的具体片头窗口才允许处理。
- 所有候选窗口再次通过与 `source_active_hard_mute` 同级的源语音保护裁剪；检测到安静字头、字尾或内部活跃语音时会缩短或拒绝该窗口。
- 片头修复窗口明确设置 `guard_exclusion_allowed=false`，并且不再加入 `authorized_cleanup_windows`，内部质量守门与独立 pair guard 都必须直接检查真实成品。
- 母带前阶段现在明确记录为 `deferred_to_post_mastering_difference_evidence`，避免先改源前缀再用修改后的结果证明自己安全。
- 独立 pair evaluator 不再把所有 `.wav` 都假定为 PCM16；原音和成品会先分别规范化为临时 PCM16，因此 32-bit float WAV 等真实录音格式也能完成发布守门。

## 回归验证

- 覆盖只处理真实被放大片头软噪声的正向用例。
- 覆盖未被放大的安静片头必须逐样本保持不变。
- 覆盖立体声 frame 索引、长度和声道布局保持。
- 保留 `pre_speech_soft_noise_boosted` 独立发布阻断，不允许授权排除洗绿。
- `pytest` 固定只收集仓库 `tests/`，不再误递归进 `output/` 中旧分享包的同名测试模块。

## 分享包

GitHub Release 提供：

- `Audio-sound-release-v0.5.1-clean-source.zip`
- `Audio-sound-release-v0.5.1-clean-source.zip.sha256`

分享包从 `v0.5.1` 标签通过 `git archive` 生成，只包含 Git 跟踪的源码、Codex skill、配置、预设、文档、测试和启动脚本；不包含 FFmpeg、模型资产、虚拟环境、用户音频或本机生成文件。

在 Windows 已有 Python 3.11/3.10 与 FFmpeg 的前提下，接收者可直接解压并用 Codex 打开目录，把音频交给 Codex 处理，无需额外说明仓库规则。首次运行会创建本地 `.venv` 并安装依赖。

## 验证边界

- 自动验证覆盖代码测试、构建、源码归档安装、确定性 final 链、格式保持和发布守门。
- 未使用可感知音频播放工具时，不声明听感通过；真实用户音频仍应由接收者做最终听感确认。
