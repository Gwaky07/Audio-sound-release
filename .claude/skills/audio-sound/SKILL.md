---
name: audio-sound
description: 按仓库固定能力契约处理、修复并验证中文口播音频。
---

# Audio-sound

执行前必须读取仓库根目录 `AGENTS.md`，它是能力边界、模式选择、质量守门与强制自查的唯一权威来源。

详细命令与验收标准复用：

- `.codex/skills/audio-sound/SKILL.md`
- `.codex/skills/audio-sound/references/commands.md`
- `.codex/skills/audio-sound/references/workflow.md`
- `.codex/skills/audio-sound/references/acceptance.md`

最终交付使用：

```powershell
run_audio_workflow.cmd "<input-audio>"
```

不得用底层 `audio-cleanup clean`、直接 FFmpeg 或未绑定报告的媒体替代最终交付路径。完成前必须运行独立 `verify_delivery.cmd`。
