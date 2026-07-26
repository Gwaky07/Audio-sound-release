# 架构

## 目标与权威入口

本仓库提供可独立安装、可由 Agent 或 shell 调用的中文口播修音流程，不依赖其他本地仓库。

最终成品的权威入口是：

```powershell
run_audio_workflow.cmd "<input-audio>"
```

它固定使用仓库 `.venv` 并调用 `scripts/audio_skill_workflow.py run`。`scripts/audio_cleanup.py` / `audio_sound.cli` 是检查、安装、底层预设控制和排障入口，不是 Best Repair 默认交付路径。

## 分层

### 1. Agent 规约与 Skill

- `AGENTS.md`：能力契约、发布阻断规则和强制自查清单
- `.codex/skills/audio-sound/`：Codex/Agent 的意图映射与命令参考
- `CLAUDE.md`、`.claude/skills/audio-sound/`：Claude Code 的轻量转发入口

规约定义“允许交付什么”；代码中的 `release_blocked`、候选淘汰原因和独立验证负责执行这些规则。

### 2. 稳定工作流与候选编排

- `audio_sound/skill_workflow.py`：稳定的最终交付入口、报告和 WAV/MP3 命名
- `audio_sound/auto_workflow.py`：固定候选集合、候选评估与选择
- `audio_sound/agent_judgment.py`：`capability_plan`、`repair_scorecard` 和修音意图完成度

默认 `auto` 至少竞争：

1. `natural_baseline`
2. `final_repair_best`

Respiro、DeepFilterNet 和组合模型候选只在运行时真实可用且输入证据满足时加入。`natural_baseline` 只能作为安全回退，不能冒充修音意图完成。

### 3. 确定性处理与质量守门

- `audio_sound/pipeline.py`：媒体探测、格式保持、确定性修音、报告和 preservation guard
- `audio_sound/config.py`、`presets/*.json`：固定预设及受控覆盖

`final` 主链按证据执行：

1. 从原媒体提取并保持采样率、声道数、声道布局和时长
2. Respiro/辅助证据驱动的局部呼吸处理
3. 输入适用时执行保守降噪；清洁源跳过有害整段降噪
4. 清晰度 EQ、轻去齿音、温和压缩/稳量和响度标准化
5. 只在确认静音核心执行 `speech_safe_autogate`
6. 输出逐文件 JSON/Markdown 报告和批次报告

模型存在、可导入、真实执行和可交付是四个不同状态。fallback 或无可量化收益的模型候选必须淘汰。

### 4. 运行时与环境

- `audio_sound/bootstrap.py`
- `doctor.cmd` / `check_runtime.cmd`

运行时层验证 Python 3.10/3.11、FFmpeg、FFprobe、Respiro 权重与依赖以及 DeepFilterNet。最终处理必须使用仓库 `.venv`。

### 5. 独立发布验证

- `audio_sound/delivery_verifier.py`
- `scripts/evaluate_audio_pair.py`
- `audio_sound/release_audit.py`

内部报告 PASS 不是唯一依据。发布验证必须针对实际交付 WAV：

- 检查报告闭环与 `release_blocked=false`
- 不使用授权排除窗口，独立比较原音与最终 WAV
- 检查格式、时长、硬静音、频谱损失/增强和增益稳定
- 记录 WAV/MP3 SHA-256

语音删改任务还必须通过 `audio-audit-release` 的需求—切点—ASR 绑定审计。

## 强制边界

仓库无法阻止操作者在仓库外手写任意 FFmpeg 命令，因此“墙”由三层共同组成：

1. Agent 自动读取的规约和固定入口
2. 工作流内部 fail-closed 质量守门
3. CI 与独立 `audio-verify-delivery` 验证

绕开权威入口生成的文件，没有绑定报告和独立验证结果时，不得称为仓库最终交付。

## 打包与兼容

`audio_sound` 是主包。`scripts` 作为兼容包一并安装，确保历史 `narrow_onset_cleanup` 模块在 editable install、wheel 和 `git archive` 安装后仍可导入；新业务逻辑应继续向 `audio_sound` 收敛。

## 参考输入而非运行时依赖

早期 Feishu 笔记和 `audio-preprocess` 仓库只提供设计参考。本仓库拥有自己的运行时与交付契约，即使参考仓库不存在或发生变化，也必须能独立安装、测试和运行。
