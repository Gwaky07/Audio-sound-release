# Audio-sound Agent Instructions

This repository is a Windows-oriented audio cleanup workflow for Chinese spoken-word audio. When a user asks to process, clean, repair, or prepare audio in this repository, treat it as a request for a final usable delivery by default, not a lightweight preview.

## Default Workflow

Windows 最终交付优先使用固定到仓库 `.venv` 的入口：

```powershell
run_audio_workflow.cmd "<input-audio>"
```

只有明确传入 `.venv\Scripts\python.exe` 时，才直接运行 `python scripts/audio_skill_workflow.py run "<input-audio>"`。

The default workflow mode is `auto`. Under repair intent it pursues **Best Repair**: build a `natural` safety baseline, always compete the whitelist candidate `final_repair_best` (preset `final`), optionally compete evidence-gated model candidates, and auto-deliver only when hard gates pass. `natural` is only an incomplete fallback when enhancement fails gates — never a completed repair result.

Other modes:

- `natural` for mastering-only natural delivery without candidate search; use only when the user explicitly wants a conservative baseline
- `final` for a direct audible but bounded final repair with local breath and mouth-noise cleanup, gentle denoise, clarity EQ, de-essing, and voice leveling
- `voice-isolate` when noise windows from the same file are provided
- `review` when the user wants markers and review artifacts
- `reference-style` for the older breath-cleanup style when explicitly requested
- `reference-legacy` only for explicit legacy comparison or troubleshooting; it is not a safe default

`auto` 的 Agent 判断层只选择固定能力模块，不生成任意 DSP/FFmpeg 参数。修音意图下主候选是 `final_repair_best`；另可按缺陷证据和 `doctor` 能力竞争：`respiro-breath-safe` 只做高置信检测后的有限局部 duck，`deepfilter-denoise-safe` 关闭 post-filter，`model-combined-review` 仅在两个单模型候选分别通过后生成且始终要求双 ASR。模型未真实成功、使用 fallback、没有可量化收益或严格伤害守门失败时不得交付。不得自动启用门限、硬静音、强自适应降噪或 bridge cleanup。

When the user asks to process, repair, denoise, clean breath/mouth noise, improve clarity, or deliver a final usable version, treat it as Best Repair intent: run `auto` (which prefers `final_repair_best`) or `--mode final`. A delivery that only changes loudness or peak level, or that selects `natural_baseline`, is not a completed repair for that intent. Report must include `capability_plan`, `repair_scorecard`, and `delivery_incomplete_for_repair_intent`.

The `final` mode must remain bounded: use moderate breath attenuation, local mouth-click repair, conservative FFmpeg denoise, parametric clarity EQ, light de-essing, and gentle compression. Do not use a low-pass below 16 kHz or a dynamic gate merely to make the waveform look cleaner. Optional model stages may be used only when runtime availability is verified; their absence must not prevent the deterministic local final chain from completing.

Preserve the source sample rate, channel count, and stereo layout by default. Multi-channel audio may be downmixed temporarily for diagnostics, but processing and final WAV/MP3 delivery must keep the source format. Treat any channel or sample-rate change as a release-blocking quality-guard failure.

质量守门必须阻止源活跃语音硬静音、短时增益不稳定、活跃语音衰减和增益归一化后的高频清晰度损失。单模型候选无 ASR 时必须通过更严格的衰减、频谱和局部相关性门限；双模型组合仍要求可绑定的双 ASR。任何失败都回退到已通过守门的 `natural` 基线。

## 用户意图与模式选择防错规则

- 用户说“处理音频”“修音”“最终版”“最终好效果”“修好后给我”“要听得出改善”“降噪”“清晰度”“呼吸音”“气口”“口水音”“去齿音”“稳量”时，默认 Best Repair：优先 `auto`（竞争 `final_repair_best`）或直接 `--mode final`。不得把回退的 `natural` 当作这类请求的完成结果。
- 用户只要求保守母带、尽量不改变原声或明确要求自然基线时，才直接使用 `natural`。
- 交付前必须检查 `applied_stages`、`preset_name`、`auto_selection.candidate_id`、`capability_plan` 和 `repair_scorecard`。如果只有高通、轻压缩、响度统一和导出，没有降噪、EQ、呼吸/口水音修复或其他用户点名阶段，则只能标记为“自然基线”，不能称为“最终修复版”或“最终好效果”。
- 如果修音意图下 `auto` 最终选择 `natural_baseline`，必须设置 `delivery_incomplete_for_repair_intent=true`，明确说明增强没有应用，并继续从原音运行 `final` 或排查守门失败原因；不得直接把自然版交给用户结束任务。
- 用户反馈“和原音没区别”“没有处理”时，立即使之前的完成结论失效。先列出实际应用和未应用的阶段，再从原始媒体重做正确模式；旧成品只能作为失败参考，禁止在旧成品上继续加工。
- 不得为了证明仓库“有能力”而无条件串联所有模型。Respiro、DeepFilterNet 和确定性修复阶段按缺陷证据启用；不适用的模型必须说明未触发原因，但用户点名的可安全确定性阶段不能被静默省略。
- `final` 也必须按输入证据避免已知伤害：当 `stationary_noise=false` 且 `estimated_snr_db>=35` 时，自动关闭整段 `afftdn` secondary denoise，并记录 `input_adaptations=["skip_secondary_denoise_clean_source"]`。这类清洁源仍保留 Respiro 局部呼吸处理、清晰度 EQ、去齿音和稳量，不能用有害整段降噪凑处理阶段。
- `mouth_declick_sensitivity=0` 必须表示完全关闭全局 mouth-declick，不能仍扫描并插值整段语音。未确认口水音/爆点窗口时，`final` 默认关闭全局 mouth-declick；确认后只允许窄窗口修复。
- `final` 的呼吸清理必须使用闭环：保留不可变 `raw_wav`，分别记录 Respiro、辅助停顿边缘检测和噪声型频谱证据；只处理位于语音起点之前且呈噪声型的窗口。首轮按相邻停顿底噪自适应衰减，母带后复检并做最多两次窄窗口补处理，最后仍高于局部底噪 3 dB 的确认残留以 `confirmed_breath_residual_after_retry` 阻止交付。
- “运行了 Respiro”或“窗口已衰减”不等于呼吸清理完成。报告必须包含 `breath_cleanup.status=PASS`、空的 `final_residual_windows`、实际处理窗口及授权排除窗口；残留检测失败或未验证时不得宣称干净。
- 呼吸窗口允许从 preservation 频谱/增益比较中排除，但仅限报告中记录且通过语音保护授权的窗口；其他所有语音仍必须通过吞字、相关性、频谱和增益稳定守门。不得用排除窗口掩盖整段伤害。
- 对比呼吸清理前后频谱时必须使用相同 dB 范围和颜色刻度。紫色/黑色只代表较低能量，不能仅凭自动缩放后的颜色宣称清理完成。
- 用户明确要求“中间过渡不要有那么多紫色”时，`final` 必须启用语音安全 AutoGate 等价的停顿过渡清理：只处理 FFmpeg 静音证据内部、通过原音活跃语音保护的窗口；句间两侧约 45 ms hold pad，片头/片尾可用约 25 ms 更紧 pad；确认无声核心目标为绝对静音地板 `silence_floor_dbfs=-96`（不是底噪-6），短 fade 约 8–20 ms，首轮最多 48 dB、残留二次最多 24 dB、终轮最多 36 dB。不得启用全局 Dynamics Gate / `agate` 或无证据整段硬静音。
- 停顿过渡清理完成必须同时满足 `pause_cleanup.status=PASS`、`pause_cleanup.mode=speech_safe_autogate`、`final_residual_windows` 为空，并在固定刻度局部频谱中显示中间接近黑色静音地板、两侧语音频谱连续。相对 `-96` 目标仍高出超过 3 dB 的确认残留以 `confirmed_pause_residual_after_cleanup` 阻止交付。
- 每次交付必须向用户说明实际改了什么，包括降噪、EQ、呼吸处理、去齿音、压缩和响度阶段，以及处理前后的 LUFS、true peak、格式和质量守门结果。没有应用的阶段也必须明确列出。
- “脚本成功”“质量守门 PASS”和“用户能听出改善”是三件不同的事。质量守门只证明没有检测到既定伤害；当用户要求明显效果时，还必须确认实际修复阶段已应用并取得可量化收益。
- 未使用音频播放或其他可感知听觉工具时，不得说“我听了”“听起来更好”或“听感通过”。只能说波形、频谱、模型状态和指标通过，并请用户进行最终听感确认。

## 自动模型能力与防错规则

本仓库已经具备 Respiro-en 与 DeepFilterNet 自动候选能力。后续 Agent 不得把模型资产存在、依赖可导入、模型真实执行和候选可交付混为一谈。

### 运行环境

- 所有处理工作流必须使用仓库 `.venv\Scripts\python.exe`，不得使用系统 Python 3.12、3.13 或 3.14 运行模型。
- `setup.cmd` 只允许 Python 3.11 或 3.10，优先选择 3.11；现有 `.venv` 版本不受支持时必须停止并要求重建。
- 处理前必须运行 `doctor.cmd` 或 `check_runtime.cmd`。分别检查：Python 版本、Respiro 仓库与权重、Respiro 依赖、权重可加载、DeepFilterNet 可导入、ffmpeg 和 ffprobe。
- `respiro_en.ready=true` 才表示 Respiro 可作为候选；`deepfilternet.ready=true` 才表示 DeepFilterNet 可作为候选。只有 `tools/` 文件存在不代表模型可运行。

### 固定模型候选

- `respiro_breath_safe`：仅在输入存在停顿/呼吸候选证据且 Respiro ready 时运行。必须真实得到 `respiro_detection_mode=respiro` 和 `respiro_succeeded=true`；最多做 3 dB 有淡入淡出的局部 duck，不得硬静音。
- `deepfilter_denoise_safe`：仅在稳定底噪、可用同源噪声窗、适用 SNR 和 DeepFilterNet ready 同时成立时运行。必须关闭 post-filter，不得叠加强低通、门限或多层强降噪。
- `model_combined_review`：只有上述两个单模型候选分别通过全部硬门后才允许生成，并且必须有两路独立、与候选媒体 SHA-256 绑定的 ASR 证据。缺少双 ASR 时不得交付。
- 模型运行时不可用但诊断满足触发条件时，报告必须记录 `runtime_unavailable`，继续使用安全候选，不得让整个 `auto` 流程失败。

### 模型候选交付门槛

- 报告必须区分 `model_applied`、`model_succeeded`、`fallback_used`、`benefit_score`、`harm_score` 和 `benefit_metrics`。
- fallback 只能作为排障信息，不能冒充模型成功；`fallback_used=true` 的模型候选必须淘汰。
- 模型候选必须取得可量化收益：DeepFilterNet 看噪声/SNR 改善，Respiro 看真实呼吸窗口及窗口衰减。只“调用了模型”但没有收益时，以 `no_measurable_model_benefit` 淘汰。
- 单模型无 ASR 时采用严格门限：最差活跃语音相对衰减不得低于 -3 dB、短时增益跨度不得超过 6 dB、活跃电平相关性不得低于 0.92；2–8 kHz 损失不得超过 2 dB，8–12 kHz 损失不得超过 3 dB。
- 任何格式变化、时长变化、削波增加、源活跃硬静音、吞字风险、频谱清晰度损失、短时增益不稳或模型无收益都阻止模型候选交付。
- 质量守门必须同时限制高频增加：增益归一化后，2–12 kHz 任一频带增益超过 3 dB，或 12–16 kHz 增益超过 3 dB，均以 `spectral_harshness_increased` 阻止交付。`PASS` 不得只检查“声音变闷”，还必须检查“刺音/齿音被放大”。
- Codex 只能在通过硬门的固定候选中选择最高净收益结果，不得因为“模型更高级”而偏向模型；全部增强失败时必须回退通过守门的 `natural`。

### 真实性验证

- 不得仅凭命令退出码宣称模型已使用。Respiro 必须检查 `respiro_attempted=true`、`respiro_succeeded=true`、真实命令和权重路径；DeepFilterNet 必须检查 `stage_status.deepfilternet.applied=true`。
- 新机器完成 `setup.cmd` 后，先用短音频分别跑 Respiro 和 DeepFilterNet smoke test，再运行 `auto` 回归。
- 回归必须用 `scripts/evaluate_audio_pair.py` 比较源音和最终交付，确认采样率、声道、时长保持，且没有硬静音、高频清晰度损失和异常增益波动。
- 未进行可感知音频播放时，不得声称“已听过”或“听感通过”；只能准确报告模型状态、ASR、波形、频谱、格式和质量守门证据。

The adaptive controller may select only `baseline`, `leveling_gentle`, `noise_review`, or `manual_review`. Codex/GPT may interpret diagnostics and choose among those profiles, but it must not generate arbitrary DSP/FFmpeg parameters or bypass `allow_destructive_cleanup=false`. `noise_review` records evidence for review; it does not authorize automatic strong denoise.

Do not use `scripts/audio_cleanup.py clean` as the default final-delivery path. Keep it for setup, inspection, lower-level preset control, compatibility, and troubleshooting.

To compare a source/processed pair without reprocessing:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_audio_pair.py --source "<source>" --processed "<processed>" --output "scratch/pair-report.json"
```

Bind final media to report gates and an independent no-exclusion comparison before calling a repair delivery complete:

```powershell
verify_delivery.cmd --source "<source>" --final-wav "<delivered.wav>" --final-mp3 "<delivered.mp3>" --report "<audio_process_report.json|skill-file-report.json>" --output "scratch/delivery-verification.json" --repair-intent
```

## Runtime Setup

Before processing audio on a fresh machine, check the runtime:

```powershell
python scripts/audio_cleanup.py doctor
```

If dependencies are missing, run:

```powershell
setup.cmd
```

For release packages that include `tools/Respiro-en`, `tools/respiro-en.pt`, and `ffmpeg/`, use the package-local `.env`. If Respiro assets are not present, run:

```powershell
python scripts/audio_cleanup.py setup-respiro
```

Do not claim Respiro-en, DeepFilterNet, or SpectraMini were used unless `doctor` or the workflow report shows they are actually available. If the workflow falls back to heuristic processing, say so.

## Delivery Rules

- Final user-facing audio must be delivered under `output/修音成品/`.
- Preserve the original Chinese filename with the naming rule `修音版_<原音频文件名>.wav|mp3`.
- If the final name already exists, use `_01`, `_02`, and so on. Do not rely on timestamp folders alone.
- Remove internal WAV/MP3 stage files by default after reports and spectrograms are written. Use `--keep-intermediate-audio` only for troubleshooting.
- Produce final WAV and MP3 outputs.
- Generate and review spectrogram evidence for strict cleanup work.
- If a specific timestamp still has breath, saliva noise, pause residue, or thin artifacts, inspect that local region first and then use exact narrow-window repair.
- Avoid broad edits that can swallow speech onsets or restore isolated noise in silence.
- DeepFilterNet dropout repair may only restore short gaps inside continuous speech; it must not restore isolated noise in leading silence or pause regions.
- After exact repair, re-export MP3 and re-check loudness.

## Agent Authority And Entrypoints

`AGENTS.md` 是本仓库唯一的 Agent 能力契约与发布自查权威来源。Claude / Codex / Cursor Agent 都必须遵守本文件；冲突时以本文件为准。

- Claude 入口：根目录 `CLAUDE.md`（`@AGENTS.md`）与 `.claude/skills/audio-sound/SKILL.md`（薄指针，不得另写一套交付规则）
- Codex 详细 skill：`.codex/skills/audio-sound/SKILL.md` 及其 `references/`
- 架构说明：`docs/architecture.md` 必须与本契约一致；若文档写“默认 mono 提取 / 永远 DeepFilterNet”等过时说法，以本文件与代码为准并应修正文档
- 不得用直接 FFmpeg、任意手写 DSP、或 `audio-cleanup clean` 冒充最终交付；最终交付路径是 `run_audio_workflow.cmd` / `audio-skill-workflow`
- 规划或冒烟可用 `audio-skill-workflow run <input> --mode auto --dry-run`；`--dry-run` 只验证模式解析与候选规划（必须含 `natural_baseline` 与 `final_repair_best`），**不等于**已完成修音

## Repository Audio Capability Contract

以下能力是本仓库对“修音、处理音频、最终版、Best Repair”请求的固定能力契约，不得仅凭命令成功、模型存在或报告生成就声称已完成：

1. `final` 确定性主链必须能够从原音完成：Respiro/辅助证据驱动的局部呼吸处理、按输入证据决定的保守降噪、清晰度 EQ、轻去齿音、温和压缩/稳量、响度标准化、语音安全停顿清理以及 WAV/MP3 交付。清洁源可以跳过整段降噪，但不得静默省略用户点名且适用的安全阶段。
2. `auto` 必须至少竞争 `natural_baseline` 与 `final_repair_best`；模型候选只能按固定能力模块和缺陷证据加入。`AUTO_CANDIDATE_RECIPES` 中每个 `preset_name` 必须对应仓库 `presets/*.json` 真实文件；测试与 CI 必须覆盖该契约。`natural_baseline` 不是修音意图的完成结果。
3. 语音安全 AutoGate 等价能力只能作用于 FFmpeg 静音证据与原音语音保护共同授权的核心窗口。默认目标为 `silence_floor_dbfs=-96`，句间 hold pad 约 45 ms，片头/片尾约 25 ms，使用短 fade；`allow_bridge_windows=false`，禁止无 pad bridge、全局 `agate`、无证据硬静音或以长 crossfade 掩盖残音。残留评估窗被 trim 为空时必须 fail-closed，不得假 PASS。
4. 源语音保护的敏感度不得低于 preservation hard-mute guard。当前源活跃硬静音阈值为 `-35 dBFS`；低于代表性 speech level 但仍达到该阈值的安静字头、字尾和轻声同样必须保护。清理授权必须双向裁剪首尾活跃语音，并拒绝窗内残留活跃段；不能先处理再用“授权窗口排除”隐藏。
5. 多声道音频的时间计算必须按 frame 而不是交错 sample 计数：`duration = len(samples) / (sample_rate * channels)`；秒到 sample 索引也必须乘以 `channels`。片头/片尾检测、bridge 测量、局部窗口和淡化均须覆盖单声道与立体声回归测试。
6. 默认保持原采样率、声道数、声道布局和时长；任何意外变化都是发布阻断问题。最终必须同时生成 WAV 与 MP3，报告处理前后 LUFS、true peak、格式、实际应用阶段和未应用阶段。
7. 报告中的授权排除窗口只用于避免把已确认的呼吸/停顿清理误判为伤害，不能替代对完整源音和最终成品的独立 preservation 检查。内部 `quality_guard=PASS` 与独立检查冲突时，结果必须按 FAIL 处理。
8. 窄窗 onset / 参考风格清理逻辑住在 `audio_sound/narrow_onset_cleanup.py`；`scripts/narrow_onset_cleanup.py` 只是 CLI 薄包装。业务代码必须 `from audio_sound.narrow_onset_cleanup import ...` 或相对导入 `.narrow_onset_cleanup`，不得再把核心算法绑在 `scripts.*` 上。
9. `detect_runtime()` 未传 `repo_root` 时必须默认仓库 `PROJECT_ROOT`，不得依赖当前工作目录去找 `tools/`。
10. 独立发布验证是强制能力：`audio_sound/delivery_verifier.py` / `verify_delivery.cmd` / `audio-verify-delivery`。修音意图下必须加 `--repair-intent`，以阻断 `delivery_incomplete_for_repair_intent` 与非 PASS 的 `repair_scorecard`。接受 `audio_process_report.json`、`skill-file-report.json`（含 `core_report`）或 workflow summary；报告内 `deliverables.wav|mp3` 若存在，必须与 CLI 传入的最终媒体路径解析为同一文件。
11. 通用媒体能力只允许在 `audio_sound/media_utils.py` 维护一份：PCM16 WAV 读取、秒数格式化、SHA-256、MP3 命令与导出不得在 workflow、pipeline、release audit、delivery verifier 或 segment removal 中复制实现。
12. `process_media_file()` 是主编排器；停顿清理、呼吸残留闭环和命令执行记录必须由独立 helper 承担。架构测试将该函数限制在 600 行以内，不得再次把完整算法堆回单个千行函数。

## Package, Runtime, And CI Enforcement

这些规则把“文档里写了”变成“安装与 CI 会挡住”的约束；Agent 改代码时不得绕开：

1. **可安装包**：`pyproject.toml` 只打包 `audio_sound`，提供 `audio-cleanup`、`audio-skill-workflow`、`audio-remove-segments`、`audio-audit-release`、`audio-verify-delivery`。`setup.cmd` 必须 `pip install --editable .` 到仓库 `.venv`。wheel 必须内置 `audio_sound/presets/*.json`；仓库根 `presets/*.json` 是可编辑源，两份内容必须由测试逐字节校验，禁止漂移。仓库 `scripts/` 只是本地 CLI 薄入口，**禁止**再注册为顶层 setuptools 包名 `scripts`。
2. **Python**：仅 3.10 / 3.11；处理与模型必须用 `.venv\Scripts\python.exe`。
3. **源码安装冒烟**：CI 必须 `git archive` 后构建 wheel，在检出目录外新建 venv 安装，并至少导入 `audio_sound.auto_workflow`、`audio_sound.agent_judgment`、`audio_sound.skill_workflow`、`audio_sound.narrow_onset_cleanup`、`audio_sound.exact_window_cleanup`、`audio_sound.pair_evaluation`、`audio_sound.media_utils`、`audio_sound.delivery_verifier`，确认顶层 `scripts` 包不存在，验证 packaged presets，再跑 `audio-skill-workflow ... --dry-run` 与 `audio-verify-delivery --help`。所有含真实逻辑的模块都必须住在 `audio_sound/`；`scripts/*.py` 只能是薄 shim。
4. **测试门禁**：`.github/workflows/ci.yml` 必须在 Python 3.10 / 3.11 跑全量 `pytest`，并在隔离 wheel 环境用合成口播样本真实执行一次无模型 `final` 确定性链，生成 WAV/MP3、确认格式保持，再做独立 pair guard PASS 与 `audio-verify-delivery` 冒烟。不得把完整 `setup.cmd`（含 torch / Respiro 资产）当作 CI 阻断步骤。
5. **运行时诊断门禁**：CI 必须解析 `doctor`，确认 Python 受支持且 Respiro / DeepFilterNet capability state 字段存在。缺少 bundled `tools/`、仓库内 `ffmpeg/`、模型权重时，模型 `ready=false` 是可接受事实，但不得伪造 ready，也不得让无模型单元测试失败。
6. **对比与验证模块**：`scripts/evaluate_audio_pair.py` 是薄入口，核心在 `audio_sound/pair_evaluation.py`；独立无排除对比与 `verify_delivery` 不得传授权排除窗口来“洗绿”报告。`quality_guard.release_blocked` 缺失时必须 fail-closed 视为 `True`；测试与 CI 必须锁住该默认。
7. **禁止提交**：`.venv/`、`.env`、`tools/`、`output/`、`scratch/`、`.omx/`、`__pycache__/`、`*.egg-info/`；分享前可跑 `python scripts/audio_cleanup.py clean-repo`。

## Enforcement Map

本节是规约与代码之间的对账表。**散文与代码冲突时以代码为准**：下表左列的规则已由右列的测试或 CI 步骤机械强制，改动实现前先看会挡住你的是哪一条。规约新增条目时应先问“这条能不能变成一个断言”。

| 规则 | 强制点 |
| --- | --- |
| 守门必须能真的检出损伤（非仅声称） | `tests/test_guard_self_audit.py`：向合成口播注入低通、高频抬升、硬静音、时长截断、削波、活跃衰减、增益抖动、格式变化，逐一断言对应失败码 |
| 守门不得误报（响度变化 / 55 Hz 高通 / 预算内轻微 EQ） | `tests/test_guard_self_audit.py::GuardBaselineTests`、`GuardBlindSpotBoundaryTests` |
| 守门失败码集合不得被悄悄删减 | `tests/test_enforcement_contract.py::GuardFailureCodeContractTests` |
| `quality_guard.release_blocked` 缺失必须 fail-closed | `tests/test_enforcement_contract.py::FailClosedDefaultContractTests`、`tests/test_delivery_verifier.py` |
| `status` / `release_blocked` / `failures` 三者不得自相矛盾 | `tests/test_enforcement_contract.py::GuardDirectionContractTests` |
| 交付验证必须真的拦住坏报告（不只放行好报告） | `.github/workflows/ci.yml` → `Prove the delivery gate actually blocks`：对刚通过的报告做 7 种变异，断言退出码非 0、`release_blocked=true` 且命中预期失败码；另有缺失交付物用例 |
| 模型跑了 ≠ 模型有收益 | `tests/test_auto_workflow.py`；`model_fallback_used`、`no_measurable_model_benefit`、`asr_evidence_missing`、`unscoped_model_stage` 由 `tests/test_enforcement_contract.py` 锁定引用 |
| 真实逻辑必须在包内，`scripts/` 只能是薄 shim | `tests/test_enforcement_contract.py::PackagingContractTests`；CI 包外导入步骤 |
| 顶层 `scripts` 包不得复活 | `tests/test_enforcement_contract.py::PackagingContractTests`；CI wheel 内容断言 |
| 两份 presets 不得漂移 | `tests/test_config.py::test_packaged_presets_match_repository_presets` |
| `auto` 必须竞争 `natural_baseline` 与 `final_repair_best` | `tests/test_auto_workflow.py`；CI `--dry-run` 冒烟 |
| 无模型 `final` 确定性链必须能真实出片 | CI `Run deterministic final chain from installed wheel`（真实 ffmpeg + WAV/MP3 + 格式保持 + 独立 pair guard） |

下列要求**无法**由 CI 伪造或替代，仍必须逐次人工执行，见 `External Verification Boundaries`：真实模型执行证据、感知听觉确认、固定刻度频谱目视复核、源文档语义与红字逐条对账。

## Mandatory Audio Self-Audit Before Completion

每次修改音频算法、预设、质量守门或交付流程，以及每次生成最终音频后，Agent 必须主动完成以下自查；不得等待用户提醒：

1. **代码自查**：审阅本次完整 diff，重点检查单位、阈值方向、声道/帧索引、片头片尾边界、fade/hold 范围、回退路径、报告字段和授权排除窗口。运行受影响测试、全量测试与 linter；新增行为必须有正常、边界和回归测试。
2. **运行时真实性**：处理前运行 `doctor.cmd` 或 `check_runtime.cmd`；处理后核对 `stage_status`、真实模型成功字段、fallback、`input_adaptations`、`preset_name`、候选 ID 和执行命令。模型未真实成功不得写成“已使用”。
3. **从原音重建**：算法或守门修复后必须从原始媒体重跑，不得在先前成品上叠加修复。旧成品若已被新证据判定有误，必须立即标记为失败参考，不得继续称为最终版。
4. **报告闭环**：检查 `breath_cleanup.status=PASS`、`pause_cleanup.status=PASS`、`pause_cleanup.mode=speech_safe_autogate`、`allow_bridge_windows=false`、两类 `final_residual_windows` 为空、无 `empty_assessment_windows`、`quality_guard.status=PASS` 且 `release_blocked=false`。任一字段缺失、失败或互相矛盾都阻止交付。
5. **独立无排除复核**：必须对原音和最终 WAV 运行 `scripts/evaluate_audio_pair.py`（或等价 `audio_sound.pair_evaluation`），且不传授权排除窗口；最终再运行 `verify_delivery.cmd`（修音意图必须带 `--repair-intent`），将源文件、实际交付 WAV/MP3、报告（core / skill-file-report / summary 均可）、独立对比和 SHA-256 绑定为发布验证清单。两者必须没有 `source_active_hard_mute`、`spectral_clarity_lost`、`spectral_harshness_increased`、格式变化或异常增益波动，验证清单必须 `status=PASS` 且 `release_blocked=false`。内部守门 PASS 但独立检查 FAIL 时，撤销完成结论、定位重叠窗口、修复代码并从原音重跑。
6. **缺陷窗口反查**：对独立检查发现的每个硬静音、频谱伤害或增益异常窗口，反查 `breath_cleanup`、`pause_cleanup` 的 first/second/final pass 和 `window_evidence`，确认是否误授权或多阶段叠加。不得只调松守门阈值使报告变绿。
7. **边界与多声道检查**：明确核看片头、片尾和至少一个句间窗口；验证实际 pad、请求衰减、最终残留及两侧语音连续。立体声输入必须确认片尾窗口使用真实 frame duration，而不是双倍 sample duration。
8. **固定刻度频谱复核**：输入与最终成品必须使用相同 dB 范围、颜色刻度、采样区间。严格清理至少检查开头 0–30 秒概览和用户点名局部窗口；黑色只能证明低能量，仍须结合原音活跃语音保护与独立 hard-mute 检查。
9. **最终媒体核验**：检查最终 WAV/MP3 均存在且名称正确，记录 SHA-256、时长、采样率、声道、LUFS、true peak、削波和质量守门结果。自查对象必须是实际交付文件，不能只检查中间 WAV。
10. **真实陈述**：只有使用可感知播放工具时才能宣称完成听感试听；否则必须明确证据仅来自代码测试、报告、波形、频谱、ASR 和指标，并要求用户做最终听感确认。

上述任一步失败都意味着任务尚未完成。Agent 必须继续修复、自测、从原音重跑并重新执行整套自查，直到所有发布门槛通过或明确报告无法解决的阻断原因。

## Source-of-Truth Rebuild Rule

When the user asks to redo the edit from the beginning, discard the current delivery as an editing source and rebuild directly from the original media. A previous delivery may be used only as negative evidence for locating defects; do not cut, denoise, or remux from it.

- Re-fetch the latest source document before building the cut list. Treat crossed-out text and checklist items with `done=true` as explicitly skipped unless the user reactivates them.
- Keep all source-document timecodes in one source-timeline cut table. Do not combine source timecodes with timestamps mapped from an earlier edited delivery; this can delete the same rerecorded passage twice.
- Review the complete opening and ending for semantic continuity. The final opening must not retain the previous lesson, and the ending must preserve the complete intended farewell.
- Scan the complete exported WAV for clipped samples, true peak, abnormal transients, and impulse noise. Checking only edited seams is insufficient when the source recording itself clips.
- Before using donor audio, verify its waveform, spectrum, clipped-sample count, transient noise, and surrounding semantics. Never replace a defect with donor audio that contains the same harsh noise or belongs to a different sentence context.
- If an animation or retained visual section contains discarded-take speech, preserve the video frames but mute or replace only the unwanted audio window. A clean visual splice does not prove the audio is clean.

## Spoken-Segment Boundary Rule

When the request physically removes a spoken word, filler, sentence, or time range—or mentions boundary residue, seam noise, a stutter-like hitch, or a next word suddenly joining—do not treat it as ordinary denoise work.

1. Read `.codex/skills/audio-sound/references/spoken-segment-boundaries.md` before editing.
2. Use `scripts/remove_spoken_segments.py` for the physical cut. Treat document timecodes as candidates; when a local example clip is supplied, first align its local timeline instead of reusing a long-video timestamp verbatim.
3. Inspect the original and any problematic prior edit around every seam with local waveform/spectrogram evidence. Check the removal report fields `requested_cuts`, `cuts`, `crossfade_frames`, and `seam_pause_frames`.
4. Use the default `--boundary-search-ms 80` and `--crossfade-ms 12` for ordinary joins. Do not increase the crossfade duration to hide residual phonemes.
5. If the result still sounds like the next sentence or word suddenly joins, rerun that reviewed seam with `--seam-pause-ms 60` to `100` (start at `80`). This uses separate fade-out/fade-in and holds the prior video frame for the same duration.
6. Before delivery, re-export WAV/MP3/MP4, verify the WAV/MP4 duration delta is below one video frame, and review a `0.3–0.5s` local spectrogram/waveform on both sides of the seam.

Never claim a spoken-segment removal is complete solely because the script succeeded or the video plays. Continue if a residual tail, abrupt silence gap, click, swallowed onset, or unnaturally sudden semantic continuation remains.

## Boundary Residue / Seam Noise Operating Checklist

Use this checklist for issues described as `边界残留`, `接缝杂音`, `没切干净`, `卡一下`, `突然接上下一句`, repeated syllables such as `法法军`, or any red/yellow document mark around a cut.

1. Rebuild from the original source when the user asks to redo the edit. Previous MP4/WAV/MP3 deliveries are defect references only; never use them as the new editing source.
2. Keep one authoritative source-timeline cut table. Do not mix source-document timestamps with final-video timestamps from an older export; map final-video defects back to the source before cutting.
3. Treat every document timestamp as a candidate, not proof. Confirm the exact phoneme boundary by local listening plus waveform and spectrogram inspection before cutting.
4. For repeated homophones, stutters, or adjacent single-character words, ASR is only a hint. A pass requires local listening and waveform/spectrogram comparison against the source semantic boundary.
5. Remove residual speech with narrow physical cuts. Do not hide leftover phonemes with a long crossfade, broad denoise, or silence trim that can swallow the next retained word.
6. If the next word still feels too sudden after a clean cut, add `--seam-pause-ms 60` to `100` (start at `80`) for that reviewed seam instead of increasing `--crossfade-ms`.
7. After any correction, regenerate all deliverables and repeat the complete post-export review. Rechecking only the corrected timestamp is not acceptable.
8. Before uploading or marking complete, keep only the latest correct Feishu deliverable and remove older wrong videos/audio/reports from the document.

## Mandatory Post-Export Full Review

Every spoken-word edit must receive a second, full review of the exported final MP4/WAV—not only the edited windows and not an intermediate stage.

1. Re-fetch or re-read the latest source document immediately before final review. Treat newly added yellow/red highlights, comments, and final-video timestamps as the current source of truth.
2. Build a line-by-line checklist for every requested deletion, highlighted seam, audio repair, and visual-only requirement. Record the final-timeline timestamp and pass/fail result for each item.
3. Transcribe the complete exported audio with two independent ASR passes when local models are available. Scan for forbidden phrases, leftover red-text words, adjacent repeated characters/words, and semantically duplicated re-recorded passages. Exact-string absence alone is not sufficient because ASR wording may vary. ASR can collapse adjacent homophones or repeated characters into one token, so such cases pass only after local listening plus waveform and spectrogram comparison against the source semantic boundary.
4. Review every cut seam on the final timeline with local waveform/spectrogram evidence and speech context. Explicitly check residual phonemes, clicks, swallowed onsets, abrupt joins, and tail syllables such as “么呀”.
5. Review the complete beginning and ending, plus every visual splice, frame hold, delayed image, and retained page animation. Inspect video frame-by-frame around visual joins; audio passing does not imply video passing.
6. After any correction, re-export all deliverables and repeat the entire full review from step 1. Do not only recheck the corrected timestamp.
7. When a later cut changes the final timeline, remap every previously approved narrow repair window from the source timeline (or from the prior cut table) before re-exporting. Never reuse old final-timeline timestamps unchanged; first verify the remapped window against the rebuilt WAV, then re-run clipped-sample, true-peak, and local spectrum checks.
8. Do not upload, replace, or label a Feishu deliverable as complete until the second full review report shows every checklist item passed and the WAV/MP4 duration delta remains below one video frame.
9. If the user reports that an uploaded version still has the same defect, treat the previous validation as failed. Re-download the exact uploaded media, re-fetch the latest document, verify against the final-video timeline, and do not rely on local intermediate files or summary-only reports.
10. A repair report is not sufficient unless it records the actual repair windows, final-timeline timestamps, local ASR snippets, waveform/spectrogram comparison, duration delta, and whether old incorrect Feishu media tokens were removed after the replacement upload.

## Truthful Reporting And Evidence Integrity

Truthful status reporting is a release-blocking requirement. Never convert an assumption, script result, model inference, prior report, or agent summary into a claim that the user's requested edit is complete.

1. Do not claim `removed`, `processed`, `fixed`, `clean`, `PASS`, or equivalent solely because a command succeeded, a cut exists in a configuration or report, ASR omitted a phrase, or a previous agent marked the item complete.
2. Judge completion only against the latest source document and the exact final media intended for delivery. After upload, download the uploaded media and verify that exact file before claiming the Feishu delivery is correct.
3. If the user reports that a defect remains, immediately invalidate the conflicting prior PASS result. Treat the report as new failure evidence, re-download the uploaded media, and investigate from that media instead of defending the old report.
4. Every conclusion must be labeled by its real evidence level: `DIRECTLY VERIFIED`, `MODEL-INFERRED`, or `UNVERIFIED`. Any unresolved doubt is `FAIL` or `UNVERIFIED`, never PASS.
5. Do not say `listened`, `auditioned`, `heard`, or `passed listening review` unless the agent actually used a tool that provides perceptible audio playback or an equivalent direct auditory review. Otherwise state the precise evidence used, such as ASR, waveform, spectrum, physical cut boundaries, frame review, or duration measurements, and disclose the missing auditory check.
6. For repeated, adjacent, or homophonic words and rerecorded passages, prove which occurrence was removed and which occurrence was retained. Record the source context on both sides, the source cut, the mapped final timestamp, and evidence for the retained take.
7. Reports must identify the source-document revision, final-media SHA-256 hashes, requested and actual source cuts, final-timeline seam positions, actual local ASR snippets, waveform/spectrum evidence, WAV/MP4 duration delta, and the removal status of superseded Feishu media tokens.
8. Do not upload or remove old deliverables until every requirement has been rechecked on the actual final export. Upload the replacement first, download and hash-verify it, then remove superseded blocks and re-fetch the document to prove the old tokens are gone.
9. User trust takes precedence over preserving a previous completion claim. When evidence conflicts, state the error plainly, withdraw the unsupported claim, and continue verification or repair without minimizing the discrepancy.
10. Build and validate an explicit requirement-to-cut mapping. Never assume requirement `N` maps to cut `N`: extra source-only cuts such as removing a previous-lesson opening shift every later cut index. For each mapping, verify the document text, cut label, source word context, and final seam before using it in an audit report.
11. Treat every red-text run as an individual release-blocking target. For adjacent or repeated wording, the audit must show the exact removed source tokens and the exact retained tokens; a generic range or an ASR phrase-absence check is insufficient.
12. If a Feishu section contains an unsupported `PASS`, `complete`, or `以此版为准` claim, invalidate that claim visibly before continuing. Keep the old media only as a clearly labeled failed reference until a verified replacement is uploaded and hash-checked.
13. When the same or near-identical spoken passage occurs more than once, enumerate every source occurrence and every mapped final-media occurrence before choosing a cut. A successful cut of one occurrence does not satisfy the requirement while another matching occurrence remains at a user-reported final timestamp.
14. Normalize ASR and document text before forbidden-phrase scanning, including Unicode normalization, Simplified/Traditional Chinese conversion, punctuation removal, and explicitly reviewed pronoun or homophone variants. If raw ASR contains a target such as `它實踐` while the normalized scanner reports `它实践` absent, the audit is contradictory and must fail closed.
15. For every red-text deletion, repeated/near-identical passage, homophone, or user-reported final-media timestamp, run `python scripts/audit_spoken_release.py run ...` before upload. The requirement JSON must enumerate every source occurrence, identify remove/retain intent, protect required retained contexts, and include each user-reported final timestamp. Each final ASR JSON must identify the exact audited media through `media` or `media_sha256`, and that hash must match a `--final-media` file. A nonzero exit code, `release_blocked=true`, an undeclared source ASR occurrence, a missing retained context, an unchecked timestamp, or an unbound ASR pass blocks release; do not replace this gate with a hand-written summary.

## Keep Out Of Git

Do not commit local runtime assets or generated outputs:

- `.venv/`
- `.env`
- `tools/`
- `output/`
- `scratch/`
- `.omx/`
- `__pycache__/`
- `*.egg-info/`

Before sharing source, run:

```powershell
python scripts/audio_cleanup.py clean-repo
```

## External Verification Boundaries

实现层面的已知尾项不得留在口头说明中：wheel 预设、Python 3.10/3.11、真实确定性 final smoke、共享媒体工具、超长流程拆分、守门检出能力自检与交付门反向拦截均由代码与 CI 门禁，逐条对应见 `Enforcement Map`。仍有两类证据天然不能由无资产 CI 伪造：

1. **真实模型执行**：公共 CI 不携带 `tools/Respiro-en`、权重或 torch，因此只验证 capability state 和无模型确定性 final 链。凡声称使用 Respiro / DeepFilterNet，必须在有资产的仓库 `.venv` 先跑 `doctor`，再用真实短音频与最终媒体核对 `respiro_succeeded` / `stage_status.deepfilternet.applied`、fallback、收益和伤害守门；缺少这些证据就是未验证。
2. **感知听觉**：指标、波形、频谱和 ASR 不能替代人耳。未使用可感知播放工具时，Agent 必须把听感标为 `UNVERIFIED` 并请求用户最终确认；这是真实证据边界，不是可以通过伪造 PASS 消除的软件缺口。

## Skill Reference

The detailed Codex skill lives at:

```text
.codex/skills/audio-sound/SKILL.md
```

Read its `references/workflow.md`, `references/commands.md`, and `references/acceptance.md` when deeper process detail is needed. For physical spoken-segment deletion and seam repair, also read `references/spoken-segment-boundaries.md`.
