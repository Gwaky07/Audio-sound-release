---
name: audio-sound
description: 仓库级中文总技能。用于中文口播、课程讲解、配音、旁白、批量音频整理、删词剪辑与交付。默认以自然度、清晰度和稳定性为第一优先级：保留原始人声，只做保守母带和响度统一；呼吸、口水音、降噪、停顿残留和节点精修必须在缺陷被确认后局部启用。涉及删词时必须先做语音边界收口与音画同步验证，并把最终 WAV/MP3 交付到 output/修音成品，命名为 修音版_原名，重名时递增 _01/_02。
---

# audio-sound

这是本仓库唯一保留的音频处理总 skill。目标不是“跑一遍脚本就算完成”，而是稳定交付你在本线程里已经认可的那种成品。

先读 `references/workflow.md`。  
需要具体命令时读 `references/commands.md`。  
需要最终验收标准时读 `references/acceptance.md`。

如果任务涉及删除口播、删词、删句、边界残留、接缝杂音、卡顿或突然续接，必须额外阅读 `references/spoken-segment-boundaries.md`；该流程优先于普通降噪节点精修，不得只靠加长交叉淡化处理。

## 默认入口

在这个仓库里，只要用户说“用 skill 处理”“帮我处理音频”“清理这个音频”“修一下这个音频”，默认就理解为要最终可直接使用的严格成品，而不是普通预清理。

- 默认走 `auto`，目标是 Best Repair，不是保守母带
- `auto` 先产出 `natural` 安全基线，再强制竞争 `final_repair_best`（完整 `final` 修音链），并按诊断竞争可选模型候选；硬守门通过才自动交付
- `natural` 只在增强全部失败时作为回退，且必须标记 `delivery_incomplete_for_repair_intent=true`，不能当作修音完成
- 只有用户明确提出其他目的时才切换：
  - 只要自然基线：`natural`
  - 要旧版呼吸清理风格：`reference-style`
  - 要复现或对照旧版激进链：`reference-legacy`
  - 要直接跑完整修音链：`final`
  - 要同文件噪声窗口隔离：`voice-isolate`
  - 要审查标记和复核报告：`review`

除非用户明确要求快速预览，否则不要把 `scripts/audio_cleanup.py clean` 当成最终交付入口。最终交付优先用 `scripts/audio_skill_workflow.py run ...`。

如果用户要求降噪、呼吸音、气口、口水音、清晰度 EQ、人声增强、“最终修复版”，或者反馈上一版听起来几乎没区别，不能停在只做高通、轻压缩和响度统一且未通过增强候选的结果。优先依赖 `auto` 的 `final_repair_best`，或直接 `--mode final`；确认报告含 `capability_plan`、`repair_scorecard` 且真实应用了修复阶段。只有响度或峰值变化、没有实际修复阶段的结果不算完成。

`final` 的呼吸处理采用安全闭环：`raw_wav` 保持不可变；Respiro 和辅助停顿边缘检测分别保留证据；辅助窗口必须再通过噪声型频谱特征和语音起点保护。首轮按前置停顿底噪自适应衰减，母带后重新检测并做窄窗口补处理。最终确认残留高于局部底噪 3 dB 时必须阻止交付。只有 `breath_cleanup.status=PASS` 且 `final_residual_windows` 为空，才能报告呼吸清理完成。

当用户要求停顿过渡更干净、减少频谱中的紫色残留时，`final` 使用语音安全 AutoGate 等价策略清理静音证据内部的非语音过渡：保护两侧字头和尾音（句间约 45 ms，片头/片尾约 25 ms），确认无声核心压到 `silence_floor_dbfs=-96`，短 fade，不做全局 Dynamics Gate / `agate`。只有 `pause_cleanup.status=PASS`、`mode=speech_safe_autogate` 且其 `final_residual_windows` 为空，才能报告过渡清理完成；固定刻度局部频谱必须显示中间接近黑色静音地板且两侧语音连续。

`auto` 的 Codex 权限边界：

- 只能在白名单候选中选择：`natural_baseline`、`final_repair_best`、`clarity_leveling_safe`、`noise_cleanup_safe`、`local_defect_safe`、`respiro_breath_safe`、`deepfilter_denoise_safe`、`model_combined_review`
- 修音意图下优先 `final_repair_best`；不得因“模型更高端”压过已通过守门的完整 final 修音链
- Agent 判断层只输出能力模块 `needed/applied/skip`，不得生成任意 DSP/FFmpeg 参数
- Respiro 与 DeepFilterNet 只能按缺陷证据和 `doctor` 能力生成固定安全候选；不得启用 post-filter、门限或数字硬静音
- 单模型候选缺少 ASR 时必须通过更严格的语音衰减、局部相关性和频谱守门；组合模型始终要求可绑定双 ASR
- 模型未真实成功、使用 fallback、没有可量化收益或伤害守门失败时，只能回退已通过守门的 `natural`，并标记修音意图未完成
- ASR 冲突或高置信吞字时转 `manual_review`

## 删词与接缝的特殊入口

当用户要物理删除一句话、口头禅或片段时，使用 `scripts/remove_spoken_segments.py`，不是 `audio_cleanup.py clean`。先按 `references/spoken-segment-boundaries.md` 定位真实局部时间和语义边界，再运行删词脚本。

- 默认使用 `--boundary-search-ms 80` 与 `--crossfade-ms 12`。
- 若用户反馈“后面卡一下”或“下一句/一个字突然接上”，改用 `--seam-pause-ms 60–100`，中文讲解默认从 `80` 开始。
- 不要用更长的交叉淡化掩盖残留或语义续接；它会混叠不同音素。
- 对视频必须验证音频与 MP4 时长差小于一帧；语音安全接缝会同步冻结上一帧。

## 严格执行顺序

1. 先确认运行环境：
   - 运行 `doctor`
   - 明确 `ffmpeg`、`ffprobe`、`DeepFilterNet`、Respiro-en 是否可用
2. 再跑主流程：
   - 默认 Best Repair：`auto` 竞争 `final_repair_best`
   - `final` 主链覆盖呼吸/气口、停顿过渡、证据门控降噪、清晰度 EQ、去齿音、稳量和响度
   - 默认保留源采样率、声道数和立体声布局；多声道只为诊断临时下混，处理和交付不能因此变成单声道
   - Respiro-en 与 DeepFilterNet 可由 `auto` 按证据生成固定安全候选；自适应强降噪、门限和自动静音仍只能在问题确认后显式启用
3. 主流程完成后必须做自动诊断、能力计划和质量守门：
   - 检查底噪、削波、动态范围、可用噪声窗口和音量稳定性
   - 输出 `capability_plan` 与 `repair_scorecard`
   - 比较处理前后时长、活跃语音衰减和短时增益波动
   - 守门失败时禁止把增强候选当交付；可回退自然基线但必须标记 incomplete
4. 主流程完成后必须做频谱和响度复核：
   - 输出整体频谱图
   - 必要时输出问题节点的局部频谱图
   - 在聊天窗口中展示频谱图
5. 如果还有残留呼吸、气口、细丝、口水音：
   - 先由闭环残留守门阻止交付
   - 不要整段重做；只对确认的噪声型、非语音窄窗口继续处理
   - preservation 守门只能排除这些有证据的授权窗口，不能排除附近语音
6. 节点精修后必须再次导出 MP3、再次复测响度，并给出最终交付文件路径

## 自适应控制边界

- 自适应选择器只能返回 `baseline`、`leveling_gentle`、`noise_review`、`manual_review`。
- `baseline` 保持自然基线；`leveling_gentle` 只允许受限轻压缩；`noise_review` 只记录稳定底噪和候选窗口，不自动降噪；`manual_review` 阻止自动加强。
- Codex/GPT 只能根据诊断报告选择白名单档位、解释理由和决定是否进入复核，不能直接生成任意 FFmpeg/DSP 参数。
- 所有自动档位都必须保持 `allow_destructive_cleanup=false`；DeepFilterNet 仅允许固定的无 post-filter 候选，Respiro 仅允许有限局部 duck，门限、硬静音和整段呼吸切除仍禁止。
- `quality_guard` 必须同时检查声道、采样率、时长、削波、活跃语音衰减和短时增益波动；任一发布阻断项失败都不得交付。

## 铁律

- 只保留一个仓库 skill 入口：`audio-sound`
- 对最终成品，优先使用仓库 workflow，不用临时拼 shell 管道代替
- 在这个仓库里，“处理音频”默认就是做最终版，不默认保留轻处理思路
- 频谱图是过程证据；严格清理时，聊天窗口里要出现频谱图
- 最终用户只看 `output/修音成品/`；不要让用户从 `audio_preprocess/` 的中间文件里挑最终版
- 中文原文件名必须保留，最终命名固定为 `修音版_<原音频文件名>.wav|mp3`；同名已存在时递增为 `_01`、`_02`
- 中间 wav/mp3 默认清理掉；只有排查阶段问题时才加 `--keep-intermediate-audio` 保留
- 用户给了具体时间点时，只修确认过的窄窗口，避免吞字
- 涉及删词时，时间码只是候选范围；必须检查报告中的 `requested_cuts`、`cuts`、`crossfade_frames`、`seam_pause_frames` 和局部频谱/波形
- 用户没给时间点但说“不干净”时，先做局部频谱/能量巡检，再精修
- 不能口头宣称“Respiro-en 已调用”；必须以 `doctor` 或主流程报告中的实际状态为准
- 如果实际走了 fallback，而不是真正的 Respiro-en，必须明确说出来
- 节点精修后，整体响度不应被带偏；要复核和基底成品是否基本一致
- DeepFilterNet 防吞字恢复只允许修复夹在连续语音中的短缺口，不能恢复开头静音区或停顿区里的孤立杂音

## 对外说明口径

当用户问“这个 skill 现在到底会怎么做”时，按下面这个口径回答：

1. 先跑 `auto`：保留原声格式，建立 `natural` 安全基线
2. 自动分析底噪、削波、动态范围，并生成能力计划
3. 强制竞争 `final_repair_best`；按证据再竞争模型安全候选
4. 用格式、吞字、高频清晰度/刺音、短时增益、呼吸/停顿残留和可选双 ASR 守门筛选
5. 最好且通过守门的候选自动交付；若只能回退 `natural`，标记修音意图未完成
6. 最终交付到 `output/修音成品/`，文件名为 `修音版_原名.wav/mp3`，如果已有同名则自动加 `_01/_02`

## 本 skill 不该做的事

- 不要为了追求绝对干净而牺牲字头、辅音、高频清晰度或句间自然动态
- 不要把“脚本执行成功”当成“成品达标”
- 不要为了清理呼吸音，误吞后面起字
- 不要把“删词脚本成功执行”当成“删词接缝自然”；残留、突然归零或突然接字都必须继续修
- 不要把用户已经认可的响度重新推高
- 不要输出只有英文短名、看不出来源的文件名
- 不要擅自理解成“先随便出一版试试”；默认就是直接朝最终版做
