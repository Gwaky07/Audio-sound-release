# audio-sound 常用命令

## 1. 先做环境核查

```bash
python ../../../scripts/audio_cleanup.py doctor
```

## 2. 查看可用 workflow 模式

```bash
python ../../../scripts/audio_skill_workflow.py describe-modes
```

## 3. 默认严格成品模式

这是本 skill 的默认最终交付入口：

```bash
python ../../../scripts/audio_skill_workflow.py run "<输入音频>" --mode reference-legacy
```

最终音频默认只看：

```text
../../../output/修音成品/修音版_<原音频文件名>.wav
../../../output/修音成品/修音版_<原音频文件名>.mp3
```

如果同名已存在，会自动生成 `_01`、`_02`。中间 wav/mp3 默认清理掉，只保留报告和频谱证据。

如果要排查阶段问题，才保留中间音频：

```bash
python ../../../scripts/audio_skill_workflow.py run "<输入音频>" --mode reference-legacy --keep-intermediate-audio
```

## 4. 更自然的低响度口播模式

```bash
python ../../../scripts/audio_skill_workflow.py run "<输入音频>" --mode reference-style
```

## 5. 带局部频谱窗口的运行方式

当你已经知道要重点看哪些节点时：

```bash
python ../../../scripts/audio_skill_workflow.py run "<输入音频>" --mode reference-legacy --focus-window 节点A,234.8,1.4 --focus-window 节点B,530.3,1.1
```

排查“开头突然多出来声音”和“某个字断开/吞字”时，必须带 focus-window：

```bash
python ../../../scripts/audio_skill_workflow.py run "<输入音频>" --mode reference-legacy --focus-window start_artifact_1s,0.75,0.9 --focus-window zishi_12s,11.65,1.25
```

运行后重点检查：

- `audio_process_report.json` 里的 `respiro_succeeded`
- `audio_process_report.json` 里的 `deepfilternet_dropout_repair_windows`
- `workflow_artifacts/spectrograms/focus/` 下的局部频谱图
- 最终 WAV/MP3 是否在 `output/修音成品/`
- 最终 WAV/MP3 文件名是否是 `修音版_原名`，重名时是否递增 `_01/_02`

## 6. 同文件噪声窗口驱动的隔离模式

```bash
python ../../../scripts/audio_skill_workflow.py run "<输入音频>" --mode voice-isolate --noise-window 143.089208:144.093687 --noise-window 161.583729:169.578792
```

## 7. 精确节点补修

当主流程做完后，仍有少量明确时间点残留时，用这个脚本直接修最终 WAV。

```bash
python ../../../scripts/exact_window_cleanup.py "<基底WAV>" --output "<新WAV>" --report "<报告JSON>" --mute-window 235.466,235.785 --mute-window 530.828,531.008
```

如果不能整段硬静音，而是要轻一点压低：

```bash
python ../../../scripts/exact_window_cleanup.py "<基底WAV>" --output "<新WAV>" --report "<报告JSON>" --duck-window 235.466,235.785,0.12,8
```

说明：

- `mute-window` 适合确认是纯呼吸/纯残留
- `duck-window` 适合边界贴近起字、不能切太狠的情况

## 8. 导出节点补修后的 MP3

```bash
ffmpeg -y -hide_banner -nostdin -i "<新WAV>" -codec:a libmp3lame -q:a 2 "<新MP3>"
```

## 9. 单独做窄起字前清理

```bash
python ../../../scripts/narrow_onset_cleanup.py "<输入WAV>" --output "<输出WAV>" --report "<报告JSON>"
```

这个脚本适合做起字前的窄窗口清理，但如果用户已经明确点名具体时间点，优先直接做 `exact_window_cleanup.py`。

## 10. 分阶段能量巡检

当怀疑 DeepFilterNet 恢复错了、后处理吞字、或某个黑色断档是否影响发音时，先对比阶段文件：

```powershell
@'
from pathlib import Path
import json
from audio_sound.pipeline import _load_wave_samples, measure_segment_levels

root = Path("<本次 workflow root>")
core_report = next(root.rglob("audio_process_report.json"))
workflow_report = next(root.rglob("skill-file-report.json"))
core = json.loads(core_report.read_text(encoding="utf-8"))
workflow = json.loads(workflow_report.read_text(encoding="utf-8"))
files = {
    "raw_after_respiro_spectra": Path(core["outputs"]["raw_wav"]),
    "deepfilternet": Path(core["outputs"]["denoised_wav"]),
    "mastered": Path(core["outputs"]["clean_wav"]),
    "delivered": Path(workflow["deliverables"]["wav"]),
}
windows = [("check", 1.08, 1.20)]
for label, path in files.items():
    params, samples = _load_wave_samples(path)
    print(label, path.name)
    for name, start, end in windows:
        peak, rms = measure_segment_levels(
            samples,
            start_index=int(start * params.framerate),
            end_index=int(end * params.framerate),
        )
        print(name, start, end, peak, rms)
'@ | python -
```

## 11. 物理删词并同步剪视频

当目标不是压低呼吸/口水音，而是把一句话或几个口头语从时间线上真正删掉时，用这个脚本。它会按给定时间窗删除片段，在局部低能量边界收口，并输出最终 WAV、MP3；如果输入是视频，还会同步剪视频并输出 MP4。

单个文件：

```bash
python ../../../scripts/remove_spoken_segments.py run "<输入视频或音频>" --cut "0.62,1.82" --removed-phrase "啊说德语啊"
```

如果删完仍有“边界残留”“接缝杂音”，先复核局部频谱和报告中的实际 `cuts`，不要直接拉长淡化。若两个保留句子续接过急，使用语音安全接缝：

```bash
python ../../../scripts/remove_spoken_segments.py run "<输入视频或音频>" --cut "0.62,1.82" --removed-phrase "啊说德语啊" --seam-pause-ms 80
```

`--seam-pause-ms 60–100` 会让前句淡出、冻结视频上一帧、短暂停顿后再淡入下一句；它适合“卡一下”“突然接上一个字”，不应对所有删词默认启用。

尾段删除时，结束时间留空：

```bash
python ../../../scripts/remove_spoken_segments.py run "<输入视频或音频>" --cut "4.40," --removed-phrase "是不是"
```

同一个文件删多个位置时，重复 `--cut`：

```bash
python ../../../scripts/remove_spoken_segments.py run "<输入视频或音频>" --cut "0.00,1.80" --cut "2.10,2.34" --cut "4.40," --removed-phrase "你看，那个，是，是不是"
```

批量处理用 JSON：

```bash
python ../../../scripts/remove_spoken_segments.py run-batch jobs.json
```

`jobs.json` 示例：

```json
{
  "crossfade_ms": 12,
  "boundary_search_ms": 80,
  "jobs": [
    {
      "input": "C:/Users/Guanghe/Downloads/第二段.mp4",
      "removed_phrase": "啊说德语啊",
      "cuts": [{"start": 0.62, "end": 1.82}]
    },
    {
      "input": "C:/Users/Guanghe/Downloads/第六段.mp4",
      "removed_phrase": "你看，那个，是，是不是",
      "cuts": [
        {"start": 0.0, "end": 1.8},
        {"start": 2.1, "end": 2.34},
        {"start": 4.4}
      ]
    }
  ]
}
```

输出仍然只看：

```text
../../../output/修音成品/修音版_<原文件名>.wav
../../../output/修音成品/修音版_<原文件名>.mp3
../../../output/修音成品/修音版_<原文件名>.mp4
```

这个脚本不调用 Respiro-en 或 DeepFilterNet；报告里会明确记录它们未使用。它适合“明确给了时间点、要真正删掉文字”的场景，不替代默认的整体修音 workflow。

## 12. 发布前重复口播与红字闸门

凡是文档红字、重复口播、同音相邻词，或用户报告“最终视频某时间仍有目标词”，上传前必须运行发布审计：

```powershell
python ../../../scripts/audit_spoken_release.py run `
  --requirements "<要求映射.json>" `
  --source-asr "<原片ASR-1.json>" `
  --source-asr "<原片ASR-2.json>" `
  --final-asr "<成片ASR-1.json>" `
  --final-asr "<成片ASR-2.json>" `
  --segment-report "<segment-removal-report.json>" `
  --final-media "<最终MP4>" `
  --final-media "<最终WAV>" `
  --final-media "<最终MP3>" `
  --minimum-final-asr-engines 2 `
  --output "<release-audit.json>"
```

要求映射最小示例：

```json
{
  "source_document_revision": 422,
  "requirements": [
    {
      "id": "red-text-它实践",
      "target": "它实践",
      "variants": ["它實踐", "他实践", "他實踐"],
      "allowed_contexts": ["它这个实践", "它這個實踐", "他这个实践", "他這個實踐"],
      "source_occurrences": [
        {"start_seconds": 606.10, "end_seconds": 607.28, "action": "remove"},
        {"start_seconds": 642.18, "end_seconds": 643.34, "action": "remove"},
        {"start_seconds": 656.26, "end_seconds": 657.58, "action": "remove"},
        {"start_seconds": 672.94, "end_seconds": 674.14, "action": "remove"}
      ],
      "user_reported_final_timestamps": [554.62, 568.70],
      "expected_final_occurrences": 0,
      "minimum_allowed_context_occurrences": 1
    }
  ]
}
```

规则：

- `source_occurrences` 必须枚举原片中每一遍近似口播；只登记已知 cut 会失败。
- `allowed_contexts` 用于保留类似“它这个实践”的正确语句，避免把内含目标字符的正常上下文误删。
- `minimum_allowed_context_occurrences` 用于证明正确保留句仍在；目标词消失但保留句也被吞掉时仍会失败。
- 若某套 ASR 把已人工核对的保留句同音字识别错（例如把“他这个实践”写成“他这个时间”），只把该完整句加入 `allowed_contexts`；不要把“实践→时间”设成全局 `normalization_replacements`。
- 审计自动执行 NFKC、常用繁简体、标点和空白归一化；代词或同音词只通过 `variants` 或 `normalization_replacements` 显式配置，不做全局猜测。
- 任一源片重复项没有被实际 `cuts` 覆盖、最终 ASR 仍命中目标、或用户时间点缺少 ASR 证据时，命令返回非零并写出 `FAIL`。
- `--final-media` 必须指向真正准备上传的文件；报告会记录文件大小和 SHA-256，换文件后旧审计自动失效。
- 每个最终 ASR JSON 必须包含 `media`（实际转写文件路径）或 `media_sha256`；其哈希必须与某个 `--final-media` 完全一致。用中间 WAV 转写、再上传另一个正式 WAV/MP4 时会强制失败。
- 默认至少需要两个不同的最终 ASR `engine` 标签；同一个模型重复跑两遍不算独立复核。只有明确记录本机仅有一套模型时，才可显式降为 `--minimum-final-asr-engines 1`，并在报告中保留该限制。
- 该报告是 `MODEL-INFERRED` 证据，不等于直接听审；有直接听审工具时仍需完成听审。
