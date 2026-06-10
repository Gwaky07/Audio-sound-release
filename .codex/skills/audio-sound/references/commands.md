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

## 4. 更自然的低响度口播模式

```bash
python ../../../scripts/audio_skill_workflow.py run "<输入音频>" --mode reference-style
```

## 5. 带局部频谱窗口的运行方式

当你已经知道要重点看哪些节点时：

```bash
python ../../../scripts/audio_skill_workflow.py run "<输入音频>" --mode reference-legacy --focus-window 节点A,234.8,1.4 --focus-window 节点B,530.3,1.1
```

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
