# Preset Routing

Use this file when the user's wording is ambiguous and you need a stable routing rule.

## Default

- Default to `safe`.

## Route to `fast`

Use `fast` when the user emphasizes:

- speed
- first pass
- rough cleanup
- quick preview
- large batch throughput

Common phrases:

- "quick cleanup"
- "fast pass"
- "batch this folder first"
- "先跑一遍"
- "快速清理"
- "批量先处理"

## Route to `safe`

Use `safe` when the user emphasizes:

- natural voice
- conservative cleanup
- spoken-word quality
- preserving consonants and tails
- course, lecture, narration, or teacher audio

Common phrases:

- "keep it natural"
- "conservative cleanup"
- "clean this voiceover"
- "老师音频"
- "保守一点"
- "少伤人声"

## Route to `review`

Use `review` when the user wants:

- candidate markers
- likely pause or breath regions
- cleanup plus QA
- manual spot-check support

Common phrases:

- "flag suspicious regions"
- "give me review markers"
- "标出可疑停顿"
- "需要复核"
- "清理并标记"

## Common flag overrides

- stronger denoise:
  - `--denoise-strength aggressive`
- lighter denoise:
  - `--denoise-strength light`
- avoid gate damage:
  - `--disable-gate`
- stronger reporting:
  - `--enable-silence-report`
