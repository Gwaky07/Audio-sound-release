# Audio-sound Agent Instructions

This repository is a Windows-oriented audio cleanup workflow for Chinese spoken-word audio. When a user asks to process, clean, repair, or prepare audio in this repository, treat it as a request for a final usable delivery by default, not a lightweight preview.

## Default Workflow

Use the repo workflow entrypoint for final delivery:

```powershell
python scripts/audio_skill_workflow.py run "<input-audio>"
```

The default workflow mode is `reference-legacy`. Use another mode only when the user clearly asks for it:

- `reference-style` for a more natural and conservative result
- `final` for balanced final delivery
- `voice-isolate` when noise windows from the same file are provided
- `review` when the user wants markers and review artifacts

Do not use `scripts/audio_cleanup.py clean` as the default final-delivery path. Keep it for setup, inspection, lower-level preset control, compatibility, and troubleshooting.

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

Before sharing source, run:

```powershell
python scripts/audio_cleanup.py clean-repo
```

## Skill Reference

The detailed Codex skill lives at:

```text
.codex/skills/audio-sound/SKILL.md
```

Read its `references/workflow.md`, `references/commands.md`, and `references/acceptance.md` when deeper process detail is needed. For physical spoken-segment deletion and seam repair, also read `references/spoken-segment-boundaries.md`.
