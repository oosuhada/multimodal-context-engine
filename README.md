# Multimodal Context Engine

A clean-history, source-based evolution of LanguageBind for grouping image, audio, and video evidence into searchable semantic contexts.

The repository keeps the original multimodal encoders and processors instead of recreating them. A small task/registry layer derived from LAVIS is used to make fusion behavior extensible without turning the project into another general research framework.

```text
video ─┐
audio ─┼─ LanguageBind shared space ─ item embeddings ─┐
image ─┘                                               │
                                                      ├─ group fusion ─ context index
text query ─ LanguageBind text encoder ────────────────┘                 │
                                                                         ↓
                                                               ranked content groups
```

## What is added here

- Manifest-driven indexing of image/audio/video files.
- Multiple modality items can share a `group_id` and become one fused context.
- Natural-language retrieval against the fused context index.
- Per-result evidence list showing which media items support the match.
- Pluggable fusion strategies using the bundled LAVIS registry mechanism.
- Persistent NumPy + JSON index format; model weights and caches stay outside Git.

## Manifest format

```json
{
  "items": [
    {
      "id": "scene-001-video",
      "group_id": "scene-001",
      "modality": "video",
      "path": "./media/scene-001.mp4"
    },
    {
      "id": "scene-001-audio",
      "group_id": "scene-001",
      "modality": "audio",
      "path": "./media/scene-001.wav"
    },
    {
      "id": "scene-001-frame",
      "group_id": "scene-001",
      "modality": "image",
      "path": "./media/scene-001.jpg"
    }
  ]
}
```

## Run

LanguageBind keeps its original runtime dependency stack in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps

multimodal-context index ./manifest.json \
  --output ./artifacts/context-index

multimodal-context search ./artifacts/context-index \
  --text "people cooking together in a kitchen" \
  --top-k 5
```

## Source lineage

See [`THIRD_PARTY_NOTICE.md`](THIRD_PARTY_NOTICE.md). The repository has its own Git history; source origins are retained as attribution and licenses only.

