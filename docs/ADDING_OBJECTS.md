# Adding a New Object

Every graspable object is a `.usd` under `assets/` plus one entry in `config/drill_variants.yaml`.
This document covers the yaml entry — the annotation that has to be done per object.

```yaml
- name: drill_yellow
  variant_index: 2
  enabled: true
  usd_path: drill_yellow1.usd        # resolved relative to assets/
  forward_axis: -X
  up_axis: Z
  trigger_offset: [-0.035, 0, 0.12]
  thumb_target_local: [0.0198, -0.0332, 0.1233]
  body_mask_axis: Z
  body_mask_min: 0.0707
  body_mask_max: 0.12
  initial_pos: [0.7, 0.2, 0.0552]
  initial_rot: [0.5, 0.5, 0.5, 0.5]
  scale: [1.2, 1.2, 1.5]
```

Only `name` and `usd_path` are required — everything else has a default
(`DrillVariantCfg` in [`tasks/grasp_drill_env.py`](../tasks/grasp_drill_env.py)). The fields that
actually need annotating:

| Field | Default | What it does |
|---|---|---|
| `variant_index` | list order | **Fixed global index.** One-hot width and trained checkpoints depend on it — never renumber an existing one. |
| `enabled` | `true` | `false` keeps the index reserved but stops the object spawning. |
| `trigger_offset` | `[0,0,0]` | Trigger position in the object frame. The single most important annotation — and the one that does not transfer between geometries. |
| `thumb_target_local` | `[0,0,0]` | Thumb target in the object frame. Set to `null` when `trigger_link` is itself a thumb link, or the two reward terms fight each other. |
| `forward_axis` / `up_axis` | `Z` / `Y` | Object orientation convention, e.g. `-X`. |
| `body_mask_axis` + `body_mask_min/max` | `Z`, `-0.03`…`0.045` | Slice of the object's surface points counted as graspable body, used by the envelopment reward. |
| `initial_pos` / `initial_rot` | origin / identity | Pose on the table. Add `initial_pos_1`, `initial_rot_1`, … for several; one is drawn per reset. |
| `goal_pos_N` / `goal_rot_N` | — | Stage-2 alignment targets, paired with `initial_*_N` by suffix. |
| `scale` | `[1,1,1]` | Applied at spawn. Keypoint offsets above are *not* pre-scaled by it, so annotate against the final size. |
| `drill_bit_offset` / `drill_bit_rot` | `[0,0,0]` / identity | Tool tip pose, for the alignment stage. |

## Non-drill objects

Tools that are not drills need a few more fields. The commented-out spray-bottle and spray-can
entries at the bottom of `drill_variants.yaml` are working templates:

| Field | Default | When you need it |
|---|---|---|
| `trigger_link` | `R_index_intermediate` | The digit that presses this object's trigger — a spray lever is worked by the middle finger, not the index. |
| `use_index_tip` | `false` | Measure trigger distance from the reconstructed fingertip instead of the link origin. |
| `min_contact_links` | `7` | How many of the 13 instrumented hand links must touch to count as enveloped. A drill handle fills the hand and supports 7; a thinner spray body does not. |
| `trigger_mesh_keywords`, `body_mesh_keywords`, `handle_mesh_keywords` | `null` | Select trigger/body meshes by name when a single axis slice can't separate them. |

> Those commented entries also carry `compose_mesh_transform`, which **no code reads** — the loader
> ignores it. Don't copy it expecting an effect.

## Checking it

```bash
python scripts/play_drill.py --drill_variants config/drill_variants.yaml --checkpoint <ckpt>
```

`--drill_variants` takes any yaml, so keep an experimental object in its own file (as
`config/drill_variants_test.yaml` does) until its keypoints are right.

Global physics — mass, hand stiffness/damping, table height, action scales — is in
`config/drill_config.yaml` and applies to every variant, not per object.
