# Adding a New Object

Every graspable object is a `.usd` under `assets/` plus one entry in `config/drill_variants.yaml`.
Getting a new one in takes three asset-prep steps and one yaml entry.

The three scripts exist because three different things can each silently misplace an object — no
error, just a policy that never reaches the trigger. Each script's docstring explains the specific
failure it prevents; this document is the order to run them in.

---

## 1. Convert to `.usd`

Skip if your asset is already `.usd`.

```bash
python tools/convert_usdz.py assets/spray/spray1.usdz          # -> assets/spray/spray1.usd
```

`usd_path` in the yaml must be a `.usd`. A `.usdz` is a zip archive, so nothing can be authored
into it — which matters at step 3. Keep the original archive next to the output: texture paths
stay anchored into it.

## 2. Normalize units, origin and size

```bash
python tools/normalize_asset.py assets/spray/spray1.usd --grip_diameter 0.045
python tools/normalize_asset.py assets/spray/*.usd --dry_run      # inspect before writing
```

This bakes three conventions into the file, so the yaml doesn't have to compensate for them:

- **Metres.** Drills are authored at `metersPerUnit=1.0`; most DCC exports arrive in centimetres.
  Isaac respects the stage's unit, but `trigger_offset` is written in metres by hand — so a
  mismatch shows up as a keypoint 100× off, not as an error.
- **Origin.** `trigger_offset` is relative to the rigid body's *root*, not to the mesh. Geometry
  sitting 2.5 m from its origin spawns 2.5 m from where the env put it.
- **Size.** `--grip_diameter` scales the object so its handle fits the hand. Doing this through the
  yaml `scale` field instead would mean pre-dividing every hand-annotated offset by the same
  factor.

After this the asset is in metres, centred on its origin, and hand-sized — so the yaml carries
`scale: [1, 1, 1]` and keypoints can be read straight off the geometry.

## 3. Author the physics schemas

```bash
python tools/add_physics_apis.py assets/spray/spray1.usd
python tools/add_physics_apis.py assets/spray/*.usd --mass 0.4 --approximation convexHull
```

`MultiAssetSpawnerCfg(rigid_props=…, collision_props=…, mass_props=…)` **modifies** properties that
must already exist; it does not create them. A geometry-only asset therefore fails late, at sim
start, with `Failed to find a rigid body when resolving '/World/envs/env_.*/Drill'`. This script
authors `PhysicsRigidBodyAPI` + `PhysicsMassAPI` on the default prim, and `PhysicsCollisionAPI` +
`PhysicsMeshCollisionAPI` on every mesh.

The default approximation is `convexDecomposition`, not `convexHull`: the index finger has to reach
*into* the trigger recess, and a hull would fill it in and make the trigger unreachable. Use
`--approximation convexHull` only for genuinely convex parts.

---

## 4. Register it in `config/drill_variants.yaml`

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
actually need annotating per object:

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
| `scale` | `[1,1,1]` | Should stay `[1,1,1]` if step 2 was run. |
| `drill_bit_offset` / `drill_bit_rot` | `[0,0,0]` / identity | Tool tip pose, for the alignment stage. |

### Non-drill objects

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

---

## 5. Check it

```bash
python scripts/play_drill.py --drill_variants config/drill_variants.yaml --checkpoint <ckpt>
```

`--drill_variants` takes any yaml, so keep an experimental object in its own file (as
`config/drill_variants_test.yaml` does) until its keypoints are right. Look for: the object resting
on the table rather than falling through it (step 3), sized to the hand (step 2), and the trigger
marker sitting on the actual trigger (step 4).

Global physics — mass, hand stiffness/damping, table height, action scales — is in
`config/drill_config.yaml` and applies to every variant, not per object.
