# WC wood-only texture binding

`materials.preserve_subvisual_materials: true` enables asset-time binding of
wood textures to individual visual meshes. It must be used with mesh materials
(`door_use_urdf_rgba` disabled) and one environment per simulation process.
`preview_wc_glass_fixed.py` enables it for the fixed WC-texture/white-wall preview.
Historical experiment YAML files are intentionally not rewritten: their old
whole-body appearance remains reproducible.

Only the wood allowlist in `rgb_door_materials.wood_visual_group` is textured.
Glass, gaskets and metal visuals retain their original geometry and materials.
No per-rigid-body texture/color setter is called on the door in this mode,
because those setters replace materials on every visual of a mixed body.
Wooden boxes become same-sized visual meshes; collisions, joints, inertias and
visual origins remain unchanged. The generated manifest lists changed/preserved
visuals, and an XML invariant rejects nonvisual changes.

## Checks

Run `python high-level/dp/tests/test_door_subvisual_materials.py` for CPU regression
checks. Existing `test_rgb*.py` checks cover compatibility with legacy mode.

On PS1, `preview_wc_glass_fixed.py probe` performs a real Isaac Gym camera test:
the same door is rendered with original alpha, opaque glass and the glass visual
removed. A red/blue background is swapped behind each door, and a central window
ROI is compared. Removing the visual is a **test control only**; it is never used
in the inference video, and even the control retains glass collision geometry.

2026-09-18 result: the original alpha=0.45 and alpha=1 images were identical in
the ROI; neither responded to the background swap (mean absolute RGB difference
0). The absent-visual control responded by 119.37 pixel levels. This confirms that
this URDF-alpha path in the tested Isaac Gym camera does **not** produce visible
transparency. Material separation is fixed, but alpha retention alone must not
be described as a successful semi-transparent glass rendering. No fake window
cutout or postprocessed transparency is substituted in policy RGB.

`inference`, `compose`, and `verify` stages capture the actual policy RGB plus an
overview, with augmentation disabled and fixed materials/lighting. The policy,
physics and historical training data/checkpoints are not changed. A single
preview rollout is not a new success-rate evaluation.
