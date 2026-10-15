# Photo-guided office scene

This variant reconstructs the six ordered local references from
`/home/sivan/Documents/door/images`. Images 3 and 4 are byte-identical.
Dimensions, camera positions, furniture detail and light output are estimated;
this is not a photogrammetric scan or measured photometric calibration.

## Scene

Door centre is `(0,0,0)`; office is +Y; entrance is -Y; +Z is up.
Estimated office is 4.45 m wide, 4.80 m deep, with a 3.05 m dark ceiling and
white perimeter soffit. Entrance suspended tile ceiling is 2.78 m high.

The previous arbitrary props and distant point lights are hidden/disabled.
The reconstruction includes a square entrance ceiling fixture; a cool linear
office pendant; warm off-white walls; black skirting; grey/beige PVC floor;
door access controls; left equipment cases, cartons and a notice board;
right oak workstation, black chairs/monitors and RGB tower; back cupboards,
printer, documents and a reflective night window. Unseen areas are inferred.
Fixture powers are rendering parameters, not measured electrical wattages.

## Motion, cameras and checks

The existing 2601-state PhysX recording is reused, not re-recorded or inferred
from the photos. The retained door and robot mesh/material signatures are
checked before/after furnishing. All 520 baked poses are checked against the
source. Front/wrist cameras remain recorded cameras. The third `overview`
camera is now an animated human-height room walkthrough.

New walls and major furniture are checked against per-frame moving-body AABBs
over all 2601 states. This conservative visual-bounds test is not a contact
simulation: floor, thin decorations and wires are excluded; architecture
abutting the door frame excludes door-vs-wall checks. No new collision actors
are added to PhysX. See `room_clearance.json` for the exact scope/results.

`preview_wc_photo_room.py` additionally renders approximate portrait phone
viewpoints with the robot hidden ONLY in those still-image comparisons. The
robot remains visible in the full animation and all three dataset cameras.
No photometric augmentation is applied in this reference-matching version
(brightness and contrast both 1.0); matching the supplied scene is the goal.

## Reproduce on PS1

Use `configs/rgb_photo_office1.json`; run the standard `pipeline.py --stage
prepare`, inspect the above preview script, then `render_wc_single_parallel.py`
with GPUs listed in the config. The final run has a separate immutable root
from development trials. `review_wc_rgb_aug5.py` verifies the exported LeRobot
dataset and makes dual/triple-view videos. `review_wc_photo_room.py` makes the
full walkthrough and six ordered real/render comparisons.

The final LeRobot export has one 52-second episode, 520 frames at 10 fps,
640×480 for each of three RGB views, and the original state/action labels.
Room rendering does not create a new independent physical trajectory.
