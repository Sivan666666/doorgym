# Pull/Push raw dataset audit

## Summary

| Dataset | Episodes | Frames | Mean frames | Max door (deg) | Max handle (deg) | Release phase door (deg) | Contact-loss door (deg) | Contact-loss speed (deg/s) | Post-contact-loss open (deg) | Both-contact (%) | Pull/push both-contact (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pull50Old | 50 | 43750 | 875.00 | 88.72 | 44.99 | 48.13 | 48.53 | 3.20 | 40.19 | 35.91 | 100.00 |
| Pull50Nested | 50 | 43750 | 875.00 | 88.97 | 44.99 | 48.13 | 48.52 | 3.12 | 40.45 | 35.88 | 100.00 |
| Pull200 | 200 | 175000 | 875.00 | 89.46 | 45.00 | 48.14 | 48.59 | 3.19 | 40.87 | 35.26 | 100.00 |
| Push200Legacy | 200 | 100000 | 500.00 | 90.00 | 44.60 | — | — | — | — | 26.05 | 37.58 |

## Phase fractions

| Dataset | close_gripper | grasp | hold_home | initial_hold | pass_through | pull_door | push_door | release_handle | return_home | rotate_handle | walk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pull50Old | 2.86% | 2.86% | 5.97% | 8.57% | 20.57% | 26.69% | 0.00% | 10.29% | 8.54% | 5.71% | 7.94% |
| Pull50Nested | 2.86% | 2.86% | 5.99% | 8.57% | 20.57% | 26.67% | 0.00% | 10.29% | 8.51% | 5.71% | 7.98% |
| Pull200 | 2.86% | 2.86% | 6.89% | 8.57% | 20.54% | 26.02% | 0.00% | 10.29% | 8.39% | 5.71% | 7.88% |
| Push200Legacy | 5.00% | 5.00% | 6.39% | 15.00% | 0.00% | 0.00% | 30.00% | 0.00% | 14.77% | 10.00% | 13.84% |

## Full metric means

| Metric | Pull50Old | Pull50Nested | Pull200 | Push200Legacy |
|---|---:|---:|---:|---:|
| action_delta_mean | 0.0186114 | 0.0186284 | 0.01862 | 0.0280634 |
| action_delta_p99 | 0.139574 | 0.139319 | 0.137772 | 0.0762291 |
| base_delta_x | -2.45564 | -2.45817 | -2.44833 | -2.45186 |
| base_delta_y | -0.00423385 | -0.00788507 | -0.00726078 | -0.00275207 |
| base_displacement_xy | 2.45596 | 2.45846 | 2.44868 | 2.45218 |
| contact_any_fraction | 0.396823 | 0.396777 | 0.388366 | 0.329 |
| contact_both_fraction | 0.359131 | 0.358766 | 0.35256 | 0.26051 |
| contact_both_longest_run | 314.24 | 313.92 | 308.49 | 130.2 |
| contact_loss_door_deg | 48.5334 | 48.5175 | 48.5891 | — |
| contact_loss_door_speed_deg_s | 3.20327 | 3.11639 | 3.18966 | — |
| contact_loss_frame | 484.5 | 484.54 | 477.955 | — |
| final_door_deg | 88.6774 | 88.9397 | 89.4186 | 86.455 |
| first_door_60_frame | 562.8 | 566.94 | 570.555 | 312.045 |
| first_handle_40_frame | 241.68 | 242.08 | 241.655 | 242.905 |
| frames | 875 | 875 | 875 | 500 |
| interaction_action_delta_mean | 0.0289767 | 0.0290147 | 0.0293041 | 0.0317203 |
| interaction_contact_both_fraction | 0.741259 | 0.740756 | 0.737225 | 0.52102 |
| max_door_deg | 88.723 | 88.9655 | 89.4587 | 89.9996 |
| max_handle_deg | 44.9885 | 44.9893 | 44.9962 | 44.5965 |
| post_contact_loss_extra_open_deg | 40.1896 | 40.448 | 40.8696 | — |
| post_release_extra_open_deg | 40.5925 | 40.833 | 41.3206 | — |
| pull_push_contact_both_fraction | 1 | 1 | 1 | 0.375833 |
| release_door_deg | 48.1304 | 48.1325 | 48.1381 | — |
| release_door_speed_deg_s | 1.8433 | 1.78725 | 1.94382 | — |
| release_frame | 478 | 478.12 | 471.64 | — |
| state_action_error_mean | 0.0653959 | 0.0676912 | 0.0870825 | 0.0727657 |
