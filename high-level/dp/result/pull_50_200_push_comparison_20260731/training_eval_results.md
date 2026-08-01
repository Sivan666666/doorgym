# Pull dataset scaling diagnosis

| Dataset | Model | Episodes | Train steps | Effective epochs | Door open | Traversal | Joint success |
|---|---|---:|---:|---:|---:|---:|---:|
| Pull50Old | Baseline | 50 | 50K | 18.29 | 27/64 = 42.19% | 26/64 = 40.62% | **25/64 = 39.06%** |
| Pull50Old | Plucker+Interaction | 50 | 50K | 18.29 | 12/64 = 18.75% | 12/64 = 18.75% | **11/64 = 17.19%** |
| Pull50Nested | Baseline | 50 | 50K | 18.29 | 24/64 = 37.50% | 22/64 = 34.38% | **21/64 = 32.81%** |
| Pull50Nested | Plucker+Interaction | 50 | 50K | 18.29 | 20/64 = 31.25% | 17/64 = 26.56% | **17/64 = 26.56%** |
| Pull200 | Baseline | 200 | 50K | 4.57 | 19/64 = 29.69% | 19/64 = 29.69% | **19/64 = 29.69%** |
| Pull200 | Plucker+Interaction | 200 | 50K | 4.57 | 7/64 = 10.94% | 9/64 = 14.06% | **6/64 = 9.38%** |
| Pull200 | Baseline | 200 | 100K | 9.14 | 19/64 = 29.69% | 15/64 = 23.44% | **12/64 = 18.75%** |
| Pull200 | Plucker+Interaction | 200 | 100K | 9.14 | 4/64 = 6.25% | 5/64 = 7.81% | **4/64 = 6.25%** |
| Pull200 | Baseline | 200 | 150K | 13.71 | 22/64 = 34.38% | 21/64 = 32.81% | **21/64 = 32.81%** |
| Pull200 | Plucker+Interaction | 200 | 150K | 13.71 | 11/64 = 17.19% | 11/64 = 17.19% | **11/64 = 17.19%** |
| Pull200 | Baseline | 200 | 200K | 18.29 | 28/64 = 43.75% | 27/64 = 42.19% | **27/64 = 42.19%** |
| Pull200 | Plucker+Interaction | 200 | 200K | 18.29 | 7/64 = 10.94% | 7/64 = 10.94% | **7/64 = 10.94%** |
