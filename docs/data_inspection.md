# Data inspection

Only measured numbers go here. Anything not yet measured is listed at the end.
Notebook: `notebooks/01_data_inspection.ipynb`.

## The three regions (Oceania + North_America2 + South_America)

Measured on the M3 Max, 2026-10-15, from `data/splits/scene.csv` and
`data/splits/manual.csv` (written by `python -m activefire_torch.splits`).

| Region | Patches | Scenes | Locations | Schroeder | Murphy | Kumar-Roy | Voting | Intersection | On disk |
|---|---|---|---|---|---|---|---|---|---|
| North_America2 | 3,031 | 369 | 314 | 610 | 1,156 | 2,619 | 808 | 452 | 4.1 GB |
| Oceania | 2,200 | 229 | 190 | 821 | 1,282 | 2,060 | 1,162 | 752 | 3.1 GB |
| South_America | 11,875 | 354 | 302 | 5,844 | 8,266 | 11,640 | 8,083 | 5,696 | 17 GB |
| All | 17,106 | 952 | 806 | 7,275 | 10,704 | 16,319 | 10,053 | 6,900 | 24.2 GB |

- Patches per scene: 18.0 on average, from 1 to 365. South America has far
  more fire per scene (33.5 patches per scene, against 8–10 elsewhere).
- Every patch has a mask from at least one algorithm (17,106 of 17,106), as
  with two regions (D16).
- Voting masks cover 68% of South America's patches, against 27%
  (North_America2) and 53% (Oceania).
- No location (path/row) appears in more than one region, so splitting each
  region separately cannot put one location in two splits.

### Split S (by location), all patches

| Region | Train | Val | Test | Total |
|---|---|---|---|---|
| North_America2 | 1,197 | 228 | 1,606 | 3,031 |
| Oceania | 1,037 | 127 | 1,036 | 2,200 |
| South_America | 4,457 | 1,331 | 6,087 | 11,875 |
| All | 6,691 (39%) | 1,686 (10%) | 8,729 (51%) | 17,106 |

### Split S, patches with a voting mask

| Region | Train | Val | Test | Total |
|---|---|---|---|---|
| North_America2 | 312 | 39 | 457 | 808 |
| Oceania | 528 | 89 | 545 | 1,162 |
| South_America | 3,088 | 1,017 | 3,978 | 8,083 |
| All | 3,928 | 1,145 | 4,980 | 10,053 |

### Manual set

- 9,044 patches from 13 scenes at 13 locations. Only **100** patches have a
  manual mask (a mask file exists only where there is fire); 2,091 patches
  are in the 3 fire-free scenes.
- **Overlap with the training regions:** 2 of the 13 locations.
  - `003006` (northern Greenland, probably the fire-free Greenland scene):
    35 patches in North_America2.
  - `105069`: 9 patches in Oceania.
- Both locations fall in Split S **test**, so no Stage 2 model trains on
  them. Split P (Stage 1) assigns patches at random, so some of these 44
  patches end up in Stage 1 training, as they would have in the original;
  this is reported next to the comparison with the paper.

## The two regions (Oceania + North_America2)

Measured with the notebook on the M3 Max, 2026-10-15.

| | |
|---|---|
| Scenes | 598 |
| Image patches | 5,231 |
| Patches per scene | 8.7 on average, from 1 to 119 |
| Images with a mask from at least one of Schroeder, Murphy, Kumar-Roy | 5,231 of 5,231 |

A full Landsat-8 scene is about 7,600 × 7,600 pixels, which would give about
880 patches of 256 × 256. With 8.7 per scene, and every image having at least
one mask, **the dataset only keeps patches where at least one algorithm found
fire** (D16).

Mask files per source (a mask file exists only where that source found fire):

| Source | Mask files | Share of the 5,231 images |
|---|---|---|
| Kumar-Roy | 4,679 | 89% |
| Murphy | 2,438 | 47% |
| Voting (at least 2 of 3 agree) | 1,970 | 38% |
| Schroeder | 1,431 | 27% |
| Intersection (all 3 agree) | 1,204 | 23% |

Voting and intersection masks are included in the download, so they do not
have to be built from the three algorithms.

The 598 scenes cover **504 locations** (WRS path/rows), so 94 scenes are a
second picture of a location already in the list, on another date (D15).

### Split S (by location), all patches

| Region | Train | Val | Test | Total |
|---|---|---|---|---|
| North_America2 | 1,197 (39%) | 228 (8%) | 1,606 (53%) | 3,031 |
| Oceania | 1,037 (47%) | 127 (6%) | 1,036 (47%) | 2,200 |
| Both | 2,234 (43%) | 355 (7%) | 2,642 (51%) | 5,231 |

The 40 / 10 / 50 is counted in locations, and locations have very different
numbers of patches (a scene has 1 to 119), so the patch shares drift. The
validation split came out smallest: 7% instead of 10%.

### Split S, patches with a voting mask (the Stage 1 and T1 training set)

| Region | Train | Val | Test | Total |
|---|---|---|---|---|
| North_America2 | 312 | 39 | 457 | 808 |
| Oceania | 528 | 89 | 545 | 1,162 |
| Both | 840 | 128 | 1,002 | 1,970 |

Voting masks are much more common in Oceania (1,162 of 2,200 patches, 53%)
than in North_America2 (808 of 3,031, 27%): the three algorithms agree more
often there.

### Open question for Checkpoint A: the small validation set

**Resolved:** South_America was added (option 3, D17); the split is kept as is (D19) and T1b changed to leave-one-region-out (D20). The text below is the question as it stood with two regions.

With voting masks, Split S leaves 128 validation patches in total, and only
39 for North_America2. That matters most for T1b: training on North America
alone means 312 training patches and choosing the stopping point and
threshold from 39. Stage 1 (Split P) is not affected: it splits patches
directly, so it gets exactly 40 / 10 / 50.

Options:

1. **Keep the split as is.** Stage 1 and T1 work; the North America → Oceania
   direction of T1b will probably be inconclusive, and is reported that way.
2. **Balance the location split by patch count.** Assign whole locations so
   the patch shares come close to 40 / 10 / 50 in each region. This would
   bring North America's validation to about 80 voting patches. The rule is
   fixed before any training result exists.
3. **Add a third region.** More data for every split; costs a download and a
   re-run of the inspection and the split.

Not an option: trying seeds until the split looks good. Choosing a seed by its
outcome is a habit worth avoiding, even when the outcome is only patch
counts.

## The 100-patch sample (`samples.zip`)

Measured on this machine, 2026-10-13 to 2026-10-14.

- 100 image patches from 96 scenes, taken 2020-08-06 to 2020-08-30.
- All 100 have Schroeder, Murphy and Kumar-Roy masks (Kumar-Roy is named
  `GOLI_v2` here). No voting or intersection masks.
- Images: 10 bands, `uint16`, 256 × 256, georeferenced (UTM, 30 m pixels),
  no nodata value set.
- Masks: `uint8`, values 0 and 1, on the same grid as the images.
- Band order (consistent with `paper_notes.md` §2, file bands = B1–B7, B9,
  B10, B11):
  - The two thermal bands (file bands 9 and 10) are the smoothest.
  - Cirrus (file band 8, B9) is the darkest band on 98% of patches.
  - Fire pixels vs. the rest: about 4× brighter in B7, 1.5–1.9× in B6, and
    about the same (1.07×) in the visible bands.
- Raw values (B7, B6, B2): averages about 10,000–13,000, spreads about
  4,000–5,000, and some pixels at the maximum, 65,535.

## Not yet measured

- Share of fire pixels per source, and whether any mask file is all zeros.
- The figure of 8–10 patches with every mask source.
