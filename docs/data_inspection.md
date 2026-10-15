# Data inspection

Only measured numbers go here. Anything not yet measured is listed at the end.
Notebook: `notebooks/01_data_inspection.ipynb`.

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

- Disk size.
- Share of fire pixels per source, and whether any mask file is all zeros.
- Overlap by WRS path/row between the manual set and the training regions
  (the manual set is not downloaded yet).
- The figure of 8–10 patches with every mask source.
