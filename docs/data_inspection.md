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

- Per-region numbers (scenes, patches, masks) and disk size.
- Share of fire pixels per source, and whether any mask file is all zeros.
- Overlap by WRS path/row between the manual set and the training regions
  (the manual set is not downloaded yet).
- The figure of 8–10 patches with every mask source.
