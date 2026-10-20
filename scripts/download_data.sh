#!/usr/bin/env bash
# Download parts of the ActiveFire dataset (Pereira et al., 2021) into data/raw/.
#
# Parts (pass any of them as arguments; default: all):
#   subset   the "small subset" from the original repo (100 patches; only
#            useful for smoke tests)
#   manual   the manually annotated patches (final evaluation set)
#   regions  three regions of the full dataset, Oceania (~2 GB),
#            North_America2 (~3 GB) and South_America (~13 GB), used for
#            training. Regions already downloaded are skipped.
#   weights  the authors' trained Keras weights, one zip per mask source
#            (~265 MB each, three models per zip), for R0
#
# The full dataset (~200 GB, 11 regions) is not downloaded.
# Google Drive file IDs are taken from the original repository:
#   src/utils/download_dataset.py            (SUBSET_SAMPLES, REGIONS)
#   src/utils/download_manual_annotations.py (PATCHES)
#   src/utils/download_weights.py            (WEIGHT_FILES)
#   https://github.com/pereira-gha/activefire
#
# Archives are extracted as-is, including the zips nested inside them, without
# renaming or moving files. Interpreting the layout is done in Phase 2 and
# documented in docs/data_inspection.md.
#
# Google Drive sometimes refuses or cuts off large downloads. Re-run the
# script: finished files are skipped, and a cut-off download resumes from its
# partial file (data/raw/compressed/*.part*.part). As a last resort, download
# by hand from the URLs printed below and place them in data/raw/compressed/.
#
# Usage (from the repo root):
#   scripts/download_data.sh                 # everything
#   scripts/download_data.sh regions manual  # only some parts

set -euo pipefail

GDOWN="uvx gdown@6.4.2"
RAW_DIR="data/raw"
ZIP_DIR="$RAW_DIR/compressed"

# name:Google Drive file ID
SUBSET_ID="1gwQdhXrxCybcO16vem09DfW5fPadAA_p"
MANUAL_IDS=(
    "landsat_patches:1uZnc65_GRFdAoavGoUKkJfeVQOUI8lVg"
    "masks_patches:1RCURItVvqsT_oMxlhB5NYiRp8SJ9_xZ3"
    "manual_annotations_patches:1LdsX-rH5hy_82jfc1akO8p4n0_N8lRgf"
)
REGION_IDS=(
    "Oceania:1xHYTICHKU0u3-kIrq-pM9k0YaeQt60Bt"
    "North_America2:1zW_pEIggJ5Li7uQX9XKMfHkcgL3kiUoi"
    "South_America:1hnBUmYkvFYohTf9hTnKiyp2MXVevglK7"
)
WEIGHT_IDS=(
    "voting:1__LJRdpSgPIRm8XFKzSYsCUFpfdntg8T"
    "schroeder:1gNstHSeNlT8FLd3vvJoBNu6xAP_AN7zp"
    "murphy:1eaxW2DwzXl4pKtFrzkJYXsWlmKC1Z5vE"
    "kumar-roy:1MVt2hJXL4K2_MOWQeXpVutaEso8M4MVz"
    "intersection:1WEWtFztvXkIjXBEIzNEu80VGV_lHnTCp"
)

download() {
    local id="$1" out="$2"
    if [[ -s "$out" ]]; then
        echo "Already downloaded: $out"
        return
    fi
    echo "Downloading $out (https://drive.google.com/file/d/$id/view)"
    # --continue resumes from a partial file left by an earlier run;
    # --retries resumes within this run if the connection drops.
    $GDOWN --continue --retries 5 "$id" -O "$out.part"
    mv "$out.part" "$out"
}

# Extract a zip into a directory, then extract any zips found inside it into
# a folder next to each nested zip (same name without .zip).
extract() {
    local zip="$1" dest="$2"
    if [[ -d "$dest" ]]; then
        echo "Already extracted: $dest"
        return
    fi
    echo "Extracting $zip -> $dest"
    rm -rf "$dest.tmp"
    mkdir -p "$dest.tmp"
    # Extract into a temporary folder first, so a half-finished extraction is
    # never mistaken for a finished one.
    unzip -q "$zip" -d "$dest.tmp"

    # For every zip found inside (the loop also works with spaces in names):
    find "$dest.tmp" -name '*.zip' -print0 | while IFS= read -r -d '' inner; do
        echo "  nested: $inner"
        unzip -q "$inner" -d "${inner%.zip}"  # "a/b.zip" -> into "a/b"
        rm "$inner"
    done
    mv "$dest.tmp" "$dest"
}

get_subset() {
    download "$SUBSET_ID" "$ZIP_DIR/samples.zip"
    extract "$ZIP_DIR/samples.zip" "$RAW_DIR/samples"
}

# Each entry in MANUAL_IDS, REGION_IDS and WEIGHT_IDS is "name:id".
#   ${entry%%:*}  is the part before the colon (the name)
#   ${entry#*:}   is the part after the colon (the Google Drive ID)
get_manual() {
    local entry name id
    for entry in "${MANUAL_IDS[@]}"; do
        name="${entry%%:*}"
        id="${entry#*:}"
        download "$id" "$ZIP_DIR/$name.zip"
        extract "$ZIP_DIR/$name.zip" "$RAW_DIR/manual_annotations/$name"
    done
}

get_regions() {
    local entry name id
    for entry in "${REGION_IDS[@]}"; do
        name="${entry%%:*}"
        id="${entry#*:}"
        download "$id" "$ZIP_DIR/$name.zip"
        extract "$ZIP_DIR/$name.zip" "$RAW_DIR/regions/$name"
    done
}

get_weights() {
    local entry name id
    for entry in "${WEIGHT_IDS[@]}"; do
        name="${entry%%:*}"
        id="${entry#*:}"
        download "$id" "$ZIP_DIR/weights_$name.zip"
        extract "$ZIP_DIR/weights_$name.zip" "$RAW_DIR/weights/$name"
    done
}

# The parts asked for on the command line, or all of them if none.
parts=("$@")
if [[ ${#parts[@]} -eq 0 ]]; then
    parts=(subset manual regions weights)
fi

mkdir -p "$ZIP_DIR"
for part in "${parts[@]}"; do
    case "$part" in
        subset) get_subset ;;
        manual) get_manual ;;
        regions) get_regions ;;
        weights) get_weights ;;
        *) echo "Unknown part: $part (expected subset, manual, regions or weights)" >&2; exit 1 ;;
    esac
done

echo
echo "Done. Sizes and .tif counts:"
for d in "$RAW_DIR/samples" "$RAW_DIR"/manual_annotations/* "$RAW_DIR"/regions/* "$RAW_DIR"/weights/*; do
    [[ -d "$d" ]] || continue
    printf '%8s  %8d tif  %s\n' "$(du -sh "$d" | cut -f1)" "$(find "$d" -name '*.tif' | wc -l)" "$d"
done
