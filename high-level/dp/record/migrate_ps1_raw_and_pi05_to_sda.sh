#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ps/workspace/txc/doorgym
SDA_ROOT=/mnt/sda/ps/doorgym_archive
RAW_SRC="$REPO/high-level/data/door_dp_raw"
RAW_DST=/mnt/sda/ps/doorgym_data/door_dp_raw
PI_SRC="$REPO/high-level/dp/logs/lerobot-train"
PI_DST="$SDA_ROOT/lerobot-train-pi05"
LOG_DIR="$REPO/high-level/logs/storage-migration/20260807_raw_pi05"

mkdir -p "$LOG_DIR" "$RAW_DST" "$PI_DST"
exec >>"$LOG_DIR/migration.log" 2>&1

stage() {
    printf '%s %s\n' "$(date '+%F %T')" "$*" | tee "$LOG_DIR/status.txt"
}

stage MOVE_DOOR_DP_RAW
if [[ -L "$RAW_SRC" ]]; then
    current=$(readlink -f "$RAW_SRC")
    [[ "$current" == "$RAW_DST" ]] || {
        echo "Unexpected raw symlink target: $current" >&2
        exit 1
    }
else
    shopt -s dotglob nullglob
    for entry in "$RAW_SRC"/*; do
        name=$(basename "$entry")
        destination="$RAW_DST/$name"
        if [[ -L "$entry" && "$(readlink -f "$entry")" == "$(realpath -m "$destination")" ]]; then
            # This item was already migrated independently; remove only the
            # redundant child symlink before replacing the parent directory.
            rm -- "$entry"
        elif [[ -e "$destination" || -L "$destination" ]]; then
            echo "Refusing to merge conflicting raw entry: $destination" >&2
            exit 1
        else
            echo "MOVE_RAW $entry -> $destination"
            mv -- "$entry" "$destination"
        fi
    done
    rmdir "$RAW_SRC"
    ln -s "$RAW_DST" "$RAW_SRC"
fi

stage MOVE_OLD_PI05_CHECKPOINTS
shopt -s nullglob
mapfile -t pi_runs < <(
    find "$PI_SRC" -mindepth 1 -maxdepth 1 -type d \
        \( -iname '*pi05*' -o -iname '*pi0*5*' \) -print | sort -u
)
for run in "${pi_runs[@]}"; do
    [[ -d "$run/checkpoints" ]] || continue
    run_name=$(basename "$run")
    mapfile -t checkpoints < <(
        find "$run/checkpoints" -mindepth 1 -maxdepth 1 -type d \
            -regextype posix-extended -regex '.*/[0-9]+' -printf '%f\n' | sort -V
    )
    ((${#checkpoints[@]} > 0)) || continue
    latest=${checkpoints[$((${#checkpoints[@]} - 1))]}
    echo "KEEP_LATEST $run_name/$latest"
    printf '%s latest=%s\n' "$run_name" "$latest" >>"$LOG_DIR/pi05_latest_manifest.txt"
    for step in "${checkpoints[@]:0:$((${#checkpoints[@]} - 1))}"; do
        source="$run/checkpoints/$step"
        destination="$PI_DST/$run_name/checkpoints/$step"
        mkdir -p "$(dirname "$destination")"
        [[ ! -e "$destination" && ! -L "$destination" ]] || {
            echo "Refusing to overwrite archived checkpoint: $destination" >&2
            exit 1
        }
        echo "MOVE_PI05 $source -> $destination"
        mv -- "$source" "$destination"
        ln -s "$destination" "$source"
    done
done

stage VERIFY
[[ -L "$RAW_SRC" && "$(readlink -f "$RAW_SRC")" == "$RAW_DST" ]]
find "$PI_SRC" -mindepth 3 -maxdepth 3 -type l \
    \( -path '*pi05*/checkpoints/*' -o -path '*pi0*5*/checkpoints/*' \) \
    -exec test -e {} \;
df -h / /mnt/sda | tee "$LOG_DIR/final_df.txt"
du -shL "$RAW_SRC" | tee "$LOG_DIR/final_raw_size.txt"

stage COMPLETE
