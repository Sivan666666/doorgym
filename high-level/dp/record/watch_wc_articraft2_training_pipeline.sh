#!/usr/bin/env bash
set -u

CONVERT_SERVICE=${CONVERT_SERVICE:-wc-articraft2-repair-convert-200.service}
SYNC_SERVICE=${SYNC_SERVICE:-wc-articraft2-sync-train-ps1.service}
REMOTE_HOST=${REMOTE_HOST:-ps@192.168.1.104}
REMOTE_REPO=${REMOTE_REPO:-/home/ps/workspace/txc/doorgym}
POLL_SECONDS=${POLL_SECONDS:-30}

BASELINE_RUN=leroact_a2w_wc_articraft2_200_baseline_50k_chunk100_exec50_bs16_seed1000_0728
INTERACTION_RUN=leroact_a2w_wc_articraft2_200_plucker_fov55_interaction_decoderchunk_w005_50k_chunk100_exec50_bs16_seed1000_0728

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

ssh_failures=0
while true; do
  convert_state=$(systemctl --user is-active "$CONVERT_SERVICE" 2>/dev/null || true)
  sync_state=$(systemctl --user is-active "$SYNC_SERVICE" 2>/dev/null || true)
  progress=$(journalctl --user -u "$CONVERT_SERVICE" -n 160 --no-pager 2>/dev/null \
    | grep -E 'Converted episode_[0-9]+.*\([0-9]+/200\)|PIPELINE_COMPLETE' \
    | tail -n 1 || true)

  remote=$(
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE_HOST" "
      baseline='$REMOTE_REPO/high-level/dp/logs/lerobot-train/$BASELINE_RUN'
      interaction='$REMOTE_REPO/high-level/dp/logs/lerobot-train/$INTERACTION_RUN'
      baseline_ok=0
      interaction_ok=0
      test -f \"\$baseline/checkpoints/050000/pretrained_model/config.json\" && baseline_ok=1
      test -f \"\$interaction/checkpoints/050000/pretrained_model/config.json\" && interaction_ok=1
      baseline_procs=\$(ps -eo args | grep -c '[l]erobot_train.py.*$BASELINE_RUN' || true)
      interaction_procs=\$(ps -eo args | grep -c '[l]erobot_train.py.*$INTERACTION_RUN' || true)
      echo \"\$baseline_ok \$interaction_ok \$baseline_procs \$interaction_procs\"
    " 2>/dev/null
  )
  ssh_status=$?

  if (( ssh_status != 0 )); then
    ssh_failures=$((ssh_failures + 1))
    echo "[$(timestamp)] WARNING SSH check failed ($ssh_failures consecutive); retrying."
    if (( ssh_failures % 10 == 0 )); then
      echo "[$(timestamp)] ALERT PS1 remains unreachable after $ssh_failures consecutive checks; monitoring continues." >&2
    fi
    sleep "$POLL_SECONDS"
    continue
  fi
  ssh_failures=0

  read -r baseline_ok interaction_ok baseline_procs interaction_procs <<<"$remote"
  train_procs=$((baseline_procs + interaction_procs))
  echo "[$(timestamp)] convert=$convert_state sync=$sync_state baseline_50k=$baseline_ok interaction_50k=$interaction_ok baseline_processes=$baseline_procs interaction_processes=$interaction_procs progress=${progress:-none}"

  if (( baseline_ok == 1 && interaction_ok == 1 )); then
    echo "[$(timestamp)] COMPLETE both valid 50K checkpoints exist."
    exit 0
  fi
  if [[ "$convert_state" == "failed" ]]; then
    echo "[$(timestamp)] ALERT conversion service failed." >&2
    journalctl --user -u "$CONVERT_SERVICE" -n 120 --no-pager >&2
    exit 20
  fi
  if [[ "$sync_state" == "failed" ]]; then
    echo "[$(timestamp)] ALERT sync/launch service failed." >&2
    journalctl --user -u "$SYNC_SERVICE" -n 120 --no-pager >&2
    exit 21
  fi
  if [[ "$sync_state" != "active" ]] && (( baseline_ok == 0 && baseline_procs == 0 )); then
    echo "[$(timestamp)] ALERT Baseline ACT stopped before producing a valid 50K checkpoint." >&2
    exit 23
  fi
  if [[ "$sync_state" != "active" ]] && (( interaction_ok == 0 && interaction_procs == 0 )); then
    echo "[$(timestamp)] ALERT Plucker Interaction ACT stopped before producing a valid 50K checkpoint." >&2
    exit 24
  fi
  if [[ "$convert_state" != "active" && "$sync_state" != "active" ]] && (( train_procs == 0 )); then
    echo "[$(timestamp)] ALERT pipeline became inactive before both checkpoints completed." >&2
    exit 22
  fi

  sleep "$POLL_SECONDS"
done
