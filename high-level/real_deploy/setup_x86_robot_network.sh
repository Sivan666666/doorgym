#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-/home/robo/txc/door_act_deploy}"
REPO_ROOT="${REPO_ROOT:-${DEPLOY_ROOT}/visual_whole_body}"
PROFILE_NAME="${ROBOT_NETWORK_PROFILE:-door-robot-124}"
ROBOT_ADDRESS="${ROBOT_NETWORK_ADDRESS:-192.168.124.25/24}"
IFACE="${1:-}"

if [[ -z "${IFACE}" ]]; then
  mapfile -t active_ifaces < <(
    for carrier in /sys/class/net/*/carrier; do
      [[ -r "${carrier}" ]] || continue
      iface="$(basename "$(dirname "${carrier}")")"
      [[ "${iface}" == "lo" ]] && continue
      [[ "$(cat "${carrier}")" == "1" ]] || continue
      [[ "${iface}" == wl* ]] && continue
      printf '%s\n' "${iface}"
    done
  )
  if [[ "${#active_ifaces[@]}" -ne 1 ]]; then
    echo "Expected exactly one connected wired interface, found: ${active_ifaces[*]:-none}" >&2
    echo "Usage: $0 <interface>, for example: $0 enp3s0" >&2
    exit 2
  fi
  IFACE="${active_ifaces[0]}"
fi

if [[ ! -d "/sys/class/net/${IFACE}" ]]; then
  echo "Unknown network interface: ${IFACE}" >&2
  exit 2
fi
if [[ ! -r "/sys/class/net/${IFACE}/carrier" ]] || [[ "$(cat "/sys/class/net/${IFACE}/carrier")" != "1" ]]; then
  echo "${IFACE} has no Ethernet carrier; connect the robot-network cable first." >&2
  exit 3
fi

if nmcli -t -f NAME connection show | grep -Fxq "${PROFILE_NAME}"; then
  sudo nmcli connection modify "${PROFILE_NAME}" \
    connection.interface-name "${IFACE}" \
    connection.autoconnect yes \
    ipv4.method manual \
    ipv4.addresses "${ROBOT_ADDRESS}" \
    ipv4.gateway "" \
    ipv4.dns "" \
    ipv4.never-default yes \
    ipv6.method disabled
else
  sudo nmcli connection add \
    type ethernet \
    ifname "${IFACE}" \
    con-name "${PROFILE_NAME}" \
    connection.autoconnect yes \
    ipv4.method manual \
    ipv4.addresses "${ROBOT_ADDRESS}" \
    ipv4.never-default yes \
    ipv6.method disabled
fi
sudo nmcli connection up "${PROFILE_NAME}"

mkdir -p "${DEPLOY_ROOT}/config"
sed "s/ROBOT_INTERFACE/${IFACE}/g" \
  "${REPO_ROOT}/high-level/real_deploy/cyclonedds_x86_template.xml" \
  > "${DEPLOY_ROOT}/config/cyclonedds.xml"

echo "robot_interface=${IFACE}"
echo "robot_address=${ROBOT_ADDRESS}"
echo "cyclonedds=${DEPLOY_ROOT}/config/cyclonedds.xml"
ip -br address show dev "${IFACE}"
