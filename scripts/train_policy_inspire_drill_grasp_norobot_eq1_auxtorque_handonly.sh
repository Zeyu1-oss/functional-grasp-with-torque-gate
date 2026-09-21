#!/bin/bash
# Eq.1 Torque Gating + auxiliary torque prediction, BOTH restricted to the 6 hand joints.
#
# Two independent knobs, both set to hand-only here:
#   SCOPE=finger_single   -- the GATED observation branch only carries the 6 hand-torque dims
#                            into force_mlp/the gate (arm torque dropped from that branch
#                            entirely); this mode already existed in eq1.sh.
#   AUX_START=20          -- the auxiliary future-torque TARGET is agent_pos[20:26] (hand only)
#                            instead of [13:26] (all 13); this is the new knob added to eq1.sh
#                            alongside this script.
#
# Why both: the contact-onset Delta-tau audit (tools/plot_contact_torque_delta.py) found the arm
# torque channel is dominated by configuration/controller effort rather than contact (~3.6-3.8x
# its own empty-hand baseline at contact, vs ~22-248x for the hand channel). If arm torque is
# mostly noise wherever it enters the model, the same argument applies to both the gated
# observation branch and the auxiliary prediction target -- this run drops it from both at once
# rather than testing the two independently.
#
# If you specifically want ONLY the auxiliary target narrowed and the gate's observation branch
# left at its default (all 13, SCOPE=global) -- e.g. to isolate which of the two knobs matters --
# override SCOPE back to global when invoking this script:
#   SCOPE=global bash scripts/train_policy_inspire_drill_grasp_norobot_eq1_auxtorque_handonly.sh
#
# Pair against:
#   train_..._norobot_eq1_auxtorque.sh          (SCOPE=global,        AUX_START=13, the reported 79.7% "ours")
#   train_..._norobot_torqueboth_handaux.sh      (no gate, obs=26, AUX_START=20 -- the ungated counterpart)
#
# run_dir (via eq1.sh's naming): ..._norobot_eq1_finger_single_feature_auxon_auxstart20_seed${seed}
#
# Usage (same positional args as eq1.sh, passed through unchanged):
#   bash scripts/train_policy_inspire_drill_grasp_norobot_eq1_auxtorque_handonly.sh [data_path] [seed] [gpu_id] [epochs] [gamma]
#   AUX_BETA=0.05 bash scripts/..._handonly.sh    # change the auxiliary loss weight (default 0.1)
#
# Must be run from the repo root (eq1.sh has a cd guard).

set -e

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_base="${_here}/train_policy_inspire_drill_grasp_norobot_eq1.sh"

if [ ! -f "${_base}" ]; then
    echo -e "\033[31m[ERROR] 找不到基脚本: ${_base}\033[0m"
    exit 1
fi

export AUX=on
export AUX_BETA="${AUX_BETA:-0.1}"
export AUX_START=20
export SCOPE="${SCOPE:-finger_single}"

echo -e "\033[36m[AUX+GATE, HAND-ONLY] gate observation scope=${SCOPE}, auxiliary torque target=agent_pos[20:26] (6 hand joints).\033[0m"
echo -e "\033[36m                      beta=${AUX_BETA}. Everything else follows $(basename "${_base}").\033[0m"

exec bash "${_base}" "$@"
