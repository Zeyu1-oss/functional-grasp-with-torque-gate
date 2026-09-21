#!/bin/bash
# Arm A: the gate ARCHITECTURE with NO contact supervision (contact_gate.beta = gamma = 0).
#
# What this row is for
# --------------------
# Table 4.3 shows that adding the gate on top of the torque observation is worth ~8 points. It
# does NOT show why. Two explanations are alive:
#
#   (i)  the contact supervision teaches phi to track contact, so the torque branch is admitted
#        only when it carries contact information -- the claim the thesis makes; or
#   (ii) the gated branch is simply a better-conditioned architecture. phi*f_tau + (1-phi)*f_star
#        is a learned multiplicative modulation with a learned free-space embedding, and that
#        alone can regularise regardless of what phi ends up meaning.
#
# This run separates them. Same data, same schedule, same architecture -- the only thing removed
# is the BCE term that supervises phi. If it lands near the supervised gate, the gain was
# architectural and the "contact-aware" framing has to be dropped. If it lands near the ungated
# observation row (74.3%), the supervision is what did the work.
#
# Lei et al. (arXiv:2604.01414) report the failure mode this tests for: an MoE router trained
# through the downstream loss alone assigned nearly identical weight to the torque expert with
# and without contact -- a gate left to emerge from the imitation objective need not learn to gate.
#
# What gamma=0 actually does
# --------------------------
# simple_dp3.py still builds the gate MLP, still computes phi, still mixes with f_star, and still
# computes the BCE for LOGGING -- it just multiplies it by zero before adding it to the loss, so
# no gradient reaches phi from the contact labels. The dataset must still carry data/contact
# (the diagnostics read it), which norobot.zarr does.
#
# Watch `gate/separation` in the run's logs: mean(phi | contact) - mean(phi | no contact). The
# supervised gate settles near 0.7 and its deployed rollout measures 0.68. If this run keeps it
# near zero, that is explanation (ii) visible in the logs before any deployment number is in.
#
# Pairing
# -------
# Compare against the AUX=off, gamma=0.1 run -- "Observation + gating", 78.3% -- which differs
# from this one in exactly one thing. AUX is left off here for that reason; turning it on would
# move two variables at once.
#
# Usage:
#   bash scripts/train_policy_inspire_drill_grasp_norobot_eq1_gate_unsupervised.sh
#   bash scripts/..._gate_unsupervised.sh /path/to.zarr 0 0 200      # data seed gpu epochs
#
# run_dir: data/outputs/..._norobot_eq1_global_feature_auxoff_gamma0_seed0
#          The _gamma0 suffix is what keeps this from resuming the supervised run's checkpoint;
#          without it the two land in the same directory and the second silently continues the
#          first (see the naming guard in the base script).

set -e

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_base="${_here}/train_policy_inspire_drill_grasp_norobot_eq1.sh"

if [ ! -f "${_base}" ]; then
    echo -e "\033[31m[ERROR] 找不到基脚本: ${_base}\033[0m"
    exit 1
fi

data_path=${1:-/home/zeyu/inspire_drill/data/norobot.zarr}
seed=${2:-0}
gpu_id=${3:-0}
num_epochs=${4:-200}

export AUX=off      # single variable: only the gate's supervision weight changes

echo -e "\033[36m[ARM A] contact gate WITHOUT supervision (contact_gate.beta=0).\033[0m"
echo -e "\033[36m        The gate still gates; nothing teaches phi what contact is.\033[0m"
echo -e "\033[36m        Pair with the gamma=0.1 AUX=off run (78.3%) -- one variable apart.\033[0m"
echo -e "\033[36m        Watch gate/separation: ~0.7 supervised, ~0 means the gate never learned.\033[0m"

# 5th positional is gamma -> policy.contact_gate.beta
exec bash "${_base}" "${data_path}" "${seed}" "${gpu_id}" "${num_epochs}" 0
