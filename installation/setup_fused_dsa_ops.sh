#!/bin/bash
# Build and install the fused compressed lightning indexer (cmp_ratio) from CANN ops-transformer.
#
# Why: CANN 9.0.1 ships only aclnnLightningIndexer (V1), which has no cmp_ratio, so
# --use-fused-lightning-indexer cannot work on cr=4 layers. This builds aclnnLightningIndexerV2 +
# V2Metadata from source as a custom op package, plus the cann_ops_transformer PyTorch wheel.
# See FUSED_DSA_OPS.md for the full story.
#
# Run INSIDE the training/test container (it writes to the container's CANN tree):
#   bash installation/setup_fused_dsa_ops.sh
#
# Requires: the ops-transformer checkout on branch 9.2.0 or master (9.0.0/9.1.0 lack the op or its
# binding), gcc >= 9.4, cmake, ninja, network access for third-party deps on the first build.
#
# Env overrides: REPO, VENDOR, SOC, OPS, SKIP_KERNEL=1, SKIP_WHEEL=1

set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO="${REPO:-$ROOT/ops-transformer}"
VENDOR="${VENDOR:-qlora}"
SOC="${SOC:-ascend910b}"          # Atlas A2 (910B3); A3 is ascend910_93
OPS="${OPS:-lightning_indexer_v2,lightning_indexer_v2_metadata}"
ARCH="$(uname -m)"
VENDOR_DIR_NAME="${VENDOR}_transformer"   # build.sh appends _transformer

die() { echo "FAILED: $*" >&2; exit 1; }
step() { echo; echo "=== $* ==="; }

# ---------------------------------------------------------------- preflight
step "Preflight"
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH is unset -- source the CANN set_env.sh first"
echo "CANN:   $ASCEND_HOME_PATH"
[ -d "$REPO" ] || die "no ops-transformer checkout at $REPO"
[ -d "$REPO/attention/lightning_indexer_v2" ] || die \
  "$REPO has no attention/lightning_indexer_v2 -- wrong branch (need 9.2.0 or master)"
echo "repo:   $REPO ($(cd "$REPO" && git log --oneline -1 2>/dev/null))"
for t in gcc cmake ninja python3; do command -v "$t" >/dev/null || die "$t not found"; done
echo "gcc:    $(gcc -dumpversion)   cmake: $(cmake --version | head -1 | awk '{print $3}')"
echo "python: $(python3 -V 2>&1)"
VENDORS_DIR="$ASCEND_HOME_PATH/opp/vendors"
[ -w "$VENDORS_DIR" ] || die "$VENDORS_DIR is not writable"

# ---------------------------------------------------------------- 1. kernel
if [ "${SKIP_KERNEL:-0}" != "1" ]; then
  step "1/4 Building custom op package ($OPS, soc=$SOC)"
  echo "This downloads third-party deps on the first run (needs the proxy) and takes a few minutes."
  cd "$REPO" || die "cd $REPO"
  bash build.sh --pkg --soc="$SOC" --vendor_name="$VENDOR" --ops="$OPS" > build_li.log 2>&1 \
    || { tail -25 build_li.log; die "kernel build failed, see $REPO/build_li.log"; }
  RUN="$REPO/build_out/cann-ops-transformer-${VENDOR}_linux-${ARCH}.run"
  [ -f "$RUN" ] || die "expected package not produced: $RUN"
  echo "built: $RUN ($(du -h "$RUN" | cut -f1))"

  step "2/4 Installing custom op package"
  "$RUN" --quiet 2>&1 | tail -3 | sed 's/^/  /'
  grep -q "$VENDOR_DIR_NAME" "$VENDORS_DIR/config.ini" \
    || die "$VENDOR_DIR_NAME missing from $VENDORS_DIR/config.ini -- check ownership of $VENDORS_DIR"
  echo "  $(grep load_priority "$VENDORS_DIR/config.ini")"

  LIB="$VENDORS_DIR/$VENDOR_DIR_NAME/op_api/lib/libcust_opapi.so"
  [ -f "$LIB" ] || die "installed lib not found: $LIB"
  SYMS=$(nm -D --defined-only "$LIB" 2>/dev/null | grep -c "aclnnLightningIndexerV2")
  [ "$SYMS" -ge 2 ] || die "libcust_opapi.so does not export aclnnLightningIndexerV2*"
  echo "  symbols OK: $(nm -D --defined-only "$LIB" | grep -o 'aclnnLightningIndexerV[0-9]' | sort -u | tr '\n' ' ')"
else
  step "1-2/4 Kernel build/install skipped (SKIP_KERNEL=1)"
fi

# ---------------------------------------------------------------- 2. wheel
if [ "${SKIP_WHEEL:-0}" != "1" ]; then
  step "3/4 Building cann_ops_transformer wheel"
  # 'build' is required; without it build.sh silently skips the wheel. Check it from a neutral cwd and
  # look for build.__main__: $REPO contains a cmake 'build/' directory which otherwise satisfies a bare
  # "import build" as a namespace package, hiding the fact that the real module is missing.
  (cd / && python3 -c "import build.__main__" 2>/dev/null) \
    || { echo "  installing python 'build'..."; pip install -q build || die "pip install build failed"; }
  cd "$REPO" || die "cd $REPO"
  # Full wheel (no --ops): import path stays 'cann_ops_transformer', which MindSpeed expects.
  bash build.sh --torch_extension > build_wheel.log 2>&1 \
    || { tail -25 build_wheel.log; die "wheel build failed, see $REPO/build_wheel.log"; }
  WHL=$(ls -t "$REPO"/build_out/cann_ops_transformer-*.whl 2>/dev/null | head -1)
  [ -n "$WHL" ] || die "no wheel produced (is python 'build' installed?)"
  echo "  built: $WHL"

  step "4/4 Installing wheel"
  pip install -q "$WHL" --force-reinstall --no-deps || die "pip install of the wheel failed"
  python3 - <<'PY' || die "cann_ops_transformer does not expose the indexer entry points"
import cann_ops_transformer.ops as o
assert hasattr(o, "lightning_indexer") and hasattr(o, "lightning_indexer_metadata")
print("  import OK:", o.__name__, "-> lightning_indexer, lightning_indexer_metadata")
PY
else
  step "3-4/4 Wheel build/install skipped (SKIP_WHEEL=1)"
fi

# ---------------------------------------------------------------- done
cat <<EOF

=== Done ===

The kernel is resolved by dlsym at call time, so one of these must be set for every run that uses it:

  export ASCEND_CUSTOM_OPP_PATH=$VENDORS_DIR/$VENDOR_DIR_NAME
  # or:  export LD_LIBRARY_PATH=$VENDORS_DIR/$VENDOR_DIR_NAME/op_api/lib/:\${LD_LIBRARY_PATH}

Keep JIT artifacts out of root's home:

  export TORCH_EXTENSIONS_DIR=$ROOT/.torch_extensions

Verify with the standalone correctness check (expects an idle NPU):

  ASCEND_RT_VISIBLE_DEVICES=0 SEQ=4096 python <scripts_qlora>/test_fused_li.py   # copy it over first

This install is container-local: it lives in the container overlay and is lost if the container is
recreated. To remove it: delete $VENDORS_DIR/$VENDOR_DIR_NAME and drop
'$VENDOR_DIR_NAME' from the load_priority line in $VENDORS_DIR/config.ini.
EOF
