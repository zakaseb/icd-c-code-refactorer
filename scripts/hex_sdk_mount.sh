# Sourced by the serve scripts. Sets HEX_DOCKER_ARGS to a bind-mount of
# the host HEX / VirtuosoNext SDK plus HEX_SDK_DIR inside the container.
#
# The application image does not contain the SDK (it is ~415 MB and
# gitignored). Without this mount, api/hex_sdk.py running in the
# container cannot see VisualDesigner-HEX-* and every file that
# includes <L1_api.h> fails the compile gate.
#
# Requires PROJECT_ROOT. Leaves HEX_DOCKER_ARGS empty when no SDK is
# present so a machine without the RTOS SDK still starts.

HEX_DOCKER_ARGS=""

_hex_host=""
if [ -n "${HEX_SDK_DIR:-}" ] && [ -d "${HEX_SDK_DIR}/targets" ]; then
  _hex_host="$HEX_SDK_DIR"
else
  for _cand in "$PROJECT_ROOT"/VisualDesigner-HEX-* \
               "$PROJECT_ROOT"/VirtuosoNext* \
               "$PROJECT_ROOT"/HEX-* \
               "$PROJECT_ROOT"/hex-sdk; do
    if [ -d "$_cand/targets" ]; then
      _hex_host="$_cand"
      break
    fi
  done
fi

if [ -n "$_hex_host" ]; then
  echo "HEX SDK: $_hex_host -> /opt/hex-sdk (read-only)"
  HEX_DOCKER_ARGS="-v ${_hex_host}:/opt/hex-sdk:ro -e HEX_SDK_DIR=/opt/hex-sdk"
fi

unset _hex_host _cand
