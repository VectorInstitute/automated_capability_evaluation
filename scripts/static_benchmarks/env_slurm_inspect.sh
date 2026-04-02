# Sourced by *_eval.sh SLURM jobs. Puts platformdirs user_data / cache on local
# scratch so Inspect's samplebuffer and logging are not on flaky NFS home mounts.
if [ -n "${SLURM_TMPDIR:-}" ]; then
  export XDG_DATA_HOME="${SLURM_TMPDIR}/inspect_xdg_data"
  export XDG_CACHE_HOME="${SLURM_TMPDIR}/inspect_xdg_cache"
  mkdir -p "$XDG_DATA_HOME" "$XDG_CACHE_HOME"
fi
