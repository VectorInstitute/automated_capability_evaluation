# Sourced by *_eval.sh SLURM jobs. Puts platformdirs user_data / cache on local
# scratch so Inspect's samplebuffer and logging are not on flaky NFS home mounts.
# Avoid /tmp — it is often shared and fills up when vLLM writes torch compile caches.
_SCRATCH="${SLURM_TMPDIR:-}"
if [ -z "$_SCRATCH" ] || [ "$_SCRATCH" = "/tmp" ]; then
  _SCRATCH="/projects/DeepLesion/tmp_cache"
fi
export XDG_DATA_HOME="${_SCRATCH}/inspect_xdg_data"
export XDG_CACHE_HOME="${_SCRATCH}/inspect_xdg_cache"
export TMPDIR="${_SCRATCH}"
_CACHE_BASE="${_SCRATCH}/job_${SLURM_JOB_ID:-local}"
export TORCHINDUCTOR_CACHE_DIR="${_CACHE_BASE}/torchinductor_${USER:-user}"
export TRITON_CACHE_DIR="${_CACHE_BASE}/triton_${USER:-user}"
mkdir -p "$XDG_DATA_HOME" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
unset _SCRATCH
unset _CACHE_BASE
