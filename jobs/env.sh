# Shared setup of the SLURM jobs. Submit from the repo root, e.g.
#   IMAGE=geneformer sbatch jobs/job_train scfm=geneformer-v2 data=pancreas sae.target_layer=5
# Arguments after the job script are Hydra overrides.
#   REPO        repo checkout (default: the submit directory)
#   CONTAINERS  folder with the Singularity images scgpt.sif, scfoundation.sif, geneformer.sif
#               (default: $REPO/containers)
#   IMAGE       image of the job (scgpt, scfoundation or geneformer); scripts that load an scFM
#               need the matching one
#   BINDS       extra folders to mount, comma-separated, e.g. the target of a data/ symlink that
#               points outside the repo (the container only sees the repo otherwise)
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
CONTAINERS=${CONTAINERS:-$REPO/containers}
IMAGE=${IMAGE:-scgpt}

run() {  # run IMAGE CMD...: CMD inside $CONTAINERS/IMAGE.sif with the repo mounted (GPUs when allocated)
  local image=$1; shift
  singularity run --pwd "$REPO" --bind "$REPO:$REPO${BINDS:+,$BINDS}" ${CUDA_VISIBLE_DEVICES:+--nv} "$CONTAINERS/$image.sif" "$@"
}
