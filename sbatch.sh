#!/bin/bash
#SBATCH --partition=only-one-gpu
# RESOURCES
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
# OUTPUT FILES
#SBATCH --output=job_logs/out_%x_%j.log

start_time=$(date +%s)

### Definitions
export LOCDIR="/scratch_local"
export BASEDIR=$PWD

gen_args=( --openfoam-dir /usr/lib/openfoam/openfoam2412 --openfoam-procs 8 )
train_args=()
eval_args=(--save-plots)
data_root=""
generate_data=true
container_path=""

while getopts "c:x:r:e:i:b:o:m:n:p:s:t:v:wga" opt; do
  case $opt in
    a)
      gen_args+=( --meta-only );;
    c)
      container_path="$OPTARG/";;
    x)
      BASEDIR="$BASEDIR/examples/$OPTARG";;
    r)
      gen_args+=( --data-root-dir "$OPTARG" )
      data_root="$OPTARG/"
      ;;
    t)
      train_dir="$data_root$OPTARG"
      train_args+=( --train-dir "$train_dir" )
      eval_args+=(--meta-dir  "$train_dir")
      ;;
    v)
      train_args+=( --val-dir "$data_root$OPTARG" );;
    w)
      eval_args+=( --data-dir "$data_root$OPTARG" );;
    g)
      generate_data=false;;
    e)
      train_args+=( --epochs "$OPTARG" );;
    i)
      train_args+=( --n-internal "$OPTARG" )
      eval_args+=( --n-internal "$OPTARG" );;
    b)
      train_args+=( --n-boundary "$OPTARG" )
      eval_args+=( --n-boundary "$OPTARG" );;
    o)
      train_args+=( --n-observations "$OPTARG" )
      eval_args+=( --n-observations "$OPTARG" );;
    m)
      train_args+=( --model "$OPTARG" );;
    n)
      train_args+=( --name "$OPTARG" )
      eval_args+=( --checkpoint "lightning_logs/$OPTARG/model.ckpt" );;
    p)
      train_args+=( --precision "$OPTARG" )
      eval_args+=( --precision "$OPTARG" );;
    s)
      train_args+=( --batch-size "$OPTARG" );;
    *)
      ;;
  esac
done

# File system setup
if [ "$BASEDIR" = "$PWD" ]; then
  echo "Please provide an experiment to run with the -x argument."
  exit 1
fi

if [ "$container_path" = "" ]; then
  echo "Please provide a Singularity container with the -c argument."
  exit 1
fi

### Header
pwd; hostname; date

### Software dependencies
module load amd/gcc-8.5.0/openmpi-4.1.6
module load intel/nvidia/cuda-12.3.2

### Executable script
cd $BASEDIR
export PYTHONPATH="$(dirname $(dirname $PWD)):$PWD"
export PYTHONUNBUFFERED=1

if [ "$generate_data" == true ]; then
  singularity exec "$container_path" python generate_data.py "${gen_args[@]}"
fi
singularity exec --nv "$container_path" python train.py "${train_args[@]}"
singularity exec --nv "$container_path" python inference.py "${eval_args[@]}"
singularity exec --nv "$container_path" python evaluate.py "${eval_args[@]}"

### Footer
date

end_time=$(date +%s)
elapsed=$((($end_time-$start_time)/60))
echo "Total execution time: $elapsed minutes"