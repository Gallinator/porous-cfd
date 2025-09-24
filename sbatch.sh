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
inf_args=()
eval_args=(--save-plots)
data_root=""
generate_data=false

while getopts "x:r:e:i:b:o:m:n:p:s:t:v:wg" opt; do
  case $opt in
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
      generate_data=true;;
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
      train_args+=( --model "$OPTARG" )
      eval_args+=( --model "$OPTARG" );;
    n)
      train_args+=( --name "$OPTARG" )
      eval_args+=( --checkpoint "lightning_logs/$OPTARG/model.ckpt" );;
    p)
      train_args+=( --precision "$OPTARG" )
      eval_args+=( --precision "$OPTARG" );;
    s)
      train_args+=( --batch-size "$OPTARG" );;   *)
      ;;
  esac
done

# File system setup
if [ "$BASEDIR" = "$PWD" ]; then
  echo "Please provide an experiment to run with the -x argument."
  exit 1
fi

### Header
pwd; hostname; date

### Software dependencies
source /opt/share/sw/amd/gcc-8.5.0/miniforge3-24.3.0-0/etc/profile.d/conda.sh
conda init bash
conda activate porous-cfd

### Executable script
cd $BASEDIR
export PYTHONPATH="../..:."
export PYTHONUNBUFFERED=1

if [ "$generate_data" == true ]; then
  python generate_data.py "${gen_args[@]}"
fi
python train.py "${train_args[@]}"
python inference.py "${inf_args[@]}"
python evaluate.py "${eval_args[@]}"

### Footer
date

end_time=$(date +%s)
elapsed=$((($end_time-$start_time)/60))
echo "Total execution time: $elapsed minutes"