#!/bin/bash
#SBATCH --partition=only-one-gpu
# RESOURCES
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --time=00:15:00
# OUTPUT FILES
#SBATCH --output=job_logs/out_%x_%j.log        # Standard output and error log, with job name and id

### Definitions
export BASEDIR="porous-cfd"
#export SHRDIR="/scratch_share/mmsp/`whoami`"
export LOCDIR="/scratch_local"
#export TMPDIR=$SHRDIR/$BASEDIR/tmp_$SLURM_JOB_NAME_$SLURM_JOB_ID

### File System Setup
cd $HOME/$BASEDIR                  # use a folder in home directory
#cd $SHRDIR/$BASEDIR                # use a folder in scratch_share
#mkdir -p $TMPDIR                   # create a folder for temporary data
#cp $HOME/<input_data> $TMPDIR      # copy input data to temp folder

### Header
pwd; hostname; date    #prints first line of output file

### Software dependencies
source /opt/share/sw/amd/gcc-8.5.0/miniforge3-24.3.0-0/etc/profile.d/conda.sh
conda init bash
conda activate porous-cfd

### Executable script
python data_generator.py --openfoam-dir $HOME/compile/OpenFOAM-v2412 --openfoam-procs 8



### File system cleanup
#cp $TMPDIR/<output_data> $HOME/$BASEDIR/job_logs/    # copy output data to output folder
#rm -r $TMPDIR                                         #clean temporary data

### Footer
date    #prints last line of output file