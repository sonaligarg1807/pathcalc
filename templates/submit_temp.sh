#!/bin/bash
#$$ -cwd -pe nproc $nproc
#$$ -N $job_name
#$$ -l mem=$memory M
#$$ -e "./$$JOB_ID.err"
#$$ -o "./$$JOB_ID.out"

scratch=/scratch/$$USER/$$JOB_ID
current=$$PWD

filename='$filename'

echo "Job started on $$HOSTNAME at $$(date) in $$scratch"
echo " "

mkdir -p $$scratch
cp * $$scratch
cd $$scratch

export GMXLIB=/data/fghalami/gromacs-sh-old_Eik/test_plumed/gromacs-sh-old/COUPLED-DYNAMICS/share/top
export LIBRARY_PATH=$$LIBRARY_PATH:/usr/local/lib
export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:/usr/local/lib
export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:/data/fghalami/gromacs-sh-old_Eik/gromacs-sh-old_Eik/test_plumed/gromacs-sh-old/COUPLED-DYNAMICS/release-tomas-jan2023/lib
export LD_LIBRARY_PATH=/usr/local/run/plumed-2.5.1/lib:$$LD_LIBRARY_PATH


$grompp_path -f $mdp -c $gro -p $top -o $filename -maxwarn 1
$mdrun_path -ntomp $ntomp -deffnm $filename >gmx.out

cp -r * $$current
cd .. && rm -rf $$scratch
cd $$current
rm pp.out submit.sh .spec .cpt .tpr .mdp .dat

echo "FINISHED at $$(date)"