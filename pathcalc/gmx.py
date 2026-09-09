import subprocess as sp 

class gmx:
    def __init__(self, exe: str, **kwargs):
        self.exe = exe
        self.top = None
        self.gro = None
        self.mdp = None
        self.tpr = None
        self.ndx = None
        self.nt = 1

        for k, v in kwargs.items():
            if isinstance(v, str):
                if ".top" in v:
                    self.top = v
                elif ".gro" in v:
                    self.gro = v
                elif ".tpr" in v:
                    self.tpr = v
                elif ".mdp" in v:
                    self.mdp = v
                elif ".ndx" in v:
                    self.ndx = v
                else:
                    setattr(self, k, v)
            else:
                setattr(self, k, v)

    def renum(self):
        run = f"{self.exe} editconf -f {self.gro} -resnr 1 -o {self.gro}"
        return sp.run(run, check=True, shell=True)

    def grompp(self):
        if self.ndx:
            run = f"{self.exe}/grompp -c {self.gro} -p {self.top} -f {self.mdp} -n {self.ndx} -o {self.tpr} -maxwarn 1"
        else:
            run = f"{self.exe}/grompp -c {self.gro} -p {self.top} -f {self.mdp} -o {self.tpr} -maxwarn 1"
        return sp.run(run, check=True, shell=True)

    def mdrun(self, nt):
        if nt is None:
            nt = self.nt
            
        env_setup_cmd = ["export GMXLIB=/home/fghalami/GMX/Gromacs-SH/COUPLED-DYNAMICS/share/top",
                        "export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib", "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib",
                        "export LD_LIBRARY_PATH=/usr/local/run/plumed-2.5.1/lib:$LD_LIBRARY_PATH",
                    "export LD_LIBRARY_PATH=$HOME/miniconda3/lib:$LD_LIBRARY_PATH"]
        env_setup = " && ".join(env_setup_cmd)
        
        if not self.tpr: 
            run = f"{self.exe}/mdrun -ntomp {nt} -deffnm calc"
        else:
            run = f"{self.exe}/mdrun -ntomp {nt} -deffnm {self.tpr}"
            
        full_run = env_setup + " && " + run
        return sp.run(full_run, check=True, shell=True)

