import json
import os 

class inpManager:
    def __init__(self, inp_name: str):
        self.inp_name = inp_name
        self.temp_path = self._path()
        self.temp = self._parse()

    def _path(self) -> str:
        base_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_path, '..', 'templates', f'{self.inp_name}.json')


    def _parse(self) -> dict:
        try:
            with open(self.temp_path, 'r') as f:
                return  json.load(f)
        except Exception as e:
            print(f"Failed to load JSON: {e}")


    def update(self, **kwargs):
        self.temp.update(kwargs)
        if self.inp_name == "ct":
            self.temp['nsites'] = len(kwargs.get('sites'))
            self.temp['zonesize'] = len(kwargs.get('sites'))
            self.temp['wavefunctionreal'] = [0.0 for i in range(self.temp['nsites'])]
            self.temp['wavefunctionreal'][0] = 1.0
            self.temp['sitetypes'] = [1 for i in range(self.temp['nsites'])]
            self.temp['foshift'] = [0.0 for i in range(self.temp['nsites'])]
            self.temp['sitescc'] = [0 for i in range(self.temp['nsites'])]

    def showme(self):
        for k, v in self.temp.items():
            if isinstance (v, list):
                print(f"{k} = {' '.join(map(str, v))}")
            else:
                print(f"{k} = {v}")
            
    def save(self, savein:str):
        savein = os.path.abspath(savein)
        with open(savein, 'w') as f:
            for k, v in self.temp.items():
                if isinstance(v, list):
                    f.write(f"{k} = {' '.join(map(str, v))}\n")
                else:
                    f.write(f"{k} = {v}\n")