import os as os
import json
import copy
from string import Template

def loadTemp(name: str):
    path = os.path.join(os.getcwd(), 'templates', f'{name}.json')
    with open (path, 'r') as f:
        temp = json.load(f)
    return  copy.deepcopy(temp)

def checkSettings(settings: dict) -> None:
    for key, val in settings.items():
        if val == None or val == []:
            raise settingError (f'{key} is required')

def writeSettings(settings: dict, path) -> None:
    with open(path, 'w') as f:
        for key, val in settings.items():
            if isinstance(val, list):
                val_str = ' '.join(str(v) for v in val)
            else:
                val_str = str(val)
            f.write(f'{key} = {val_str}\n')

    
def bringCTFile(savein: str, **kwargs) -> None:
    ct = loadTemp('ct')
    nsites = len(kwargs.get('sites'))
    ct['sites'] = kwargs.get('sites')
    ct['nsites'] = nsites 
    ct['zonesize'] = nsites
    for key, val in kwargs.items():
        if key not in ct.keys():
            raise ValueError (f'no inp as {key}!\nPossible options:\n{ct.keys()}')
        ct[key] = val
    
    ct['wavefunctionreal'] = [0.0 for i in range(nsites)] 
    ct['wavefunctionreal'][0] = 1.0
    ct['sitetypes'] = [1 for i in range(nsites)]
    ct['foshift'] = [0.0 for i in range(nsites)]
    ct['sitescc'] = [0 for i in range(nsites)]

    if not checkSettings(ct):
        writeSettings(ct, savein)
    print(f'ct file successfully generated in {savein}')

def bringSpecFile(savein: str, **kwargs) -> None:
    spec = loadTemp('spec')
    for key, val in kwargs.items():
        if key not in spec.keys():
            raise ValueError (f'no inp as {key}!\nPossible options:\n{spec.keys()}')
        spec[key] = val

    if not checkSettings(spec):
        writeSettings(spec, savein)
    print(f'spec file successfully generated in {savein}')

def bringSubFile(savein:str, **kwargs) -> None:

    tempFile = os.path.join(os.getcwd(), 'templates', 'submit_temp.sh') 
    with open(tempFile, 'r') as f:
        tempContent = f.read()
    submitTemp = Template(tempContent)
    submit = submitTemp.substitute(kwargs)

    with open(savein, "w") as f:
        f.write(submit)
    print (f'submit file generated successfully in the {savein}')