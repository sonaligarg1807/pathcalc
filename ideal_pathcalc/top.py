import pandas as pd
import itertools

class top:
    def __init__ (self, topFilePath: str):
        self.topFilePath = topFilePath
    
    @property
    def loadFile(self):
        with open(self.topFilePath, 'r') as f:
            return f.readlines()

    @property
    def avilableParamsList(self) -> list:
        sections = []
        for line in self.loadFile:
            if line[0] == '[':
                sections.append(line[1:-1].strip())
        return sections
    
    def get(self, sec: str):
        up_cut = itertools.dropwhile(lambda line: '[ '+sec not in line, self.loadFile)
        next(up_cut, None)
        section = itertools.takewhile(lambda line: not line.startswith('[ '), up_cut)
    
        data = []
        labels = []
        for i in section:
            if i.startswith(';'):
                raw = i[1:].strip()
                # Fix: Override labels in a safe way if data columns don't match label count
                tentative_labels = raw.split()
            else:
                row = list(itertools.takewhile(lambda x: ';' not in x, i.strip().split()))
                data.append(row)
    
        # Now infer labels if they were given and safe to use
        if labels:
            data_df = pd.DataFrame(data, columns=labels)
        elif 'tentative_labels' in locals():
            # Only use labels if their length matches the number of columns
            if data and len(tentative_labels) == len(data[0]):
                labels = tentative_labels
                data_df = pd.DataFrame(data, columns=labels)
            else:
                data_df = pd.DataFrame(data)
        else:
            data_df = pd.DataFrame(data)
    
        return data_df

    def update_molecule_count(self, mol_name: str, new_count: int):
        lines = self.loadFile  # keeps all lines as-is
        in_section = False
        updated_lines = []
    
        for line in lines:
            stripped = line.strip()
    
            if stripped.startswith('[ molecules ]'):
                in_section = True
                updated_lines.append(line)
                continue
    
            if in_section:
                if stripped.startswith('['):  # exiting [ molecules ]
                    in_section = False
                    updated_lines.append(line)
                    continue
    
                if stripped.startswith(';') or not stripped:
                    updated_lines.append(line)
                    continue
    
                parts = stripped.split()
                if parts and parts[0] == mol_name:
                    # Replace count, but preserve original whitespace prefix
                    prefix = line[:line.find(parts[0])]
                    updated_line = f"{prefix}{parts[0]:<20}{new_count}\n"
                    updated_lines.append(updated_line)
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
    
        with open(self.topFilePath, 'w') as f:
            f.writelines(updated_lines)
