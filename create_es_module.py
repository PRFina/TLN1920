"""QUick script to use when start a new exercise module. Just call script with
python create_es_module <numero esercitazione>
"""

from pathlib import Path
import sys

MODULE_NAME =  'esercitazione' + sys.argv[1]
DATA_NAME = 'data'
OUTPUT_NAME = 'output'
CODE_NAME = 'src'

top_level = [Path(DATA_NAME), 
             Path(OUTPUT_NAME),
             Path(CODE_NAME)]

module_path = Path() / Path(MODULE_NAME)
module_path.mkdir()

for dir in top_level:
    (module_path / dir).mkdir()

code_path = top_level[2]
(module_path / code_path / '__init__.py').open(mode='w') # init file for src package

(module_path / MODULE_NAME).with_suffix('.ipynb').open(mode='w') # notebook file


