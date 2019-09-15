
from os import path
from pathlib import Path

from submodules import LIBRARY_OUT

ABSA_ROOT = Path(path.realpath(__file__)).parent

TRAIN_LEXICONS = ABSA_ROOT / 'train' / 'lexicons'

TRAIN_CONF = ABSA_ROOT / 'train' / 'config.ini'

TRAIN_OUT = LIBRARY_OUT / 'absa' / 'train'

INFERENCE_LEXICONS = ABSA_ROOT / 'inference' / 'lexicons'

INFERENCE_OUT = LIBRARY_OUT / 'absa' / 'inference'

GENERIC_OP_LEX = ABSA_ROOT / 'train' / 'lexicons' / 'GenericOpinionLex.csv'
