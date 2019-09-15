
import logging
from abc import ABC
from os import path
from pathlib import Path

logger = logging.getLogger(__name__)

LIBRARY_PATH = Path(path.realpath(__file__)).parent
LIBRARY_ROOT = LIBRARY_PATH.parent
LIBRARY_OUT = Path(Path.home()) / 'nlp-architect' / 'cache'
LIBRARY_DATASETS = LIBRARY_ROOT / 'datasets'


class TrainableModel(ABC):
    def convert_to_tensors(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def inference(self, *args, **kwargs):
        pass

    def save_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass
