
import time
from pathlib import Path, PosixPath
from os import PathLike
from submodules.absapp import TRAIN_OUT
from submodules.absapp.train.acquire_terms import AcquireTerms
from submodules.absapp.train.rerank_terms import RerankTerms
from submodules.absapp.utils import parse_docs, _download_pretrained_rerank_model, \
    _write_aspect_lex, _write_opinion_lex
from utils.io import download_unzip

EMBEDDING_URL = 'http://nlp.stanford.edu/data', 'glove.840B.300d.zip'
EMBEDDING_PATH = TRAIN_OUT / 'word_emb_unzipped' / 'glove.840B.300d.txt'
RERANK_MODEL_DEFAULT_PATH = rerank_model_dir = TRAIN_OUT / 'reranking_model' / 'rerank_model.h5'
GENERATED_OPINION_OUT_DIR = TRAIN_OUT / 'output'
GENERATED_ASPECT_OUT_DIR = TRAIN_OUT / 'output'


class TrainSentiment(object):
    def __init__(self, parse: bool = True, rerank_model: PathLike = None, asp_thresh=3,
        op_thresh=2):
        self.start_time = time.time()
        self.acquire_lexicon = AcquireTerms(asp_thresh)
        if parse:
            from utils.spacy_bist import SpacyBISTParser
            self.parser = SpacyBISTParser()
        else:
            self.parser = None

        if not rerank_model:
            print('using pre-trained reranking model')
            rerank_model = _download_pretrained_rerank_model(RERANK_MODEL_DEFAULT_PATH)

        download_unzip(*EMBEDDING_URL, EMBEDDING_PATH, license_msg="Glove word embeddings.")
        self.rerank = RerankTerms(vector_cache=True, rerank_model=rerank_model,
                                  emb_model_path=EMBEDDING_PATH)

    def run(self, out_dir: PathLike = None, data: PathLike = None, parsed_data: PathLike = None):
        if not parsed_data:
            if not self.parser:
                raise RuntimeError("Parser not initialized (try parse=True at init )")
            data_dir, data_file = Path(data).parent.stem, Path(data).stem
            parsed_dir = TRAIN_OUT / 'parsed' / data_dir / data_file
            self.parse_data(data, parsed_dir)
            parsed_data = parsed_dir

        generated_aspect_lex = self.acquire_lexicon.acquire_lexicons(parsed_data)
        _write_aspect_lex(parsed_data, generated_aspect_lex, out_dir if out_dir else GENERATED_ASPECT_OUT_DIR)

        generated_opinion_lex_reranked = \
            self.rerank.predict(AcquireTerms.acquired_opinion_terms_path,
                                AcquireTerms.generic_opinion_lex_path)
        _write_opinion_lex(parsed_data, generated_opinion_lex_reranked, out_dir if out_dir else GENERATED_OPINION_OUT_DIR)

        return generated_opinion_lex_reranked, generated_aspect_lex

    def parse_data(self, data: PathLike or PosixPath, parsed_dir: PathLike or PosixPath):
        _, data_size = parse_docs(self.parser, data, out_dir=parsed_dir)
        if data_size < 1000:
            raise ValueError('The data contains only {0} sentences. A minimum of 1000 '
                             'sentences is required for training.'.format(data_size))
