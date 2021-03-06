
import argparse
import io
import logging
import os

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils.sequential_tagging import (TokenClsInputExample,
                                      TokenClsProcessor)
from utils.utils import write_column_tagged_file
from submodules.transformers.token_classification import TransformerTokenClassifier
from procedures.procedure import Procedure
from procedures.transformers.base import (create_base_args,
                                                        inference_args,
                                                        set_seed,
                                                        setup_backend,
                                                        train_args)
from utils.io import prepare_output_path
from utils.text import SpacyInstance

logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class TransformerTokenClsTrain(Procedure):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument("--data_dir", default=None, type=str, required=True,
                            help="The input data dir. Should contain dataset files to be parsed "
                                 + "by the dataloaders.")
        train_args(parser, models_family=TransformerTokenClassifier.MODEL_CLASS.keys())
        create_base_args(parser, model_types=TransformerTokenClassifier.MODEL_CLASS.keys())

    @staticmethod
    def run_procedure(args):
        do_training(args)


class TransformerTokenClsRun(Procedure):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument("--data_file", default=None, type=str, required=True,
                            help="The data file containing data for inference")
        inference_args(parser)
        create_base_args(parser, model_types=TransformerTokenClassifier.MODEL_CLASS.keys())

    @staticmethod
    def run_procedure(args):
        do_inference(args)


def do_training(args):
    prepare_output_path(args.output_dir, args.overwrite_output_dir)
    device, n_gpus = setup_backend(args.no_cuda)
    # Set seed
    set_seed(args.seed, n_gpus)
    # prepare data
    processor = TokenClsProcessor(args.data_dir)

    classifier = TransformerTokenClassifier(model_type=args.model_type,
                                            model_name_or_path=args.model_name_or_path,
                                            labels=processor.get_labels(),
                                            config_name=args.config_name,
                                            tokenizer_name=args.tokenizer_name,
                                            do_lower_case=args.do_lower_case,
                                            output_path=args.output_dir,
                                            device=device,
                                            n_gpus=n_gpus)

    train_ex = processor.get_train_examples()
    if train_ex is None:
        raise Exception("No train examples found, quitting.")
    dev_ex = processor.get_dev_examples()
    test_ex = processor.get_test_examples()

    train_batch_size = args.per_gpu_train_batch_size * max(1, n_gpus)

    train_dataset = classifier.convert_to_tensors(train_ex,
                                                  max_seq_length=args.max_seq_length)
    train_sampler = RandomSampler(train_dataset)
    train_dl = DataLoader(train_dataset, sampler=train_sampler,
                          batch_size=train_batch_size)
    dev_dl = None
    test_dl = None
    if dev_ex is not None:
        dev_dataset = classifier.convert_to_tensors(dev_ex,
                                                    max_seq_length=args.max_seq_length)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dl = DataLoader(dev_dataset, sampler=dev_sampler,
                            batch_size=args.per_gpu_eval_batch_size)

    if test_ex is not None:
        test_dataset = classifier.convert_to_tensors(test_ex,
                                                     max_seq_length=args.max_seq_length)
        test_sampler = SequentialSampler(test_dataset)
        test_dl = DataLoader(test_dataset, sampler=test_sampler,
                             batch_size=args.per_gpu_eval_batch_size)

    total_steps, _ = classifier.get_train_steps_epochs(args.max_steps,
                                                       args.num_train_epochs,
                                                       args.per_gpu_train_batch_size,
                                                       len(train_dataset))

    classifier.setup_default_optimizer(weight_decay=args.weight_decay,
                                       learning_rate=args.learning_rate,
                                       adam_epsilon=args.adam_epsilon,
                                       warmup_steps=args.warmup_steps,
                                       total_steps=total_steps)
    return classifier.train(train_dl, dev_dl, test_dl, 
                     gradient_accumulation_steps=args.gradient_accumulation_steps,
                     per_gpu_train_batch_size=args.per_gpu_train_batch_size,
                     max_steps=args.max_steps,
                     num_train_epochs=args.num_train_epochs,
                     max_grad_norm=args.max_grad_norm,
                     logging_steps=args.logging_steps,
                     save_steps=args.save_steps)
    # classifier.save_model(args.output_dir, args=args)


def do_inference(args):
    prepare_output_path(args.output_dir, args.overwrite_output_dir)
    device, n_gpus = setup_backend(args.no_cuda)
    args.batch_size = args.per_gpu_eval_batch_size * max(1, n_gpus)
    inference_examples = process_inference_input(args.data_file)
    classifier = TransformerTokenClassifier.load_model(model_path=args.model_path,
                                                       model_type=args.model_type)
    classifier.to(device, n_gpus)
    output = classifier.inference(inference_examples, args.batch_size)
    write_column_tagged_file(args.output_dir + os.sep + "output.txt", output)


def process_inference_input(input_file):
    with io.open(input_file) as fp:
        texts = [l.strip() for l in fp.readlines()]
    tokenizer = SpacyInstance(disable=["tagger", "parser", "ner"])
    examples = []
    for i, t in enumerate(texts):
        examples.append(TokenClsInputExample(str(i), t, tokenizer.tokenize(t)))
    return examples


if __name__ == '__main__':
    # TransformerTokenClsTrain.run()
    TransformerTokenClsRun.run()
