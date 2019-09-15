from procedures.transformers.seq_tag import do_training
from pathlib import Path
import pickle
import torch
from bert_absa.io import sentiment_doc_to_bio
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from submodules.absapp.inference.inference import SentimentInference as ABSAppClassification
from submodules.absapp.train.train import TrainSentiment as ABSAppTraining
from bert_absa import cross_domain_settings, in_domain_settings, all_settings, num_splits
import random

import os
module_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(module_path)
in_base = root_path + '/data/conll/'
out_base = str(Path.home()) + '/absa/'


def run_bert_experiments():
    args = pickle.load(open(module_path + '/train_args', 'rb'))
    # default values in "train_args":
    # adam_epsilon=1e-08, cache_dir='', config_name='', data_dir='', do_lower_case=False,
    # eval_all_checkpoints=False, evaluate_during_training=False, gradient_accumulation_steps=1,
    # learning_rate=5e-05, logging_steps=50, max_grad_norm=1.0, max_seq_length=128, max_steps=-1,
    # model_name_or_path='bert-base-uncased', model_type='bert',
    # no_cuda=False, num_train_epochs=3, output_dir='', overwrite_cache=False,
    # overwrite_output_dir=True, per_gpu_eval_batch_size=8, per_gpu_train_batch_size=8,
    # save_steps=500, seed=42, tokenizer_name='', warmup_steps=0, weight_decay=0.0
    # args.no_cuda = True
    args.num_train_epochs = 5
    args.max_seq_length = 84
    args.model_name_or_path = 'bert-base-uncased'
    args.logging_steps = 4000
    args.save_steps = 0
    args.no_cuda=True

    detailed_out = out_base + 'res_detailed.txt'
    open(detailed_out, 'w').write('')
    set_train_res = defaultdict(list)

    for seed_i, seed in enumerate(random.sample(range(100), 3)):
        args.seed = seed
        for setting in in_domain_settings:

            variants = ['', 'with_pos_dep', 'with_absapp_th_2']
            for variant in variants:
                splits = [''] if setting in in_domain_settings else\
                    ['_' + str(i + 1) for i in range(3)]
                for split in splits:
                    set_train = setting + '/' + variant
                    set_split_train = setting + split + '/' + variant
                    set_train_split_seed = set_split_train + '_seed' + str(seed_i)

                    args.model_type = 'bert+core' if variant == 'with_pos_dep' else 'bert'
                    score, best_epoch = train_eval(set_split_train, args, set_train_split_seed)
                    set_train_res[set_train].append(score)
                    open(detailed_out, 'a').write(set_train_split_seed + ': ' + str(score) +
                                                  '(epoch ' + str(best_epoch + 1) + ')\n')

    open(detailed_out, 'a').write('\n' + str(args))
    final_res = ''
    for set_train, scores in set_train_res.items():
        p, r, f1 = np.array(scores).mean(axis=0)
        final_res += '{}: P: {}, R: {}, F1: {}\n'.format(set_train, p, r, f1)
    print('\n\n=================FINAL RES====================\n' + final_res)
    open(out_base + 'res.txt', 'w').write(final_res)

    
def train_eval(exp_path, args, exp_full_name):
    args.data_dir = in_base + exp_path
    args.output_dir = out_base + exp_path
    print('\n'.join(str(k) + ': ' + str(v) for k, v in args.__dict__.items()))
    epochs_eval_dev_test = do_training(args)
    epochs_eval_test = [evals['test'] for evals in epochs_eval_dev_test]
    best_epoch = np.argmax([f1 for p, r, f1 in epochs_eval_test])
    p, r, f1 = epochs_eval_test[best_epoch]
    print("\n\n===================================EVALUATION==================================")
    print(exp_full_name)
    print("P: {}, R: {}, F1: {}".format(p, r, f1))
    print("\n\n===============================================================================")
    torch.cuda.empty_cache()
    return [p, r, f1], best_epoch


def run_absapp_training():
    base_parsed = str(Path.home()) + '/cache/absa/train/parsed/'
    for setting in cross_domain_settings:
        complement_setting = cross_domain_settings[1 - cross_domain_settings.index(setting)]
        for split_i in range(1, num_splits + 1):
            for asp_thresh in (2, 3):
                split_name = setting + '_' + str(split_i)
                complement_split = complement_setting + '_' + str(split_i)
                exp_dir = Path(in_base + split_name + '/with_absapp_th_' + str(asp_thresh))
                ABSAppTraining(asp_thresh=asp_thresh, parse=False)\
                    .run(out_dir=exp_dir, data=None,
                         parsed_data=base_parsed + complement_split + '/raw_train')


def run_absapp_classification():
    for setting in cross_domain_settings:
        for split_i in range(1, num_splits + 1):
            for asp_thresh in (0,):
                split_dir = in_base + setting + '_' + str(split_i) + '/'
                exper_dir = split_dir + 'with_absapp_th_' + str(asp_thresh) + '/'
                inference = ABSAppClassification(Path(exper_dir + 'generated_aspect_lex.csv'),
                                                 exper_dir + 'generated_opinion_lex_reranked.csv')
                with open(exper_dir + 'noisy_tagged.txt', 'w', encoding='utf-8') as out_conll_f:
                    with open(split_dir + 'raw_target.txt', encoding='utf-8') as test_f:
                        for line in tqdm(test_f):
                            sentiment_doc = inference.run(line.strip())
                            if sentiment_doc.sentences:
                                conll_sentences = sentiment_doc_to_bio(sentiment_doc)
                                out_conll_f.write(conll_sentences)


if __name__ == '__main__':
    run_bert_experiments()

