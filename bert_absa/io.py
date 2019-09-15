import json
from random import shuffle
import os
from submodules.absapp.inference.data_types import SentimentDoc
from pathlib import Path
from submodules.absapp.inference.data_types import TermType
from shutil import copyfile
import subprocess
from bert_absa import cross_domain_settings, in_domain_settings, all_settings, num_splits, base


def dai2019_single_to_conll_and_raw(sent_file, tok_file, conll_out: str, raw_out: str, opinion_labels=False):
    """Converts ABSA datasets from Dai (2019) format to CoNLL format.
    Args:
        sentence: Path to textfile sentence desciptors, one json per line.
        token_spans: Path to textfile containing token char ranges
        conll_out: Path for output file.
    """
    sentences = []
    token_spans = []
    aspect_spans = []
    opinions = []

    if isinstance(sent_file, str) and isinstance(tok_file, str):
        with open(sent_file, encoding='utf-8') as sentence_f:
            sent_file = [line for line in sentence_f]
        with open(tok_file, encoding='utf-8') as tok_f:
            tok_file = [line for i, line in enumerate(tok_f) if i % 2 == 1]

    assert len(sent_file) == len(tok_file)

    for json_line in sent_file:
        sent_json = json.loads(json_line)
        sentences.append(sent_json['text'])
        curr_aspects = [term['span'] for term in sent_json['terms']] if 'terms' in sent_json else []
        curr_opinions = sent_json.get('opinions', [])
        opinions.append(curr_opinions)
        aspect_spans.append(curr_aspects)

    for i, line in enumerate(tok_file):
        curr_toks = []
        indices = line.split()
        for j in range(0, len(indices), 2):
            curr_toks.append([int(indices[j]), int(indices[j + 1])])
        token_spans.append(curr_toks)

    assert len(sentences) == len(token_spans)
    assert len(token_spans) == len(aspect_spans)
    assert len(token_spans) == len(opinions)

    with open(raw_out, 'w', encoding='utf-8') as raw_f:
        raw_f.write('\n'.join(sentences))

    with open(conll_out, 'w', encoding='utf-8') as conll_f:
        for sentence, tok_indices, asp_indices, op_words in zip(sentences, token_spans, aspect_spans, opinions):
            tokens = [sentence[s: e] for s, e in tok_indices]
            tags = ['O' for i in range(len(tokens))]

            if opinion_labels and op_words:
                for i, token in enumerate(tokens):
                    if token in set(op_words):
                        tags[i] = 'B-OP'

            if asp_indices:
                curr_asp = 0
                inside_aspect = False
                for i, (tok_start, tok_end) in enumerate(tok_indices):
                    if curr_asp == len(asp_indices):
                        break
                    if inside_aspect:
                        tags[i] = 'I-ASP'
                    elif tok_start == asp_indices[curr_asp][0]:
                        inside_aspect = True
                        tags[i] = 'B-ASP'
                    if tok_end == asp_indices[curr_asp][1]:
                        curr_asp += 1
                        inside_aspect = False
        
            conll_f.write('\n'.join(['\t'.join((_)) for _ in zip(tokens, tags)]) + '\n\n')


def preprocess_dai2019(opinion_labels=False):
    in_base = 'nlp_architect/models/absa_neural/data/Dai2019/semeval'
    out_base = 'nlp_architect/models/absa_neural/data/conll'
    out_base += '_op/' if opinion_labels else '/'
    sets = {'laptops': ('14',), 'restaurants': ('14', '15')}
    all_out_dirs = []

    for domain, years in sets.items():
        all_domain_sents = []
        for year in years:
            out_dir = out_base + domain + year + '/'
            all_out_dirs.append(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            for ds in 'train', 'test':
                ds_path = in_base +  year + '/' + domain + '/' + domain + '_' + ds
                sent_file = ds_path + '_sents.json'
                tok_file = ds_path + '_texts_tok_pos.txt'
                dai2019_single_to_conll_and_raw(sent_file, tok_file, out_dir + ds + '.txt', 
                    out_dir + 'raw_' + ds + '.txt', opinion_labels)

                with open(sent_file, encoding='utf-8') as sentence_f:
                    with open(tok_file, encoding='utf-8') as tok_f:
                        for json_line, tok_line in zip(sentence_f, (line for i, line in enumerate(tok_f) if i % 2 == 1)):
                            all_domain_sents.append((json_line, tok_line))
        
        for split_i in range(num_splits):
            split_num = str(split_i + 1)
            lt_to_res_dir = out_base + 'laptops_to_restaurants_' + split_num + '/'
            res_to_lt_dir = out_base + 'restaurants_to_laptops_' + split_num + '/'
            all_out_dirs.extend([lt_to_res_dir, res_to_lt_dir])
            os.makedirs(lt_to_res_dir, exist_ok=True)
            os.makedirs(res_to_lt_dir, exist_ok=True)

            if domain == 'laptops':
                out_test_dir = res_to_lt_dir
                out_train_dir = lt_to_res_dir
            else:
                out_test_dir = lt_to_res_dir
                out_train_dir = res_to_lt_dir
            
            shuffle(all_domain_sents)
            split = round(0.75 * len(all_domain_sents))
            train = all_domain_sents[:split]
            test = all_domain_sents[split:]
            
            dai2019_single_to_conll_and_raw([p[0] for p in train], [p[1] for p in train], out_train_dir + 'train.txt', 
                out_train_dir + 'raw_train.txt', opinion_labels)
            dai2019_single_to_conll_and_raw([p[0] for p in test], [p[1] for p in test], out_test_dir + 'test.txt', out_test_dir + 'raw_test.txt',
                opinion_labels)           

    labels = ['O', 'B-ASP', 'I-ASP']
    if opinion_labels:
        labels.append('B-OP')
    
    for dir in all_out_dirs:
        with open(dir + 'labels.txt', 'w', encoding='utf-8') as labels_f:
            labels_f.write('\n'.join(labels))


def get_label(aspects: set, start: int, length: int):
    res = 'O'
    end = start + length
    for asp_start, asp_end in aspects:
        if start == asp_start:
            res = 'B-ASP'
        if start > asp_start and end <= asp_end:
            return 'I-ASP'
    return res


def sentiment_doc_to_bio(doc: SentimentDoc) -> str:
    aspects = [(e._start, e._start + e._len) for sentence in doc.sentences 
        for e_pair in sentence._events for e in e_pair if e._type == TermType.ASPECT]

    conll_doc = []
    for token_list in doc.tokens:
        conll_sentence = []
        for start, length, text in token_list:
            label = get_label(aspects, start, length)
            conll_sentence.append('\t'.join((text, label)))
        conll_doc.append(conll_sentence)

    return '\n\n'.join(['\n'.join(sent) for sent in conll_doc]) + '\n\n'


def prepare_noisy_labeling():
    for setting in cross_domain_settings:
        for split_i in range(1, num_splits + 1):
            split_dir = base + setting + '_' + str(split_i) + '/'
            for asp_thresh in (0,):
                exper_dir = split_dir + 'asp_th_' + str(asp_thresh) + '/'
                for filename in 'labels.txt', 'train.txt', 'test.txt':
                    copyfile(split_dir + filename, exper_dir + filename)
                with open(exper_dir + 'train.txt', 'a', encoding='utf-8') as train_f:
                    with open(exper_dir + 'noisy_tagged.txt', encoding='utf-8') as noisy_tagged_f:
                        train_f.write(noisy_tagged_f.read())


def add_dep_pos_tags_to_conll():
    for setting in all_settings:
        for ds in 'test', 'train':
            splits = [''] if setting in in_domain_settings else ['_' + str(i + 1) for i in range(num_splits)]
            conll_to_space_tokenised_raw(setting, ds, splits)
            space_tokenized_raw_to_spacy_conll(setting, ds, splits)
            join_bio_conll_with_spacy_conll(setting, ds, splits)


def conll_to_space_tokenised_raw(setting, ds, splits):
    for split in splits:
        ds_path = base + setting + split + '/' + ds
        with open(ds_path + '_toks.txt', 'w') as toks_f:
            with open(ds_path + '.txt') as conll_f:
                toks = []
                for line in conll_f:
                    line = line.strip() 
                    if not line:
                        toks_f.write(' '.join(toks) + '\n')
                        toks = []
                    else:
                        toks.append(line.split('\t')[0])


def space_tokenized_raw_to_spacy_conll(setting, ds, splits):
    # prerequisites: `pip install spacy_conll`; adjust spacy_conll_path variable below
    spacy_conll_path = str(Path.home()) + \
        '/envs/nlp_architect_env/lib/python3.6/site-packages/spacy_conll/__main__.py'
    for split in splits:
        ds_path = base + setting + split + '/' + ds
        subprocess.run(["python", spacy_conll_path, "-f", ds_path + '_toks.txt',
            "-t", "-o", ds_path + '_conll.txt', "-m", "en_core_web_lg", "-s"])


def join_bio_conll_with_spacy_conll(setting, ds, splits):
    for split in splits:
        split_path = base + setting + split
        os.makedirs(split_path + '/core_tagged', exist_ok=True)
        with open(split_path + '/core_tagged/' + ds + '.txt', 'w') as out_conll:
            with open(split_path + '/' + ds + '.txt') as bio_tagged:
                with open(split_path + '/' + ds + '_conll.txt') as core_tagged:
                    sents_bio = bio_tagged.read().split('\n\n')[:-1]
                    sents_core = core_tagged.read().split('\n\n')[:-1]
                    assert len(sents_bio) == len(sents_core)

                    for bio_sent, core_sent in zip(sents_bio, sents_core):
                        bio_toks = [tok_line.split('\t') for tok_line in bio_sent.split('\n')]
                        core_toks = [tok_line.split('\t') for tok_line in core_sent.split('\n')]

                        if len(bio_toks) == len(core_toks):
                            for bio_tok, core_tok in zip(bio_toks, core_toks):
                                text = bio_tok[0]
                                pos = core_tok[3]
                                gov = core_tok[6]
                                rel = core_tok[7]
                                bio = bio_tok[-1]
                                out_conll.write('\t'.join((text, pos, gov, rel, bio)) + '\n')
                            out_conll.write('\n')

        copyfile(base + setting + '/labels.txt', split_path + '/core_tagged/labels.txt')


def get_pos_and_sep_tagsets():
    pos = set()
    dep = set()
    max_gov = -1
    max_sent_len = -1
    for setting in all_settings:
        for ds in 'test', 'train':
            splits = [''] if setting in in_domain_settings else ['_' + str(i + 1) for i in range(num_splits)]
            for split in splits:
                with open(base + setting + split + '/' + ds + '_conll.txt') as f:
                    sent_len = 0
                    for line in f:
                        line = line.strip()
                        if line:
                            sent_len += 1
                            line_s = line.split('\t')
                            pos.add(line_s[3])
                            dep.add(line_s[7])
                            max_gov = max(max_gov, int(line_s[6]))
                        else:
                            max_sent_len = max(max_sent_len, sent_len)
                            sent_len = 0
            
    print('{} pos tags: {}'.format(len(pos), list(pos)))
    print('{} dep tags: {}'.format(len(dep), list(dep)))
    print('max gov index: {}'.format(max_gov))
    print('max sent. len: {}'.format(max_sent_len))
