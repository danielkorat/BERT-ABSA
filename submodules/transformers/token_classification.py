
from typing import Union, List

import torch
from torch.nn import CrossEntropyLoss, functional as F

from pytorch_transformers import BertForTokenClassification, XLNetPreTrainedModel, BertModel
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler

from utils.sequential_tagging import TokenClsInputExample
from submodules.transformers.base_model import TransformerBase, logger, InputFeatures
from utils.metrics import tagging
from torch import nn
from sklearn.preprocessing import LabelBinarizer

POS_TAGS = ['NUM', 'X', 'ADP', 'INTJ', 'CCONJ', 'SYM', 'PRON', 'ADJ', 'PROPN', 'ADV', 'VERB', 'PART', 'DET', 'NOUN', 'PUNCT']

DEP_TAGS = ['oprd', 'conj', 'pcomp', 'xcomp', 'nsubj', 'det', 'relcl', 'subtok', 'neg', 'intj', 'appos', 'punct', 'advcl', 'nmod', 'poss', 'aux', 'csubjpass',
            'quantmod', 'pobj', 'prt', 'auxpass', 'case', 'npadvmod', 'cc', 'dative', 'ROOT', 'compound', 'dobj', 'amod', 'parataxis', 'advmod', 'prep', 'nummod', 
            'agent', 'predet', 'acl', 'attr', 'acomp', 'ccomp', 'nsubjpass', 'csubj', 'mark', 'preconj', 'dep', 'meta', 'expl']

class BertForTokenConcatPosDep(BertForTokenClassification):
    """BERT token classification head with linear classifier.

       The forward requires an additional 'valid_ids' map that maps the tensors
       for valid tokens (e.g., ignores additional word piece tokens generated by 
       the tokenizer, as in NER task the 'X' label).
    """

    def __init__(self, config):
        super(BertForTokenConcatPosDep, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.gov_emb_size = 6
        self.gov_embedding = nn.Embedding(84, self.gov_emb_size)
        self.core_feat_size = len(POS_TAGS) + len(DEP_TAGS) + self.gov_emb_size
        self.classifier = nn.Linear(config.hidden_size + self.core_feat_size, config.num_labels)

        # self.core_out_size = 30
        # self.core_linear = nn.Linear(self.core_feat_size, self.core_out_size)
        # self.classifier = nn.Linear(config.hidden_size + self.core_out_size, config.num_labels)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, valid_ids=None, core_feats=None, gov=None):
        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask)
        
        output = outputs[0]

        split = torch.split(core_feats, [len(POS_TAGS), len(DEP_TAGS)], dim=2)
        pos = split[0]
        dep = split[1]
        
        gov_emb = self.gov_embedding(gov)
        # core_out = self.core_linear(torch.cat((pos, dep, gov_emb), dim=2))
        # core_out = self.dropout(core_out)

        # output = torch.cat((output, core_out), dim=2)
        output = torch.cat((output, pos, dep, gov_emb), dim=2)
        output = self.dropout(output)
        output = self.classifier(output)

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            active_positions = valid_ids.view(-1) != 0.0
            active_labels = labels.view(-1)[active_positions]
            active_logits = output.view(-1, self.num_labels)[active_positions]
            loss = loss_fct(active_logits, active_labels)
            return (loss, output, labels)
        return (output,)

       
class BertForTokenLinear(BertForTokenClassification):
    """BERT token classification head with linear classifier.

       The forward requires an additional 'valid_ids' map that maps the tensors
       for valid tokens (e.g., ignores additional word piece tokens generated by 
       the tokenizer, as in NER task the 'X' label).
    """

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, valid_ids=None):
        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            active_positions = valid_ids.view(-1) != 0.0
            active_labels = labels.view(-1)[active_positions]
            active_logits = logits.view(-1, self.num_labels)[active_positions]
            loss = loss_fct(active_logits, active_labels)
            return (loss, logits, labels)
        return (logits,)


class XLNetForTokenClassification(XLNetPreTrainedModel):
    def __init__(self, config):
        super(XLNetForTokenClassification, self).__init__(config)
        raise NotImplementedError

    def forward(self, input_ids, token_type_ids=None, input_mask=None, attention_mask=None,
                mems=None, perm_mask=None, target_mapping=None,
                labels=None, head_mask=None):
        raise NotImplementedError


class TransformerTokenClassifier(TransformerBase):
    MODEL_CLASS = {
        'bert': BertForTokenLinear,
        'bert+core': BertForTokenConcatPosDep,
        'xlnet': XLNetForTokenClassification,
    }

    def __init__(self, model_type, *args, **kwargs):
        assert model_type in self.MODEL_CLASS.keys(), "unsupported model type"
        super(TransformerTokenClassifier, self).__init__(model_type, *args, **kwargs)

        self.model_class = self.MODEL_CLASS[model_type]
        self.model = self.model_class.from_pretrained(self.model_name_or_path, from_tf=bool(
            '.ckpt' in self.model_name_or_path), config=self.config)
        self.to(self.device, self.n_gpus)

    def train(self,
              train_data_set: DataLoader,
              dev_data_set: Union[DataLoader, List[DataLoader]] = None,
              test_data_set: Union[DataLoader, List[DataLoader]] = None,
              gradient_accumulation_steps: int = 1,
              per_gpu_train_batch_size: int = 8,
              max_steps: int = -1,
              num_train_epochs: int = 3,
              max_grad_norm: float = 1.0,
              logging_steps: int = 50,
              save_steps: int = 100):
        return self._train(train_data_set,
                    dev_data_set,
                    test_data_set,
                    gradient_accumulation_steps,
                    per_gpu_train_batch_size,
                    max_steps,
                    num_train_epochs,
                    max_grad_norm,
                    logging_steps=logging_steps,
                    save_steps=save_steps)

    def _batch_mapper(self, batch):
        mapping = {'input_ids': batch[0],
                   'attention_mask': batch[1],
                   # XLM don't use segment_ids
                   'token_type_ids': batch[2],
                   'valid_ids': batch[3]}
        if len(batch) > 4:
            mapping.update({'labels': batch[4]})
        if len(batch) > 5:
            mapping.update({'core_feats': batch[5]})
            mapping.update({'gov': batch[6]})
        return mapping

    def evaluate_predictions(self, logits, label_ids):
        active_positions = label_ids.view(-1) != 0.0
        active_labels = label_ids.view(-1)[active_positions]
        active_logits = logits.view(-1, len(self.labels_id_map) + 1)[active_positions]
        logits = torch.argmax(F.log_softmax(active_logits, dim=1), dim=1)
        logits = logits.detach().cpu().numpy()
        out_label_ids = active_labels.detach().cpu().numpy()
        p, r, f1 = self.extract_labels(out_label_ids, self.labels_id_map, logits)
        logger.info("Evaluation on set = F1: {}".format(f1))
        return p, r, f1

    @staticmethod
    def extract_labels(label_ids, label_map, logits):
        y_true = []
        y_pred = []
        for p, y in zip(logits, label_ids):
            y_pred.append(label_map.get(p, 'O'))
            y_true.append(label_map.get(y, 'O'))
        assert len(y_true) == len(y_pred)
        return tagging(y_pred, y_true)

    def convert_to_tensors(self,
                           examples: List[TokenClsInputExample],
                           max_seq_length: int = 128,
                           include_labels: bool = True) -> TensorDataset:
        features = self._convert_examples_to_features(examples,
                                                      max_seq_length,
                                                      self.tokenizer,
                                                      include_labels,
                                                      # xlnet has a cls token at the end
                                                      cls_token_at_end=bool(
                                                          self.model_type in [
                                                              'xlnet']),
                                                      cls_token=self.tokenizer.cls_token,
                                                      sep_token=self.tokenizer.sep_token,
                                                      cls_token_segment_id=2 if self.model_type in
                                                                                ['xlnet'] else 1,
                                                      # pad on the left for xlnet
                                                      pad_on_left=bool(
                                                          self.model_type in ['xlnet']),
                                                      pad_token_segment_id=4
                                                      if self.model_type in [
                                                          'xlnet'] else 0)
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in features], dtype=torch.long)
        if include_labels:
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

            if self.model_class == BertForTokenConcatPosDep:
                core_feats = torch.tensor([f.core_feats for f in features], dtype=torch.float)
                gov = torch.tensor([f.gov for f in features], dtype=torch.long)

                dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                        all_valid_ids, all_label_ids, core_feats, gov)            
            else:
                dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                        all_valid_ids, all_label_ids)
        else:
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_valid_ids)
        return dataset

    def _convert_examples_to_features(self,
                                      examples: List[TokenClsInputExample],
                                      max_seq_length,
                                      tokenizer,
                                      include_labels=True,
                                      cls_token_at_end=False, pad_on_left=False,
                                      cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                      sequence_segment_id=0,
                                      cls_token_segment_id=1, pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token
            (0 for BERT, 2 for XLNet)
        """
        if self.model_class == BertForTokenConcatPosDep:
            pos_encoder = LabelBinarizer()
            transfomed_pos = pos_encoder.fit_transform(POS_TAGS)
            pos_mappings = {label: i for i, label in enumerate(pos_encoder.classes_)}

            dep_encoder = LabelBinarizer()
            transfomed_dep = dep_encoder.fit_transform(DEP_TAGS)
            dep_mappings = {label: i for i, label in enumerate(dep_encoder.classes_)}
            core_feat_pad = [0 for i in range(len(POS_TAGS) + len(DEP_TAGS) + 1)]

        if include_labels:
            label_map = {v: k for k, v in self.labels_id_map.items()}
            label_pad = 0

        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Processing example %d of %d" % (ex_index, len(examples)))

            tokens = []
            labels = []
            core_feats = []
            valid_tokens = []
            for i, token in enumerate(example.tokens):
                new_tokens = tokenizer.tokenize(token)
                tokens.extend(new_tokens)
                v_tok = [0] * (len(new_tokens))
                v_tok[0] = 1
                valid_tokens.extend(v_tok)
                if include_labels:
                    v_lbl = [label_pad] * (len(new_tokens))
                    v_lbl[0] = label_map.get(example.label[i])
                    labels.extend(v_lbl)

                if self.model_class == BertForTokenConcatPosDep:
                    pos_id = transfomed_pos[pos_mappings[example.feats[i][0]]].tolist()
                    dep_id = transfomed_dep[dep_mappings[example.feats[i][2]]].tolist()
                    gov = [int(example.feats[i][1]) + 1]
                    core_feat = [pos_id + dep_id + gov]
                    core_feats.extend(core_feat * len(new_tokens))

            # truncate by max_seq_length
            tokens = tokens[:(max_seq_length - 2)]
            if include_labels:
                labels = labels[:(max_seq_length - 2)]
            valid_tokens = valid_tokens[:(max_seq_length - 2)]

            if self.model_class == BertForTokenConcatPosDep:
                core_feats = core_feats[:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = tokens + [sep_token]
            if include_labels:
                labels = labels + [label_pad]
            valid_tokens = valid_tokens + [0]
            segment_ids = [sequence_segment_id] * len(tokens)

            if self.model_class == BertForTokenConcatPosDep:
                core_feats = core_feats + [core_feat_pad]

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
                if include_labels:
                    labels = labels + [label_pad]
                valid_tokens = valid_tokens + [0]
                
                if self.model_class == BertForTokenConcatPosDep:
                    core_feats = core_feats + [core_feat_pad]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids
                if include_labels:
                    labels = [label_pad] + labels
                valid_tokens = [0] + valid_tokens
                if self.model_class == BertForTokenConcatPosDep:
                    core_feats = [core_feat_pad] + core_feats

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                if include_labels:
                    labels = ([label_pad] * padding_length) + labels
                valid_tokens = ([0] * padding_length) + valid_tokens
                if self.model_class == BertForTokenConcatPosDep:
                    core_feats = ([core_feat_pad] * padding_length) + core_feats
                
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
                if include_labels:
                    labels = labels + ([label_pad] * padding_length)
                valid_tokens = valid_tokens + ([0] * padding_length)

                if self.model_class == BertForTokenConcatPosDep:
                    core_feats = core_feats + ([core_feat_pad] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(valid_tokens) == max_seq_length
            if self.model_class == BertForTokenConcatPosDep:
                assert len(core_feats) == max_seq_length
            if include_labels:
                assert len(labels) == max_seq_length

            features.append(InputFeatures(input_ids=input_ids,
                                          input_mask=input_mask,
                                          segment_ids=segment_ids,
                                          label_id=labels,
                                          valid_ids=valid_tokens,
                                          core_feats=core_feats))
        return features

    def inference(self, examples: List[TokenClsInputExample], batch_size: int = 64):
        data_set = self.convert_to_tensors(examples, include_labels=False)
        inf_sampler = SequentialSampler(data_set)
        inf_dataloader = DataLoader(data_set, sampler=inf_sampler, batch_size=batch_size)
        logits = self._evaluate(inf_dataloader)
        active_positions = data_set.tensors[-1].view(len(data_set), -1) != 0.0
        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        res_ids = []
        for i in range(logits.size()[0]):
            res_ids.append(logits[i][active_positions[i]].detach().cpu().numpy())
        output = []
        for tag_ids, ex in zip(res_ids, examples):
            tokens = ex.tokens
            tags = [self.labels_id_map.get(t, 'O') for t in tag_ids]
            output.append((tokens, tags))
        return output
