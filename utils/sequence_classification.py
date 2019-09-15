
from __future__ import absolute_import, division, print_function

from utils.utils import InputExample


class SequenceClsInputExample(InputExample):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid: str, text: str, text_b: str=None, label: str=None):
        """Constructs a SequenceClassInputExample.
        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label/tags of the example.
            This should be specified for train and dev examples, but not for test examples.
        """
        super(SequenceClsInputExample, self).__init__(guid, text, label)
        self.text_b = text_b
