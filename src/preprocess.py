from tokenizers import decoders, processors, Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFD
from tokenizers.pre_tokenizers import Sequence, Whitespace, ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from urllib.parse import unquote


def remove_separators(payload: str) -> str:
    """Remove line separators from the payload"""
    payload = payload.replace("\\r\\n", "")
    return payload


def remove_url_encoding(payload: str) -> str:
    """Decode URL encoded characters from the payload"""

    # You need this loop to consider repetitive(mostly double) encoding
    while True:
        unquoted_payload = unquote(payload)
        if unquoted_payload == payload:
            break
        payload = unquoted_payload

    return payload


class WAFTokenizer():

    def __init__(self, vocab_size, min_frequency=5):
        # Initialize tokenizer
        # self.tokenizer = Tokenizer(BPE(unk_token="<UNK>", pad_token="<PAD>"))
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = Sequence([Whitespace(), ByteLevel()])
        self.tokenizer.normalizer = NFD()
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<PAD>")

        # Initialize BPE trainer
        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<UNK>", "<PAD>"]
        )

    def train(self, iterator):
        self.tokenizer.train_from_iterator(iterator, trainer=self.trainer)

    def get_fast_tokenizer(self):
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            unk_token="<UNK>",
            pad_token="<PAD>"
        )
        return fast_tokenizer

