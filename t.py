from outlines_core import Index, Vocabulary, Guide
from outlines_core.kernels.torch import allocate_token_bitmask
import torch

v = Vocabulary.from_pretrained("gpt2")

import time

st = time.time()
i = Index(r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", v)
print(time.time() - st)

mask = allocate_token_bitmask(len(v))

a = i.get_allowed_tokens(0)

i = Guide(i)
st = time.time()
i.write_mask_into(mask.data_ptr(), mask.numel(), mask.element_size())
print(time.time() - st)
i.advance(a[0], prefill_mask=True, return_tokens=False)
st = time.time()
i.write_mask_into(mask.data_ptr(), mask.numel(), mask.element_size())
print(time.time() - st)

def index() -> Index:
    eos_token_id = 3
    tokens = {"1": [1], "2": [2]}
    regex = r"[1-9]"

    vocabulary = Vocabulary(eos_token_id, tokens)
    return Index(regex, vocabulary)


def test_basic_interface(index):
    init_state = index.get_initial_state()
    assert index.is_final_state(init_state) is False

    allowed_tokens = index.get_allowed_tokens(init_state)
    print(allowed_tokens)
    assert sorted(allowed_tokens) == [1, 2]

    next_state = index.get_next_state(init_state, allowed_tokens[-1])
    assert index.is_final_state(next_state) is True
    assert index.get_final_states() == {next_state}

    expected_transitions = {
        init_state: {
            1: next_state,
            2: next_state,
        },
        next_state: {
            3: next_state,
        },
    }
    assert index.get_transitions() == expected_transitions

test_basic_interface(index())