import time
from outlines_core import Index, Vocabulary

regex = r"A: [\w \.\*\-=\+,\?/]{10,50}\. The answer is [1-9][0-9]{0,9}\."
vocab = Vocabulary.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")

t0 = time.perf_counter()
idx = Index(regex, vocab)         # just build once
dt = time.perf_counter() - t0
print(f"Index build: {dt:.3f}s")
