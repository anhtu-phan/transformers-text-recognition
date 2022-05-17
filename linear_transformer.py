import torch
from fast_transformers.builders import TransformerEncoderBuilder


bert = TransformerEncoderBuilder.from_kwargs(
    n_layers=12,
    n_heads=12,
    query_dimensions=64,
    value_dimensions=64,
    feed_forward_dimensions=3072,
    attention_type="full", # change this to use another
                           # attention implementation
    activation="gelu"
).get()

y = bert(torch.rand(
    10,    # batch_size
    512,   # sequence length
    64*12  # features
))
print(y)
