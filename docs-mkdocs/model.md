# modeling interface

Models should have a consistent interface, regardless of the backend!

```
NFFLrModel:
    forward(x: Atoms) -> dict[str, torch.Tensor]
```

Depending on the model and the prediction task, both the input representation and output format may vary.

## input representation
For efficient training, models should also be able to operate on preprocessed structures,
e.g. a graph neural network could implement `forward(x: dgl.DGLGraph)` to allow asynchronous
construction of graph batches in a `DataLoader`.

- `dgl.DGLGraph`
- PyG input format
- `Atoms` with ghost atom padding
- some other custom input representation


## output representation
Depending on the task, a model may return predictions in different formats.

- Single-target tasks like scalar or vector regression or classification naturally return a single tensor `forward(x: Atoms) -> torch.Tensor`.
- A force-field should return a `dict[str, torch.Tensor]`: `{"total_energy": torch.Tensor, "forces": torch.Tensor, "stress": torch.Tensor}`, where `total_energy` is a scalar, `forces` is an `(n_atoms, n_spatial_dimensions)` tensor, and `stress` is a `(n_spatial_dimensions, n_spatial_dimensions)` tensor
- A custom multi-target task might also return a `dict` of
