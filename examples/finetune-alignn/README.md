## Example: fine-tuning a pretrained alignn model

This example shows how to use a pretrained model from [usnistgov/alignn](https://github.com/usnistgov/alignn) with nfflr.
After loading the model from `alignn.pretrained`, define a custom `ALIGNNTransform` and batch collation functions to preprocess materials to the format used by usnistgov/alignn.

from this directory, run:

```sh
uv sync
uv run nff train config.py
```
