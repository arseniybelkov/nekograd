# nekograd
PyTorch-Lightning wrapped in some shit code
```bash
pip install nekograd
```

## Example

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from nekograd.model import CoreModel
from sklearn.metrics import accuracy_score


architecture: nn.Module = ...
criterion: Callable = ...
metrics: Dict[str, Callable] = {"accuracy": accuracy_score}


model = CoreModel(architecture, criterion, metrics)
device = "gpu" if torch.cuda.is_available() else "cpu"

trainer = pl.Trainer(max_epochs=20, accelerator=device)

trainer.fit(model, datamodule=...)
trainer.test(model, datamodule=...)
```