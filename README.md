[![codecov](https://codecov.io/gh/arseniybelkov/nekograd/branch/master/graph/badge.svg?token=J49UF83POB)](https://codecov.io/gh/arseniybelkov/nekograd)
# nekograd
_Fast & Flexible_ ~~(just like a catgirl)~~ deep learning framework.
  
All frameworks require vast manuscripts of code written in order to
create the simplest trainable model configuration.
We propose `nekograd` as a convinient way of creating such pipelines
with the least amount of code needed to be written.
```bash
pip install nekograd
```

## Example

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from nekograd.model import CoreModel
from nekograd.model.policy import Multiply
from sklearn.metrics import accuracy_score


# Simplest use case, which covers many DL tasks.
# You just define architecture, loss function, metrics,
# optimizer and lr_scheduler.

architecture: nn.Module = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
criterion: Callable = nn.CrossEntropyLoss()
metrics: Dict[str, Callable] = {"accuracy": accuracy_score}

optimizer = torch.optim.Adam(architecture.parameters())
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                 Multiply({10: 0.1}))

model = CoreModel(architecture, criterion, metrics,
                  optimizer=optimizer, lr_scheduler=lr_scheduler)

device = "gpu" if torch.cuda.is_available() else "cpu"

trainer = pl.Trainer(max_epochs=20, accelerator=device)

trainer.fit(model, datamodule=...)
trainer.test(model, datamodule=...)
```