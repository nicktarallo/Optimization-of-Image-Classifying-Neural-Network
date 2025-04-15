import torch.nn.utils.prune as prune
import torchvision.models as models
import torch.nn as nn

# Utilize unstructured pruning (prune 20% of weights)
def prune_model(model, amount=0.2):
    # Prune all Conv2d and Linear layers - prune weights that contribute least to end result
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')

