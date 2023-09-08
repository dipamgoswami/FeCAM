from models.finetune import Finetune
from models.fetril import FeTrIL
from models.fecam import FeCAM

def get_model(model_name, args):
    name = model_name.lower()
    if name == "finetune":
        return Finetune(args)
    elif name == "fetril":
        return FeTrIL(args)
    elif name == "fecam":
        return FeCAM(args)
    else:
        assert 0
