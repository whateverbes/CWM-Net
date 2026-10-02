import os


from Config.options import opt


def _latest_numbered(stem: str):
    if not os.path.exists(opt.checkpoints_dir):
        return None
    epochs = []
    prefix = stem + "_"
    for x in os.listdir(opt.checkpoints_dir):
        name, ext = os.path.splitext(x)
        if ext != ".pt" or not name.startswith(prefix):
            continue
        tail = name[len(prefix):]
        if tail.isdigit():
            epochs.append(int(tail))
    if not epochs:
        return None
    return "{}_{}.pt".format(stem, max(epochs))


def getLatestCheckpointName():
    """Resume helper for train_uieb.py (CWMNetBase_<epoch>.pt)."""
    return _latest_numbered("CWMNetBase")


def getLatestCwmnetCheckpointName():
    """Resume helper for train_cwmnet.py (CWMNet_<epoch>.pt)."""
    return _latest_numbered("CWMNet")
