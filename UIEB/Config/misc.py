import os



import torch

import numpy as np

from Config.options import opt





def getLatestCheckpointName():

    """Resume helper for train_uieb.py (DICAM_<epoch>.pt)."""

    if not os.path.exists(opt.checkpoints_dir):

        return None

    epochs = []

    for x in os.listdir(opt.checkpoints_dir):

        stem, ext = os.path.splitext(x)

        if ext != ".pt":

            continue

        parts = stem.split("_")

        if len(parts) == 2 and parts[0] == "DICAM":

            try:

                epochs.append(int(parts[1]))

            except ValueError:

                pass

    if not epochs:

        return None

    return "DICAM_{}.pt".format(max(epochs))





def getLatestUccpCheckpointName():

    """Resume helper for train_uccp.py (DICAM_UCCP_<epoch>.pt)."""

    if not os.path.exists(opt.checkpoints_dir):

        return None

    epochs = []

    for x in os.listdir(opt.checkpoints_dir):

        stem, ext = os.path.splitext(x)

        if ext != ".pt":

            continue

        parts = stem.split("_")

        if len(parts) == 3 and parts[0] == "DICAM" and parts[1] == "UCCP":

            try:

                epochs.append(int(parts[2]))

            except ValueError:

                pass

    if not epochs:

        return None

    return "DICAM_UCCP_{}.pt".format(max(epochs))

