# -*- coding: utf-8 -*-
"""Full CWM-Net (CCRM + WAM)."""
from __future__ import annotations

from models.cwmnet import CWMNet, load_ccrm_into_cwmnet, load_partial_into_cwmnet

__all__ = ["CWMNet", "load_ccrm_into_cwmnet", "load_partial_into_cwmnet"]
