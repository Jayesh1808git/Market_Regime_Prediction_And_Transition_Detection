from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.models.trainer import load_model_bundle
from src.models.transition_detector import load_transition_detector
from src.data.labeler import load_hmm_artifacts
from src.utils.config import cfg


@dataclass
class ModelStore:
    main_bundle: Optional[dict] = None
    transition_detector: Optional[object] = None
    transition_feat_cols: Optional[list] = None
    hmm_bundle: Optional[dict] = None

    def load(self) -> None:
        self.main_bundle = load_model_bundle(cfg.deployment.model_bundle_path)
        detector, feat_cols = load_transition_detector("models/transition_detector.pkl")
        self.transition_detector = detector
        self.transition_feat_cols = feat_cols
        self.hmm_bundle = load_hmm_artifacts(cfg.hmm.model_path)

    @property
    def is_main_loaded(self) -> bool:
        return self.main_bundle is not None

    @property
    def is_transition_loaded(self) -> bool:
        return self.transition_detector is not None and self.transition_feat_cols is not None

    @property
    def is_hmm_loaded(self) -> bool:
        return self.hmm_bundle is not None

    def latest_metrics_table(self) -> pd.DataFrame:
        path = Path(cfg.training.model_comparison_path)
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        if "model" in df.columns:
            return df.sort_values("f1_macro", ascending=False, ignore_index=True)
        return df
