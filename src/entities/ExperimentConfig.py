from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    legacy_features: bool = False
    use_specialist_4_7: bool = False
    use_specialist_16_17: bool = False
    use_velocity: bool = False # General model velocity
    specialist_only_velocity: bool = False
    train_specialist_only: Optional[int] = None
    unroll_lstm: bool = False
    balance_specialist_data: bool = False
    evaluate_mode: bool = False
    load_models_from: Optional[str] = None
    general_only_velocity: bool = False
    general_only_expansion: bool = False
    lstm_units: int = 512
    batch_size: int = 1024
    
    @property
    def extraction_use_velocity(self) -> bool:
        """New features (126) + Velocity"""
        return self.use_velocity or self.specialist_only_velocity
