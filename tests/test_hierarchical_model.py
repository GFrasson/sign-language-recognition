import sys
import types
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from entities.Settings import GeometricFeaturesSettings


class _ModelStub:
    pass


class _SpecialistModelStub:
    pass


model_module = types.ModuleType("entities.Model")
model_module.Model = _ModelStub
sys.modules["entities.Model"] = model_module

specialist_model_module = types.ModuleType("entities.SpecialistModel")
specialist_model_module.SpecialistModel = _SpecialistModelStub
sys.modules["entities.SpecialistModel"] = specialist_model_module

from entities.HierarchicalModel import HierarchicalModel


class HierarchicalModelFeatureSelectionTests(unittest.TestCase):
    def setUp(self):
        GeometricFeaturesSettings.configure(use_legacy=False, use_velocity=True)

    def test_general_model_uses_expansion_and_velocity_when_enabled(self):
        features = np.arange(2 * 3 * GeometricFeaturesSettings.N_FEATURES).reshape(
            2, 3, GeometricFeaturesSettings.N_FEATURES
        )
        model = HierarchicalModel(
            merge_class_map={7: 4},
            specialist_configs={4: [4, 7]},
            general_use_velocity=True,
            general_only_expansion=True,
        )

        selected_features = model._get_general_features(features)
        expansion_start = (
            2 * GeometricFeaturesSettings.NUM_ANGLES_PER_HAND
            + GeometricFeaturesSettings.NUM_POSE_DISTANCES
        )

        np.testing.assert_array_equal(selected_features, features[:, :, expansion_start:])
        self.assertEqual(selected_features.shape[2], 114)

    def test_general_model_keeps_all_features_with_velocity_without_expansion(self):
        features = np.arange(2 * 3 * GeometricFeaturesSettings.N_FEATURES).reshape(
            2, 3, GeometricFeaturesSettings.N_FEATURES
        )
        model = HierarchicalModel(
            merge_class_map={7: 4},
            specialist_configs={4: [4, 7]},
            general_use_velocity=True,
        )

        selected_features = model._get_general_features(features)

        np.testing.assert_array_equal(selected_features, features)
        self.assertEqual(selected_features.shape[2], 192)


if __name__ == "__main__":
    unittest.main()
