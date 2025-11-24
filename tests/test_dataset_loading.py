import unittest
from unittest.mock import MagicMock, patch
import torch
from src.dataset.configs import WorldDatasetConfig, LeRobotDatasetConfig, KineticsDatasetConfig, OpenImagesDatasetConfig
from src.dataset.world_dataset import WorldDataset
from src.dataset.wrappers import LeRobotDatasetWrapper, KineticsDatasetWrapper, OpenImagesDatasetWrapper
from src.dataset.batch import WorldBatch

class TestWorldDataset(unittest.TestCase):
    def setUp(self):
        self.droid_cfg = LeRobotDatasetConfig(
            repo_id="mock/droid",
            action_keys=("action",),
            sequence_length_distribution={10: 1.0}
        )
        self.kinetics_cfg = KineticsDatasetConfig(root="mock/kinetics")
        self.openimages_cfg = OpenImagesDatasetConfig(root="mock/openimages", split="train")
        
        self.weights = {"droid": 0.5, "kinetics": 0.3, "openimages": 0.2}
        
        self.dataset_cfg = WorldDatasetConfig(
            datasets={
                "droid": self.droid_cfg,
                "kinetics": self.kinetics_cfg,
                "openimages": self.openimages_cfg
            },
            weights=self.weights
        )

    @patch("src.dataset.world_dataset.LeRobotDatasetMetadata")
    @patch("src.dataset.world_dataset._ensure_delta_timestamps")
    @patch("src.dataset.world_dataset.LeRobotDataset")
    @patch("src.dataset.world_dataset.Kinetics")
    @patch("src.dataset.world_dataset.LeRobotDatasetWrapper")
    @patch("src.dataset.world_dataset.KineticsDatasetWrapper")
    @patch("src.dataset.world_dataset.OpenImagesDatasetWrapper")
    def test_dataset_creation_and_sampling(self, 
                                           mock_openimages_wrapper, mock_kinetics_wrapper, mock_lerobot_wrapper,
                                           mock_kinetics, mock_lerobot_dataset,
                                           mock_ensure_timestamps, mock_metadata):
        # Mock datasets
        droid_ds = MagicMock()
        droid_ds.__len__.return_value = 100
        droid_ds.__getitem__.return_value = WorldBatch(
            sequence_frames=torch.randn(10, 3, 64, 64),
            sequence_actions=torch.randn(10, 6),
            independent_frames_mask=torch.tensor(False),
            actions_mask=torch.ones(10, dtype=torch.bool),
            frames_valid_mask=torch.ones(10, dtype=torch.bool),
            dataset_indices=torch.tensor(-1)
        )
        mock_lerobot_wrapper.return_value = droid_ds
        
        kinetics_ds = MagicMock()
        kinetics_ds.__len__.return_value = 50
        kinetics_ds.__getitem__.return_value = WorldBatch(
            sequence_frames=torch.randn(16, 3, 64, 64),
            sequence_actions=torch.zeros(16, 1),
            independent_frames_mask=torch.tensor(False),
            actions_mask=torch.zeros(16, dtype=torch.bool),
            frames_valid_mask=torch.ones(16, dtype=torch.bool),
            dataset_indices=torch.tensor(-1)
        )
        mock_kinetics_wrapper.return_value = kinetics_ds
        
        openimages_ds = MagicMock()
        openimages_ds.__len__.return_value = 200
        openimages_ds.__getitem__.return_value = WorldBatch(
            sequence_frames=torch.randn(16, 3, 64, 64),
            sequence_actions=torch.zeros(16, 1),
            independent_frames_mask=torch.tensor(True),
            actions_mask=torch.zeros(16, dtype=torch.bool),
            frames_valid_mask=torch.ones(16, dtype=torch.bool),
            dataset_indices=torch.tensor(-1)
        )
        mock_openimages_wrapper.return_value = openimages_ds

        # Instantiate WorldDataset with config
        wm_dataset = WorldDataset(self.dataset_cfg)
        
        # Check total length
        # Max required: 
        # droid: 100 / 0.5 = 200
        # kinetics: 50 / 0.3 = 166.6
        # openimages: 200 / 0.2 = 1000
        # So total length should be around 1000
        self.assertAlmostEqual(len(wm_dataset), 1000, delta=5)
        
        # Check sampling distribution
        counts = {"droid": 0, "kinetics": 0, "openimages": 0}
        for i in range(len(wm_dataset)):
            batch = wm_dataset[i]
            idx = batch.dataset_indices.item()
            name = wm_dataset.dataset_names[idx]
            counts[name] += 1
            
        total = len(wm_dataset)
        print(f"Counts: {counts}")
        self.assertAlmostEqual(counts["droid"] / total, 0.5, delta=0.05)
        self.assertAlmostEqual(counts["kinetics"] / total, 0.3, delta=0.05)
        self.assertAlmostEqual(counts["openimages"] / total, 0.2, delta=0.05)

if __name__ == "__main__":
    unittest.main()
