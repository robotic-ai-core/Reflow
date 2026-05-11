"""
Pipeline component integration tests.

Exercises model + datamodule wiring (forward pass, loss, checkpoint round-trip,
data determinism) without mocking Trainer.fit. True end-to-end fit/pause/resume
flows live in tests/integration/test_resume_e2e.py.
"""

import torch

from lightning_reflow.data import SimpleDataModule
from lightning_reflow.models import SimpleReflowModel


class TestPipelineComponents:

    def test_model_forward_and_loss_on_real_batch(self):
        model = SimpleReflowModel(
            input_dim=20, hidden_dim=32, output_dim=3,
            learning_rate=0.01, loss_type='cross_entropy',
        )
        data_module = SimpleDataModule(
            batch_size=8, train_samples=24, val_samples=8,
            input_dim=20, output_dim=3,
            task_type='classification', seed=42,
        )
        data_module.setup()

        train_batch = next(iter(data_module.train_dataloader()))
        assert train_batch['input'].shape == (8, 20)
        assert train_batch['target'].shape == (8,)

        output = model(train_batch['input'])
        assert output.shape == (8, 3)

        loss = model.training_step(train_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    def test_checkpoint_round_trip_preserves_model_parameters(self, temp_dir):
        model = SimpleReflowModel(input_dim=20, hidden_dim=32, output_dim=3)

        checkpoint_path = temp_dir / "checkpoint.ckpt"
        torch.save({
            'state_dict': model.state_dict(),
            'hyper_parameters': dict(model.hparams),
            'epoch': 5,
            'global_step': 100,
            'wandb_run_id': 'test-run-123',
        }, checkpoint_path)

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert checkpoint['epoch'] == 5
        assert checkpoint['global_step'] == 100
        assert checkpoint['wandb_run_id'] == 'test-run-123'

        new_model = SimpleReflowModel(**checkpoint['hyper_parameters'])
        new_model.load_state_dict(checkpoint['state_dict'])

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            torch.testing.assert_close(p1, p2)

    def test_data_pipeline_is_deterministic_for_fixed_seed(self):
        configs = [
            {'batch_size': 4, 'seed': 42, 'num_workers': 0},
            {'batch_size': 8, 'seed': 42, 'num_workers': 0},
            {'batch_size': 4, 'seed': 42, 'num_workers': 0},
        ]
        datasets = []
        for cfg in configs:
            dm = SimpleDataModule(
                train_samples=16, input_dim=10, output_dim=2, **cfg
            )
            dm.setup()
            datasets.append(dm)

        ds1 = datasets[0].train_dataset.dataset if hasattr(datasets[0].train_dataset, 'dataset') else datasets[0].train_dataset
        ds3 = datasets[2].train_dataset.dataset if hasattr(datasets[2].train_dataset, 'dataset') else datasets[2].train_dataset

        sample1 = ds1[0]
        sample3 = ds3[0]
        torch.testing.assert_close(sample1['input'], sample3['input'])
        torch.testing.assert_close(sample1['target'], sample3['target'])
