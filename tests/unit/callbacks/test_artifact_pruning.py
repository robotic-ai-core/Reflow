"""Tests for artifact version pruning in UnifiedArtifactManager.

Verifies that keep_n_versions correctly prunes old artifact versions
without accidentally deleting versions that should be kept.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from lightning_reflow.utils.wandb.unified_artifact_manager import UnifiedArtifactManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeArtifactVersion:
    """Mimics wandb.Artifact version returned by api.artifacts()."""
    version: str
    created_at: str
    aliases: List[str] = field(default_factory=list)
    _deleted: bool = field(default=False, init=False)
    _delete_raises: Optional[Exception] = field(default=None, init=False)

    def delete(self):
        if self._delete_raises:
            raise self._delete_raises
        self._deleted = True


def make_versions(n: int, aliased_indices: Optional[List[int]] = None) -> List[FakeArtifactVersion]:
    """Create N fake versions in newest-first order (matching W&B API behavior).

    Args:
        n: Number of versions to create
        aliased_indices: Which positions (0-based, newest=0) get aliases
    """
    aliased_indices = aliased_indices or []
    versions = []
    for i in range(n):
        # Newest first: v(n-1), v(n-2), ..., v0
        version_num = n - 1 - i
        aliases = ["latest"] if i in aliased_indices else []
        versions.append(FakeArtifactVersion(
            version=f"v{version_num}",
            created_at=f"2026-01-01T{i:02d}:00:00Z",
            aliases=aliases,
        ))
    return versions


@pytest.fixture
def manager():
    """Create UnifiedArtifactManager with keep_n_versions=3."""
    return UnifiedArtifactManager(verbose=True, keep_n_versions=3)


# ---------------------------------------------------------------------------
# Tests: Basic pruning
# ---------------------------------------------------------------------------

class TestPruneOldVersions:
    """Tests for _prune_old_versions."""

    def test_prune_keeps_n_newest(self, manager):
        """With 5 versions and keep_n=3, the 2 oldest should be deleted."""
        versions = make_versions(5, aliased_indices=[0])  # newest has :latest

        with patch("wandb.Api") as mock_api_cls:
            mock_api = mock_api_cls.return_value
            mock_api.artifacts.return_value = versions

            manager._prune_old_versions(
                entity="e", project="p", artifact_name="test-ckpt",
                artifact_type="model",
            )

        # versions[3] = v1, versions[4] = v0 should be deleted
        assert not versions[0]._deleted  # v4 (newest, aliased) — kept
        assert not versions[1]._deleted  # v3 — kept
        assert not versions[2]._deleted  # v2 — kept
        assert versions[3]._deleted      # v1 — pruned
        assert versions[4]._deleted      # v0 — pruned

    def test_prune_nothing_when_fewer_than_n(self, manager):
        """With 2 versions and keep_n=3, nothing should be deleted."""
        versions = make_versions(2, aliased_indices=[0])

        with patch("wandb.Api") as mock_api_cls:
            mock_api = mock_api_cls.return_value
            mock_api.artifacts.return_value = versions

            manager._prune_old_versions(
                entity="e", project="p", artifact_name="test-ckpt",
                artifact_type="model",
            )

        assert not versions[0]._deleted
        assert not versions[1]._deleted

    def test_prune_nothing_when_exactly_n(self, manager):
        """With 3 versions and keep_n=3, nothing should be deleted."""
        versions = make_versions(3, aliased_indices=[0])

        with patch("wandb.Api") as mock_api_cls:
            mock_api = mock_api_cls.return_value
            mock_api.artifacts.return_value = versions

            manager._prune_old_versions(
                entity="e", project="p", artifact_name="test-ckpt",
                artifact_type="model",
            )

        for v in versions:
            assert not v._deleted

    def test_prune_with_keep_n_1(self):
        """keep_n=1: Only the newest version survives."""
        mgr = UnifiedArtifactManager(verbose=True, keep_n_versions=1)
        versions = make_versions(4, aliased_indices=[0])

        with patch("wandb.Api") as mock_api_cls:
            mock_api = mock_api_cls.return_value
            mock_api.artifacts.return_value = versions

            mgr._prune_old_versions(
                entity="e", project="p", artifact_name="test-ckpt",
                artifact_type="model",
            )

        assert not versions[0]._deleted  # v3 — kept (newest)
        assert versions[1]._deleted      # v2 — pruned
        assert versions[2]._deleted      # v1 — pruned
        assert versions[3]._deleted      # v0 — pruned


class TestAliasProtection:
    """Tests that aliased versions are never deleted."""

    def test_aliased_version_in_delete_range_is_skipped(self):
        """If an aliased version falls in the delete range, it must be skipped."""
        mgr = UnifiedArtifactManager(verbose=True, keep_n_versions=2)
        versions = make_versions(5, aliased_indices=[0])

        # Manually add alias to v1 (index 3) — simulates user-pinned alias
        versions[3].aliases = ["production"]

        with patch("wandb.Api") as mock_api_cls:
            mock_api = mock_api_cls.return_value
            mock_api.artifacts.return_value = versions

            mgr._prune_old_versions(
                entity="e", project="p", artifact_name="test-ckpt",
                artifact_type="model",
            )

        assert not versions[0]._deleted  # v4 — kept (in keep range)
        assert not versions[1]._deleted  # v3 — kept (in keep range)
        assert versions[2]._deleted      # v2 — pruned
        assert not versions[3]._deleted  # v1 — SKIPPED (aliased as "production")
        assert versions[4]._deleted      # v0 — pruned

    def test_all_versions_aliased(self):
        """If every version has aliases, nothing gets deleted."""
        mgr = UnifiedArtifactManager(verbose=True, keep_n_versions=1)
        versions = make_versions(3, aliased_indices=[0, 1, 2])

        with patch("wandb.Api") as mock_api_cls:
            mock_api = mock_api_cls.return_value
            mock_api.artifacts.return_value = versions

            mgr._prune_old_versions(
                entity="e", project="p", artifact_name="test-ckpt",
                artifact_type="model",
            )

        for v in versions:
            assert not v._deleted

    def test_latest_alias_always_on_newest(self):
        """Standard case: :latest on newest version, only it's protected by keep_n."""
        mgr = UnifiedArtifactManager(verbose=True, keep_n_versions=2)
        versions = make_versions(4, aliased_indices=[0])

        with patch("wandb.Api") as mock_api_cls:
            mock_api = mock_api_cls.return_value
            mock_api.artifacts.return_value = versions

            mgr._prune_old_versions(
                entity="e", project="p", artifact_name="test-ckpt",
                artifact_type="model",
            )

        # Newest 2 kept, oldest 2 pruned (neither has aliases)
        assert not versions[0]._deleted  # v3 (latest alias, in keep range)
        assert not versions[1]._deleted  # v2 (in keep range)
        assert versions[2]._deleted      # v1 — pruned
        assert versions[3]._deleted      # v0 — pruned


class TestErrorResilience:
    """Tests that pruning errors never propagate."""

    def test_api_failure_is_swallowed(self, manager):
        """If wandb.Api() raises, pruning silently fails."""
        with patch("wandb.Api", side_effect=Exception("network error")):
            # Should not raise
            manager._prune_old_versions(
                entity="e", project="p", artifact_name="test-ckpt",
                artifact_type="model",
            )

    def test_individual_delete_failure_continues(self, manager):
        """If one version fails to delete, others still get deleted."""
        versions = make_versions(5, aliased_indices=[0])
        # v1 (index 3) will fail to delete
        versions[3]._delete_raises = Exception("in use by downstream")

        with patch("wandb.Api") as mock_api_cls:
            mock_api = mock_api_cls.return_value
            mock_api.artifacts.return_value = versions

            manager._prune_old_versions(
                entity="e", project="p", artifact_name="test-ckpt",
                artifact_type="model",
            )

        assert not versions[3]._deleted  # Failed to delete
        assert versions[4]._deleted      # v0 still deleted successfully

    def test_empty_collection(self, manager):
        """Empty artifact collection doesn't crash."""
        with patch("wandb.Api") as mock_api_cls:
            mock_api = mock_api_cls.return_value
            mock_api.artifacts.return_value = []

            manager._prune_old_versions(
                entity="e", project="p", artifact_name="test-ckpt",
                artifact_type="model",
            )


class TestUploadArtifactIntegration:
    """Tests that pruning is correctly triggered from upload_artifact."""

    def _make_mock_trainer(self):
        trainer = MagicMock()
        trainer.is_global_zero = True
        trainer.current_epoch = 5
        trainer.global_step = 1000
        trainer.state = "running"
        return trainer

    def _make_mock_wandb_run(self):
        run = MagicMock()
        run.id = "abc123"
        run.entity = "myentity"
        run.project = "myproject"
        logged = MagicMock()
        logged.version = "v3"
        run.log_artifact.return_value = logged
        return run

    def test_keep_n_none_skips_pruning(self):
        """When keep_n_versions=None, no pruning happens."""
        mgr = UnifiedArtifactManager(verbose=True, keep_n_versions=None)
        trainer = self._make_mock_trainer()
        run = self._make_mock_wandb_run()

        with patch.object(mgr, '_validate_files', return_value={"f.ckpt": "/tmp/f.ckpt"}), \
             patch("wandb.Artifact") as mock_artifact_cls, \
             patch.object(mgr, '_prune_old_versions') as mock_prune:
            mgr.upload_artifact(
                trainer=trainer, files={"f.ckpt": "/tmp/f.ckpt"},
                artifact_name="abc123-latest", artifact_type="model",
                wandb_run=run,
            )
            mock_prune.assert_not_called()

    def test_keep_n_0_skips_pruning(self):
        """keep_n_versions=0 is rejected (guard: >= 1)."""
        mgr = UnifiedArtifactManager(verbose=True, keep_n_versions=0)
        trainer = self._make_mock_trainer()
        run = self._make_mock_wandb_run()

        with patch.object(mgr, '_validate_files', return_value={"f.ckpt": "/tmp/f.ckpt"}), \
             patch("wandb.Artifact") as mock_artifact_cls, \
             patch.object(mgr, '_prune_old_versions') as mock_prune:
            mgr.upload_artifact(
                trainer=trainer, files={"f.ckpt": "/tmp/f.ckpt"},
                artifact_name="abc123-latest", artifact_type="model",
                wandb_run=run,
            )
            mock_prune.assert_not_called()

    def test_keep_n_positive_triggers_pruning_for_model(self):
        """keep_n_versions > 0 with artifact_type='model' triggers pruning."""
        mgr = UnifiedArtifactManager(verbose=True, keep_n_versions=3)
        trainer = self._make_mock_trainer()
        run = self._make_mock_wandb_run()

        with patch.object(mgr, '_validate_files', return_value={"f.ckpt": "/tmp/f.ckpt"}), \
             patch("wandb.Artifact") as mock_artifact_cls, \
             patch.object(mgr, '_prune_old_versions') as mock_prune:
            mgr.upload_artifact(
                trainer=trainer, files={"f.ckpt": "/tmp/f.ckpt"},
                artifact_name="abc123-latest", artifact_type="model",
                wandb_run=run,
            )
            mock_prune.assert_called_once_with(
                entity="myentity", project="myproject",
                artifact_name="abc123-latest", artifact_type="model",
            )

    def test_pruning_skipped_for_config_type(self):
        """Pruning only applies to 'model' type, not 'config'."""
        mgr = UnifiedArtifactManager(verbose=True, keep_n_versions=3)
        trainer = self._make_mock_trainer()
        run = self._make_mock_wandb_run()

        with patch.object(mgr, '_validate_files', return_value={"c.yaml": "/tmp/c.yaml"}), \
             patch("wandb.Artifact") as mock_artifact_cls, \
             patch.object(mgr, '_prune_old_versions') as mock_prune:
            mgr.upload_artifact(
                trainer=trainer, files={"c.yaml": "/tmp/c.yaml"},
                artifact_name="abc123-config", artifact_type="config",
                wandb_run=run,
            )
            mock_prune.assert_not_called()

    def test_wait_called_before_pruning(self):
        """Artifact.wait() is called before pruning to ensure commit."""
        mgr = UnifiedArtifactManager(verbose=True, keep_n_versions=2)
        trainer = self._make_mock_trainer()
        run = self._make_mock_wandb_run()
        logged = run.log_artifact.return_value

        call_order = []
        logged.wait.side_effect = lambda: call_order.append("wait")

        def mock_prune(**kwargs):
            call_order.append("prune")

        with patch.object(mgr, '_validate_files', return_value={"f.ckpt": "/tmp/f.ckpt"}), \
             patch("wandb.Artifact") as mock_artifact_cls, \
             patch.object(mgr, '_prune_old_versions', side_effect=mock_prune):
            mgr.upload_artifact(
                trainer=trainer, files={"f.ckpt": "/tmp/f.ckpt"},
                artifact_name="abc123-latest", artifact_type="model",
                wandb_run=run,
            )

        assert call_order == ["wait", "prune"]


class TestCrossCollectionIsolation:
    """Verifies that pruning one artifact collection never touches another."""

    def test_best_and_latest_are_independent(self):
        """Pruning 'abc-latest' never queries or touches 'abc-best'."""
        mgr = UnifiedArtifactManager(verbose=True, keep_n_versions=2)

        latest_versions = make_versions(4, aliased_indices=[0])
        best_versions = make_versions(3, aliased_indices=[0])

        def mock_artifacts(type_name, name):
            if "latest" in name:
                return latest_versions
            elif "best" in name:
                return best_versions
            return []

        with patch("wandb.Api") as mock_api_cls:
            mock_api = mock_api_cls.return_value
            mock_api.artifacts.side_effect = mock_artifacts

            # Prune latest collection
            mgr._prune_old_versions(
                entity="e", project="p", artifact_name="abc-latest",
                artifact_type="model",
            )

        # Latest: keep 2 newest, delete 2 oldest
        assert latest_versions[2]._deleted  # v1
        assert latest_versions[3]._deleted  # v0

        # Best: completely untouched
        for v in best_versions:
            assert not v._deleted
