"""Unit tests for HuggingFace Hub upload functionality.

These tests mock the huggingface_hub library to test upload functionality
without requiring actual HF authentication or network access.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from swe_forge.export.hf_upload import (
    DatasetCard,
    HfUploadError,
    upload_dataset_folder,
    upload_to_hf,
)


class TestUploadToHf:
    """Tests for upload_to_hf function."""

    def test_upload_single_file_success(self, tmp_path: Path) -> None:
        """Test successful upload of a single file."""
        # Create a test file
        test_file = tmp_path / "test.json"
        test_file.write_text('{"test": "data"}')

        with patch("swe_forge.export.hf_upload.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api_class.return_value = mock_api

            # Mock create_repo
            mock_api.create_repo = MagicMock()
            mock_api.upload_file = MagicMock()

            result = upload_to_hf(
                path=str(test_file),
                repo_id="test-org/test-dataset",
                token="hf_test_token",
                private=False,
            )

            # Verify create_repo was called
            mock_api.create_repo.assert_called_once()
            assert (
                mock_api.create_repo.call_args[1]["repo_id"] == "test-org/test-dataset"
            )
            assert mock_api.create_repo.call_args[1]["private"] is False

            # Verify upload_file was called
            mock_api.upload_file.assert_called_once()
            assert result is True

    def test_upload_with_token_parameter(self, tmp_path: Path) -> None:
        """Test that explicit token parameter is used."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"test": "data"}')

        with patch("swe_forge.export.hf_upload.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api_class.return_value = mock_api
            mock_api.create_repo = MagicMock()
            mock_api.upload_file = MagicMock()

            upload_to_hf(
                path=str(test_file),
                repo_id="test-org/test-dataset",
                token="explicit_token",
            )

            # HfApi should be instantiated with the explicit token
            mock_api_class.assert_called_once()
            call_kwargs = mock_api_class.call_args[1]
            assert call_kwargs["token"] == "explicit_token"

    def test_upload_uses_env_token(self, tmp_path: Path) -> None:
        """Test that HF_TOKEN env var is used when no explicit token."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"test": "data"}')

        with patch.dict(os.environ, {"HF_TOKEN": "env_token"}):
            with patch("swe_forge.export.hf_upload.HfApi") as mock_api_class:
                mock_api = MagicMock()
                mock_api_class.return_value = mock_api
                mock_api.create_repo = MagicMock()
                mock_api.upload_file = MagicMock()

                upload_to_hf(
                    path=str(test_file),
                    repo_id="test-org/test-dataset",
                    # No explicit token
                )

                # HfApi should be instantiated with the env token
                mock_api_class.assert_called_once()
                call_kwargs = mock_api_class.call_args[1]
                assert call_kwargs["token"] == "env_token"

    def test_upload_raises_on_missing_token(self, tmp_path: Path) -> None:
        """Test that HfUploadError is raised when no token available."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"test": "data"}')

        # Clear any HF_TOKEN env var
        env = os.environ.copy()
        env.pop("HF_TOKEN", None)

        with patch.dict(os.environ, env, clear=True):
            with patch("swe_forge.export.hf_upload.HfApi") as mock_api_class:
                mock_api_class.side_effect = ValueError("Token is required")

                with pytest.raises(HfUploadError, match="Token required"):
                    upload_to_hf(
                        path=str(test_file),
                        repo_id="test-org/test-dataset",
                    )

    def test_upload_creates_private_repo(self, tmp_path: Path) -> None:
        """Test that private=True creates a private repo."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"test": "data"}')

        with patch("swe_forge.export.hf_upload.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api_class.return_value = mock_api
            mock_api.create_repo = MagicMock()
            mock_api.upload_file = MagicMock()

            upload_to_hf(
                path=str(test_file),
                repo_id="test-org/test-dataset",
                token="test_token",
                private=True,
            )

            mock_api.create_repo.assert_called_once()
            assert mock_api.create_repo.call_args[1]["private"] is True


class TestUploadDatasetFolder:
    """Tests for upload_dataset_folder function."""

    def test_upload_folder_success(self, tmp_path: Path) -> None:
        """Test successful upload of a folder as dataset."""
        # Create test folder with files
        folder = tmp_path / "dataset"
        folder.mkdir()
        (folder / "data.json").write_text('{"samples": []}')
        (folder / "metadata.json").write_text('{"version": "1.0"}')

        with patch("swe_forge.export.hf_upload.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api_class.return_value = mock_api
            mock_api.create_repo = MagicMock()
            mock_api.upload_folder = MagicMock()

            result = upload_dataset_folder(
                folder=str(folder),
                repo_id="test-org/test-dataset",
                token="test_token",
            )

            mock_api.create_repo.assert_called_once()
            mock_api.upload_folder.assert_called_once()
            assert result is True

    def test_upload_folder_with_dataset_card(self, tmp_path: Path) -> None:
        """Test that dataset card is generated and uploaded."""
        folder = tmp_path / "dataset"
        folder.mkdir()
        (folder / "data.json").write_text('{"samples": []}')

        with patch("swe_forge.export.hf_upload.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api_class.return_value = mock_api
            mock_api.create_repo = MagicMock()
            mock_api.upload_file = MagicMock()
            mock_api.upload_folder = MagicMock()

            # Mock DatasetCard generation
            with patch("swe_forge.export.hf_upload.DatasetCard") as mock_card_class:
                mock_card = Mock()
                mock_card.to_markdown.return_value = (
                    "---\ntags:\n- test\n---\n# Dataset"
                )
                mock_card_class.from_template = MagicMock(return_value=mock_card)

                upload_dataset_folder(
                    folder=str(folder),
                    repo_id="test-org/test-dataset",
                    token="test_token",
                    generate_card=True,
                )

                # Should upload README.md
                # Check that upload_file was called at least once for README
                calls = mock_api.upload_file.call_args_list
                assert any("README.md" in str(c) for c in calls)

    def test_upload_nonexistent_folder_raises(self) -> None:
        """Test that HfUploadError is raised for nonexistent folder."""
        with patch("swe_forge.export.hf_upload.HfApi"):
            with pytest.raises(HfUploadError, match="Folder.*does not exist"):
                upload_dataset_folder(
                    folder="/nonexistent/folder",
                    repo_id="test-org/test-dataset",
                    token="test_token",
                )


class TestDatasetCard:
    """Tests for DatasetCard generation."""

    def test_dataset_card_basic(self) -> None:
        """Test basic dataset card generation."""
        card = DatasetCard(
            repo_id="test-org/test-dataset",
            task_type="text-generation",
        )

        markdown = card.to_markdown()

        assert "test-org/test-dataset" in markdown
        assert "text-generation" in markdown
        assert "---" in markdown  # YAML frontmatter

    def test_dataset_card_with_license(self) -> None:
        """Test dataset card with license specified."""
        card = DatasetCard(
            repo_id="test-org/test-dataset",
            task_type="text-classification",
            license="apache-2.0",
        )

        markdown = card.to_markdown()

        assert "apache-2.0" in markdown

    def test_dataset_card_with_description(self) -> None:
        """Test dataset card with custom description."""
        card = DatasetCard(
            repo_id="test-org/test-dataset",
            task_type="text-generation",
            description="This is a test dataset description.",
        )

        markdown = card.to_markdown()

        assert "test dataset description" in markdown

    def test_dataset_card_yaml_frontmatter(self) -> None:
        """Test that YAML frontmatter is valid."""
        card = DatasetCard(
            repo_id="test-org/test-dataset",
            task_type="text-generation",
            license="mit",
            tags=["synthetic", "code"],
        )

        markdown = card.to_markdown()

        # Check for YAML frontmatter boundaries
        assert markdown.startswith("---")
        assert "---\n" in markdown

        # Check for expected tags
        assert "synthetic" in markdown
        assert "code" in markdown

    def test_dataset_card_from_template(self) -> None:
        """Test creating dataset card from template."""
        card = DatasetCard.from_template(
            repo_id="my-org/my-dataset",
            task_type="code-generation",
        )

        assert card.repo_id == "my-org/my-dataset"
        assert card.task_type == "code-generation"

    def test_dataset_card_repo_url(self) -> None:
        """Test that repo_url is correctly generated."""
        card = DatasetCard(repo_id="test-org/test-dataset")

        assert card.repo_url == "https://huggingface.co/datasets/test-org/test-dataset"
