"""
Tests for the load_data function from the data_utils module.

This module contains a test case for the load_data function, which is responsible for
loading a dataset based on the provided configuration.

Functions:
    test_load_data(): Tests the load_data function to ensure it loads the correct number
    of data samples.
"""

import os

from src.utils import load_data
from src.utils.data_utils import copy_file, transfer_inspect_log_to_gcp


def test_load_data():
    """
    Test the load_data function to ensure it loads the dataset correctly.

    This test checks if the dataset loaded by the load_data function has the expected
    size.
    The data configuration includes the dataset name, split, subset, and size.

    The expected size is extracted from the data configuration and compared with the
    length of the dataset returned by the load_data function.

    Raises
    ------
        AssertionError: If the length of the dataset does not match the expected size.
    """
    data_cfg = {
        "dataset_name": "qwedsacf/competition_math",
        "split": "train",
        "subset": "",
        "num_repr_samples": 3,
        "streaming": False,
    }
    test_cfg = {
        "size": 12500,
    }

    dataset = load_data(**data_cfg)
    assert len(dataset) == test_cfg["size"]


def test_copy_file_within_gcp(mocker):
    """Test the copy_file function for copying files within GCP buckets."""
    src = "gs://source-bucket/source-file.txt"
    dest = "gs://dest-bucket/dest-file.txt"

    # Mock the GCP storage client and its methods
    mock_client = mocker.patch("src.utils.data_utils.storage.Client")
    mock_src_bucket = mocker.Mock()
    mock_dest_bucket = mocker.Mock()
    mock_src_blob = mocker.Mock()
    mock_dest_blob = mocker.Mock()

    mock_client.return_value.bucket.side_effect = [mock_src_bucket, mock_dest_bucket]
    mock_src_bucket.blob.return_value = mock_src_blob
    mock_dest_bucket.blob.return_value = mock_dest_blob

    # Call the function
    copy_file(src, dest)

    # Verify the correct calls were made
    mock_client.assert_called_once()
    mock_client.return_value.bucket.assert_any_call("source-bucket")
    mock_client.return_value.bucket.assert_any_call("dest-bucket")
    mock_src_bucket.blob.assert_called_once_with("source-file.txt")
    mock_dest_bucket.blob.assert_called_once_with("dest-file.txt")
    mock_dest_blob.rewrite.assert_called_once_with(mock_src_blob)


def test_copy_file_local_to_gcp(mocker, tmp_path):
    """Test the copy_file function for copying files from local to GCP."""
    src = tmp_path / "source-file.txt"
    dest = "gs://dest-bucket/dest-file.txt"

    try:
        # Create a temporary source file
        src.write_text("test content")

        # Mock the GCP storage client and its methods
        mock_client = mocker.patch("src.utils.data_utils.storage.Client")
        mock_bucket = mocker.Mock()
        mock_blob = mocker.Mock()

        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        # Call the function
        copy_file(str(src), dest)

        # Verify the correct calls were made
        mock_client.assert_called_once()
        mock_client.return_value.bucket.assert_called_once_with("dest-bucket")
        mock_bucket.blob.assert_called_once_with("dest-file.txt")
        mock_blob.upload_from_filename.assert_called_once_with(str(src))
    finally:
        # Clean up temporary files
        if src.exists():
            src.unlink()


def test_copy_file_local_to_local(tmp_path):
    """Test the copy_file function for copying files locally."""
    src = tmp_path / "source-file.txt"
    dest = tmp_path / "dest-file.txt"

    try:
        # Create a temporary source file
        src.write_text("test content")

        # Call the function
        copy_file(str(src), str(dest))

        # Verify the file was copied correctly
        assert dest.exists()
        assert dest.read_text() == "test content"
    finally:
        # Clean up temporary files
        if src.exists():
            src.unlink()
        if dest.exists():
            dest.unlink()


def test_transfer_inspect_log_to_gcp_success(mocker, tmp_path):
    """Test transfer_inspect_log_to_gcp function for successful transfer."""
    src_dir = tmp_path / "src"
    gcp_dir = "gs://dest-bucket"
    src_dir.mkdir()
    src_file = src_dir / "inspect_log.json"
    src_file.write_text("{}")  # Create a dummy JSON file

    try:
        # Mock the copy_file function
        mock_copy_file = mocker.patch("src.utils.data_utils.copy_file")

        # Call the function
        transfer_inspect_log_to_gcp(str(src_dir), gcp_dir)

        # Verify the correct calls were made
        mock_copy_file.assert_called_once_with(
            str(src_file), os.path.join(gcp_dir, "inspect_log.json")
        )
    finally:
        # Clean up temporary files
        if src_file.exists():
            src_file.unlink()
        if src_dir.exists():
            src_dir.rmdir()


def test_transfer_inspect_log_to_gcp_multiple_files_error(tmp_path):
    """Test transfer_inspect_log_to_gcp raises an error for multiple files."""
    src_dir = tmp_path / "src"
    gcp_dir = "gs://dest-bucket"
    src_dir.mkdir()
    file1 = src_dir / "file1.json"
    file2 = src_dir / "file2.json"
    file1.write_text("{}")
    file2.write_text("{}")

    try:
        # Call the function and verify it raises an AssertionError
        transfer_inspect_log_to_gcp(str(src_dir), gcp_dir)
    except AssertionError as e:
        assert str(e) == f"Expected only one file in {src_dir}, but found 2 files."
    finally:
        # Clean up temporary files
        if file1.exists():
            file1.unlink()
        if file2.exists():
            file2.unlink()
        if src_dir.exists():
            src_dir.rmdir()


def test_transfer_inspect_log_to_gcp_non_json_file_error(tmp_path):
    """Test transfer_inspect_log_to_gcp raises an error for non-JSON files."""
    src_dir = tmp_path / "src"
    gcp_dir = "gs://dest-bucket"
    src_dir.mkdir()
    src_file = src_dir / "inspect_log.txt"
    src_file.write_text("test content")  # Create a non-JSON file

    try:
        # Call the function and verify it raises a ValueError
        transfer_inspect_log_to_gcp(str(src_dir), gcp_dir)
    except ValueError as e:
        assert str(e) == "Expected a .json file, but got: inspect_log.txt"
    finally:
        # Clean up temporary files
        if src_file.exists():
            src_file.unlink()
        if src_dir.exists():
            src_dir.rmdir()
