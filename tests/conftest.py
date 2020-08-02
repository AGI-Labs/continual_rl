"""
conftest.py files get auto-discovered by pytest and do not need to be imported.
"""

import pytest
import shutil
from pathlib import Path


@pytest.fixture
def set_tmp_directory(request):
    output_dir = str(Path(__file__).parent.absolute().joinpath("unit_test_tmp_dir"))
    request.node.experiment_output_dir = output_dir


@pytest.fixture
def cleanup_experiment(request):
    # Courtesy: https://stackoverflow.com/questions/44716237/pytest-passing-data-for-cleanup
    def cleanup():
        path_to_remove = request.node.experiment_output_dir
        print(f"Attempting to remove {path_to_remove}")

        try:
            shutil.rmtree(path_to_remove)
        except FileNotFoundError:
            pass

    request.addfinalizer(cleanup)
