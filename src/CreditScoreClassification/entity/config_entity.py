

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir : Path
    local_data_file : Path


@dataclass(frozen= True)
class DataValidationConfig:
    root_dir : Path
    local_data_file : Path
    STATUS_FILE : Path
    all_schema : Path



@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir : Path
    local_data_file : Path
    train_path : Path
    test_path : Path