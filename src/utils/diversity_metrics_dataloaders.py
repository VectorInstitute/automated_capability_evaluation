"""Dataloaders for extracting text from different dataset formats for diversity metrics.

This module provides a flexible interface for loading data from different formats
and extracting the text needed for embedding generation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
import os
import logging

logger = logging.getLogger(__name__)


class DatasetDataloader(ABC):
    """Abstract base class for dataloaders that extract text from datasets."""
    
    @abstractmethod
    def get_name(self, item: Any) -> str:
        """Extract the name/title from a dataset item."""
        pass
    
    @abstractmethod
    def get_description(self, item: Any) -> str:
        """Extract the description from a dataset item."""
        pass
    
    def get_area(self, item: Any) -> Optional[str]:
        """Extract the area/category from a dataset item (optional)."""
        return None
    
    def get_instructions(self, item: Any) -> Optional[str]:
        """Extract instructions from a dataset item (optional)."""
        return None
    
    def get_sample_tasks(self, item: Any, max_samples: int = 5) -> List[str]:
        """Extract sample tasks/problems from a dataset item (optional).
        
        Args:
            item: The dataset item
            max_samples: Maximum number of sample tasks to return
            
        Returns:
            List of task/problem strings
        """
        return []
    
    def extract_text(self, item: Any, max_task_samples: int = 5) -> str:
        """Extract full text representation from a dataset item.
        
        Args:
            item: The dataset item
            max_task_samples: Maximum number of sample tasks to include
            
        Returns:
            Text string suitable for embedding generation
        """
        text_parts = [
            f"Name: {self.get_name(item)}",
            f"Description: {self.get_description(item)}",
        ]
        
        area = self.get_area(item)
        if area:
            text_parts.append(f"Area: {area}")
        
        instructions = self.get_instructions(item)
        if instructions:
            text_parts.append(f"Instructions: {instructions}")
        
        tasks = self.get_sample_tasks(item, max_samples=max_task_samples)
        if tasks:
            task_texts = [f"Task: {task}" for task in tasks]
            text_parts.append("Tasks: " + " | ".join(task_texts))
        
        return " | ".join(text_parts)


class CapabilityDataloader(DatasetDataloader):
    """Dataloader for capability format (capability.json structure).
    
    Can handle either:
    - A single capability directory (contains capability.json)
    - A parent directory containing multiple capability subdirectories
    """
    
    def __init__(self, capability_dir: str):
        """Initialize with a capability directory.
        
        Args:
            capability_dir: Path to capability directory or parent directory with capability subdirectories
        """
        self.capability_dir = capability_dir
        self.capabilities = self._load_capabilities()
    
    def _load_capabilities(self) -> List[Dict[str, Any]]:
        """Load capabilities from directory.
        
        Returns:
            List of capability data dictionaries
        """
        capabilities = []
        
        # Check if this is a single capability directory (has capability.json)
        single_cap_json = os.path.join(self.capability_dir, "capability.json")
        if os.path.exists(single_cap_json):
            with open(single_cap_json, 'r') as f:
                capabilities.append(json.load(f))
            return capabilities
        
        # Otherwise, treat as parent directory with multiple capability subdirectories
        if not os.path.isdir(self.capability_dir):
            raise FileNotFoundError(f"Capability directory does not exist: {self.capability_dir}")
        
        for item_name in os.listdir(self.capability_dir):
            item_path = os.path.join(self.capability_dir, item_name)
            if not os.path.isdir(item_path):
                continue
            
            cap_json = os.path.join(item_path, "capability.json")
            if os.path.exists(cap_json):
                with open(cap_json, 'r') as f:
                    capabilities.append(json.load(f))
        
        if not capabilities:
            raise FileNotFoundError(f"No capabilities found in {self.capability_dir}")
        
        return capabilities
    
    def get_name(self, item: Dict[str, Any]) -> str:
        return item.get("capability_name", "")
    
    def get_description(self, item: Dict[str, Any]) -> str:
        return item.get("capability_description", "")
    
    def get_area(self, item: Dict[str, Any]) -> Optional[str]:
        return item.get("capability_area")
    
    def get_instructions(self, item: Dict[str, Any]) -> Optional[str]:
        return item.get("capability_instructions")
    
    def get_sample_tasks(self, item: Dict[str, Any], max_samples: int = 5) -> List[str]:
        tasks = item.get("capability_data", [])
        problems = []
        for task in tasks[:max_samples]:
            if isinstance(task, dict):
                problem = task.get('problem', '')
                if problem:
                    problems.append(problem)
        return problems


class HuggingFaceDatasetDataloader(DatasetDataloader):
    """Dataloader for HuggingFace datasets.
    
    Simply extracts text from a specified field in each dataset item.
    """
    
    def __init__(self, dataset, text_field: str = "problem"):
        """Initialize with a HuggingFace dataset.
        
        Args:
            dataset: HuggingFace dataset or iterable of dicts
            text_field: Field name containing the text to embed (e.g., "problem", "text", "content")
        """
        self.dataset = dataset
        self.text_field = text_field
    
    def get_name(self, item: Dict[str, Any]) -> str:
        return ""  # Not used in simplified version
    
    def get_description(self, item: Dict[str, Any]) -> str:
        return str(item.get(self.text_field, ""))
    
    def get_area(self, item: Dict[str, Any]) -> Optional[str]:
        return None  # Not used in simplified version
    
    def get_instructions(self, item: Dict[str, Any]) -> Optional[str]:
        return None  # Not used in simplified version
    
    def get_sample_tasks(self, item: Dict[str, Any], max_samples: int = 5) -> List[str]:
        return []  # Not used in simplified version
    
    def extract_text(self, item: Any, max_task_samples: int = 5) -> str:
        """Extract text from the specified field.
        
        Args:
            item: The dataset item
            max_task_samples: Ignored (kept for interface compatibility)
            
        Returns:
            Text string from the specified field
        """
        if isinstance(item, dict):
            return str(item.get(self.text_field, ""))
        return str(item)


class JSONLDataloader(DatasetDataloader):
    """Dataloader for JSONL files (one JSON object per line).
    
    Flexible loader that can handle various JSONL formats by specifying field mappings.
    """
    
    def __init__(self, jsonl_path: str, name_field: str = "name", 
                 description_field: str = "description",
                 area_field: Optional[str] = None,
                 instructions_field: Optional[str] = None,
                 task_field: Optional[str] = "problem"):
        """Initialize with a JSONL file path.
        
        Args:
            jsonl_path: Path to JSONL file
            name_field: Field name for name/title
            description_field: Field name for description
            area_field: Field name for area/category (optional)
            instructions_field: Field name for instructions (optional)
            task_field: Field name for tasks/problems (optional)
        """
        self.jsonl_path = jsonl_path
        self.name_field = name_field
        self.description_field = description_field
        self.area_field = area_field
        self.instructions_field = instructions_field
        self.task_field = task_field
    
    def get_name(self, item: Dict[str, Any]) -> str:
        return str(item.get(self.name_field, ""))
    
    def get_description(self, item: Dict[str, Any]) -> str:
        return str(item.get(self.description_field, ""))
    
    def get_area(self, item: Dict[str, Any]) -> Optional[str]:
        if self.area_field:
            return item.get(self.area_field)
        return None
    
    def get_instructions(self, item: Dict[str, Any]) -> Optional[str]:
        if self.instructions_field:
            return item.get(self.instructions_field)
        return None
    
    def get_sample_tasks(self, item: Dict[str, Any], max_samples: int = 5) -> List[str]:
        if self.task_field and self.task_field in item:
            task_value = item[self.task_field]
            if isinstance(task_value, str):
                return [task_value]
            elif isinstance(task_value, list):
                return [str(t) for t in task_value[:max_samples] if t]
        return []
    
    def load_items(self) -> List[Dict[str, Any]]:
        """Load all items from the JSONL file."""
        items = []
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items


class CSVDataloader(DatasetDataloader):
    """Dataloader for CSV files."""
    
    def __init__(self, csv_path: str, name_field: str = "name",
                 description_field: str = "description",
                 area_field: Optional[str] = None,
                 instructions_field: Optional[str] = None,
                 task_field: Optional[str] = "problem"):
        """Initialize with a CSV file path.
        
        Args:
            csv_path: Path to CSV file
            name_field: Column name for name/title
            description_field: Column name for description
            area_field: Column name for area/category (optional)
            instructions_field: Column name for instructions (optional)
            task_field: Column name for tasks/problems (optional)
        """
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.name_field = name_field
        self.description_field = description_field
        self.area_field = area_field
        self.instructions_field = instructions_field
        self.task_field = task_field
    
    def get_name(self, item: Dict[str, Any]) -> str:
        return str(item.get(self.name_field, ""))
    
    def get_description(self, item: Dict[str, Any]) -> str:
        return str(item.get(self.description_field, ""))
    
    def get_area(self, item: Dict[str, Any]) -> Optional[str]:
        if self.area_field and self.area_field in item:
            return item.get(self.area_field)
        return None
    
    def get_instructions(self, item: Dict[str, Any]) -> Optional[str]:
        if self.instructions_field and self.instructions_field in item:
            return item.get(self.instructions_field)
        return None
    
    def get_sample_tasks(self, item: Dict[str, Any], max_samples: int = 5) -> List[str]:
        if self.task_field and self.task_field in item:
            task_value = item[self.task_field]
            if isinstance(task_value, str):
                return [task_value]
        return []
    
    def load_items(self) -> List[Dict[str, Any]]:
        """Load all items from the CSV file."""
        return self.df.to_dict('records')


def load_texts_from_dataloader(dataloader: DatasetDataloader) -> List[str]:
    """Extract texts from a dataloader for embedding generation.
    
    Args:
        dataloader: A DatasetDataloader instance
        
    Returns:
        List of text strings ready for embedding
    """
    texts = []
    
    if isinstance(dataloader, CapabilityDataloader):
        # Capability format: iterate over all capabilities
        for capability_data in dataloader.capabilities:
            texts.append(dataloader.extract_text(capability_data))
    elif isinstance(dataloader, HuggingFaceDatasetDataloader):
        # HuggingFace dataset: iterate over items
        for item in dataloader.dataset:
            texts.append(dataloader.extract_text(item))
    elif isinstance(dataloader, JSONLDataloader):
        # JSONL: load all items
        items = dataloader.load_items()
        for item in items:
            texts.append(dataloader.extract_text(item))
    elif isinstance(dataloader, CSVDataloader):
        # CSV: load all items
        items = dataloader.load_items()
        for item in items:
            texts.append(dataloader.extract_text(item))
    else:
        # Generic: try to iterate
        try:
            if hasattr(dataloader, 'dataset'):
                for item in dataloader.dataset:
                    texts.append(dataloader.extract_text(item))
            elif hasattr(dataloader, 'load_items'):
                items = dataloader.load_items()
                for item in items:
                    texts.append(dataloader.extract_text(item))
            else:
                logger.error("Dataloader does not have dataset or load_items method")
                raise ValueError("Dataloader must have dataset attribute or load_items method")
        except Exception as e:
            logger.error(f"Could not extract texts from dataloader: {e}")
            raise
    
    return texts

