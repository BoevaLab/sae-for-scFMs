"""
Data Loading Module
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Iterator, Callable
import torch
from torch.utils.data import SequentialSampler
from omegaconf import DictConfig
import logging
import os

from sae4scfm.core.io_utils import load_adata

log = logging.getLogger(__name__)

# ============================================================================

class DataLoader(ABC):
    """
    Abstract interface for all data loading.
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._exhausted = False
    
    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Return iterator over data batches"""
        pass
    
    @abstractmethod
    def __next__(self) -> Dict[str, torch.Tensor]:
        """
        Get next batch.
        Raises StopIteration when all data exhausted.
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset iterator to beginning (if applicable)"""
        pass
    
    def is_exhausted(self) -> bool:
        """Check if all data has been consumed"""
        return self._exhausted

# ============================================================================

class SimpleDataLoader(DataLoader):
    """
    Single file/dataset loader with lazy initialization.
    """
    
    def __init__(
        self, 
        cfg: DictConfig, 
        adapter,
        test_mode: bool = False
    ):
        super().__init__(cfg)
        self.adapter = adapter
        self.dataloader = None
        self.iterator = None
        self.test_mode = test_mode
    
    def _initialize_dataloader(self):
        """Lazy initialization: load data and create PyTorch DataLoader"""
        if self.dataloader is not None:
            return

        # Load adata with appropriate subsampling
        if self.test_mode:
            # Load test split using cfg.data.preprocess.split fraction
            sub_sample = self.cfg.data.preprocess.split if hasattr(self.cfg.data, 'preprocess') else 1.0
        else:
            # Load full data (no subsampling for training)
            sub_sample = 1.0
        
        adata = load_adata(self.cfg, sub_sample=sub_sample)
        
        # Preprocess (adapter-specific)
        self.dataloader = self.adapter.preprocess_data(adata, shuffle=not self.test_mode)
        log.info(f"Created DataLoader with {len(self.dataloader)} batches")
    
    def __iter__(self):
        self._initialize_dataloader()
        self.iterator = iter(self.dataloader)
        self._exhausted = False
        return self
    
    def __next__(self) -> Dict[str, torch.Tensor]:
        if self.iterator is None:
            self.__iter__()
        
        try:
            return next(self.iterator)
        except StopIteration:
            self._exhausted = True
            raise
    
    def reset(self):
        """Reset to beginning of dataset"""
        self._exhausted = False
        self.iterator = None

class QueryDataLoader(DataLoader):
    """
    Handles multiple separate data files (queries, partitions) that need to be processed.
    Usually data downloaded from the cellxgene census.
    """
    
    def __init__(
        self, 
        cfg: DictConfig, 
        adapter,
        test_mode: bool = False
    ):
        """
        Args:
            cfg: Configuration
            adapter: ModelAdapter instance
        """
        super().__init__(cfg)
        self.query_list = cfg.data.query_list
        self.adapter = adapter
        self.test_mode = test_mode
        
        # Check if queries are partitioned
        self.n_partitions = cfg.data.get('n_partitions', None)
        
        self.current_query_idx = 0
        self.current_partition_idx = 0
        self.current_dataloader = None
        self.current_iterator = None
    
    def _load_partition(self):
        """
        Load a specific query/partition as a dataloader.
        """
        if self.current_query_idx >= len(self.query_list):
            self._exhausted = True
            return
        
        query_id = self.query_list[self.current_query_idx]
        n_partitions = self.n_partitions[self.current_query_idx]

        self.cfg.data.file_name = f'{query_id}/partition_{self.current_partition_idx}.h5ad'

        log.info(
            f"Loading query {self.current_query_idx + 1}/{len(self.query_list)}: '{query_id}', "
            f"partition {self.current_partition_idx + 1}/{n_partitions}"
        )

        adata = load_adata(self.cfg)
        
        self.current_dataloader = self.adapter.preprocess_data(adata, shuffle=not self.test_mode)
        self.current_iterator = iter(self.current_dataloader)
    
    def __iter__(self):
        self._load_partition()
        self._exhausted = False
        return self
    
    def _advance_to_next(self) -> bool:
        """
        Advance to next partition or query.
        Returns True if successful, False if all data exhausted.
        """
        n_partitions = self.n_partitions[self.current_query_idx]
        
        # Try next partition in current query
        if self.current_partition_idx < (n_partitions - 1):
            self.current_partition_idx += 1
            self._load_partition()  # Reload with new partition
            return True
        
        # Try next query
        if self.current_query_idx < (len(self.query_list) - 1):
            self.current_query_idx += 1
            self.current_partition_idx = 0
            self._load_partition()
            return True
        
        # All data exhausted
        return False
    
    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._exhausted:
            raise StopIteration
        
        try:
            return next(self.current_iterator)
        except StopIteration:
            # Current query/partition exhausted, try next
            if self._advance_to_next():
                return self.__next__()  # Recursively get first batch
            else:
                # All queries exhausted
                self._exhausted = True
                raise StopIteration
    
    def reset(self):
        """Reset to first query/partition"""
        self.current_query_idx = 0
        self.current_partition_idx = 0
        self._load_partition()
        self._exhausted = False

class ScBankDataLoader(DataLoader):
    """
    Handles scbank format
    """
    
    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: Configuration
        """
        super().__init__(cfg)
        
        # Query/partition tracking
        self.query_list = cfg.data.query_list
        self.n_partitions = cfg.data.n_partitions
        
        self.current_query_idx = 0
        self.current_partition_idx = 0
        self.current_iterator = None
        self.current_dataloader = None
        
    def _load_query(self):
        """Load current query"""
        from scgpt.scbank import DataBank

        query = self.query_list[self.current_query_idx]
        self.data_bank = DataBank.from_path(self.cfg.data.data_path + query + "/all_counts")
        self._load_partition()

    def _load_partition(self):
        """Load current partition"""
        from scgpt.data_collator import DataCollator

        if self._is_all_data_exhausted():
            self._exhausted = True
            return
        
        query_id = self.query_list[self.current_query_idx]
        log.info(
            f"Loading query {self.current_query_idx + 1}/{len(self.query_list)} "
            f"('{query_id}'), partition {self.current_partition_idx + 1}/"
            f"{self._get_n_partitions_for_query(self.current_query_idx)}"
        )
        
        dataset = self.data_bank.data_tables['partition_' + str(self.current_partition_idx)].data
        collator = DataCollator(
                do_padding=True,
                pad_token_id=self.cfg.scfm.preprocessing.pad_id,
                pad_value=self.cfg.scfm.preprocessing.pad_value,
                do_mlm=False,
                do_binning=True,
                max_length=self.cfg.scfm.preprocessing.max_len,
                sampling=True,
                keep_first_n_tokens=0,
            )
        n_gpus = torch.cuda.device_count() #n_gpus for buffer
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.data.preprocess.batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=collator,
            drop_last=False,
            num_workers=2 if len(os.sched_getaffinity(0)) / n_gpus >= 4 else 0,
            pin_memory=False if n_gpus > 1 else True,
        )
        
        self.current_iterator = iter(dataloader)
    
    def _is_all_data_exhausted(self) -> bool:
        """Check if we've processed all queries and partitions"""
        return self.current_query_idx >= len(self.query_list)
    
    def _advance_to_next(self) -> bool:
        """
        Advance to next partition or query.
        Returns True if successful, False if all data exhausted.
        """
        n_partitions = self.n_partitions[self.current_query_idx]
        
        # Try next partition in current query
        if self.current_partition_idx < (n_partitions - 1):
            self.current_partition_idx += 1
            self._load_partition()  # Reload with new partition
            return True
        
        # Try next query
        if self.current_query_idx < (len(self.query_list) - 1):
            self.current_query_idx += 1
            self.current_partition_idx = 0
            self._load_query()
            return True
        
        # All data exhausted
        return False
    
    def __iter__(self):
        self._load_query()
        self._exhausted = False
        return self
    
    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._exhausted:
            raise StopIteration
        
        try:
            batch = next(self.current_iterator)
            return self._preprocess_batch(batch)
        except StopIteration:
            # Current partition exhausted, try next
            if self._advance_to_next():
                return self.__next__()  # Recursively get first batch of new partition
            else:
                self._exhausted = True
                raise StopIteration
    
    def reset(self):
        """Reset to first query/partition"""
        self.current_query_idx = 0
        self.current_partition_idx = 0
        self._load_partition()
        self._exhausted = False

class MultiProcessDataLoader(DataLoader):
    """
    Wrapper that distributes partitions across multiple worker processes.
    Each worker loads different partitions in parallel.
    
    Only applicable for partitioned data 
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        base_loader_class,
        worker_id: int,
        n_workers: int,
        **base_loader_kwargs
    ):
        """
        Args:
            cfg: Configuration
            base_loader_class: The underlying loader class (e.g., ScBankDataLoader)
            worker_id: ID of this worker (0 to n_workers-1)
            n_workers: Total number of workers
            **base_loader_kwargs: Args to pass to base loader
        """
        super().__init__(cfg)
        self.worker_id = worker_id
        self.n_workers = n_workers
        
        # Create base loader but override partition assignment
        self.base_loader = base_loader_class(cfg, **base_loader_kwargs)
        
        # Assign partitions to this worker (interleaved)
        self._assign_partitions()
    
    def _assign_partitions(self):
        """
        Assign partitions to this worker using interleaved strategy.
        Worker 0 gets partitions 0, n_workers, 2*n_workers, ...
        Worker 1 gets partitions 1, n_workers+1, 2*n_workers+1, ...
        """
        # Override base loader to start at worker-specific partition
        self.base_loader.current_partition_idx = self.worker_id
        self.partition_stride = self.n_workers
    
    def __iter__(self):
        self.base_loader.current_partition_idx = self.worker_id
        return self
    
    def __next__(self) -> Dict[str, torch.Tensor]:
        try:
            return next(self.base_loader)
        except StopIteration:
            # When partition exhausted, try jump to next partition for this worker
            if not self.base_loader._exhausted:
                self.base_loader.current_partition_idx += self.partition_stride
                return self.__next__()
            raise
    
    def reset(self):
        self.base_loader.reset()
        self.base_loader.current_partition_idx = self.worker_id

# ============================================================================

def create_data_loader(cfg: DictConfig, adapter, **kwargs) -> DataLoader:
    """
    Factory function to create appropriate data loader
    
    Args:
        cfg: Configuration
        adapter: ModelAdapter instance
    
    Returns:
        Appropriate DataLoader instance
    """
    
    # scBank format
    if cfg.data.get('scbank', False):
        log.info("Using ScBankDataLoader")
        return ScBankDataLoader(
            cfg=cfg,
            **kwargs)
    
    # Multi-query or multi-partition
    elif cfg.data.get('census', False) or len(cfg.data.get('query_list', [])) > 1:
        log.info("Using QueryDataLoader")
        return QueryDataLoader(
            cfg=cfg,
            adapter=adapter,
            **kwargs
        )
    
    # Simple single dataloader
    else:
        log.info("Using SimpleDataLoader")
        return SimpleDataLoader(
            cfg=cfg,
            adapter=adapter,
            **kwargs
        )


def create_multiprocess_data_loader(
    cfg: DictConfig,
    worker_id: int,
    n_workers: int,
    **kwargs
) -> DataLoader:
    """
    Create a data loader for a specific worker in multi-process setup.
    
    Args:
        cfg: Hydra configuration
        worker_id: Worker ID (0 to n_workers-1)
        n_workers: Total number of workers
        **kwargs: Args passed to underlying loader
    
    Returns:
        DataLoader configured for this worker
    """
    # Only ScBank supports multi-process loading (partitioned data)
    if not cfg.data.get('scbank', False):
        raise ValueError("Multi-process loading only supported for partitioned data (scbank)")
    
    return MultiProcessDataLoader(
        cfg=cfg,
        base_loader_class=ScBankDataLoader,
        worker_id=worker_id,
        n_workers=n_workers,
        **kwargs
    )
