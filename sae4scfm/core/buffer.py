"""
Activation Buffer for SAE Training
"""

import torch
import torch.multiprocessing as mp
import logging
from typing import Optional, Dict, Any, List
from omegaconf import DictConfig

from sae4scfm.core.data_loader import DataLoader

log = logging.getLogger(__name__)

# ============================================================================

class ActivationBuffer:
    """
    Unified activation buffer that collects activations from a model via adapter interface.
    Supports both single-process and multi-process modes.
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        adapter,  # ModelAdapter instance
        data_loader: DataLoader = None,
        device_training: str = 'cuda',
        device_buffer: Optional[List[str]] = None,
        use_multiprocessing: bool = False
    ):
        """
        Args:
            cfg: Hydra config with buffer settings under cfg.buffer
            adapter: ModelAdapter instance
            data_loader: DataLoader (only used in single-process mode)
            device_training: Device for SAE training
            device_buffer: Device(s) for buffer population (list for multiprocessing)
            use_multiprocessing: Whether to use multi-GPU producer-consumer pattern
        """
        self.cfg = cfg
        self.adapter = adapter
        self.data_loader = data_loader  # Only for single-process mode
        self.device_training = device_training
        self.use_multiprocessing = use_multiprocessing
        
        # Buffer configuration
        self.ctx_len = cfg.buffer.ctx_len
        self.n_ctxs = cfg.buffer.n_ctxs
        self.buffer_size = int(self.n_ctxs * self.ctx_len)
        self.out_batch_size = cfg.buffer.out_batch_size
        self.refresh_batch_size = cfg.data.preprocess.batch_size
        self.activation_dim = cfg.scfm.activation_dim
        
        # Setup adapter hooks for activation extraction
        self._setup_activation_hooks()
        
        # Initialize buffer storage
        self._init_buffer_storage()
        
        # Multiprocessing setup
        if use_multiprocessing:
            self.device_buffer = device_buffer
            self._init_multiprocessing()
        else:
            self.device_buffer = device_buffer[0]
            self.adapter.model.to(self.device_buffer)
        
        # Evaluation configuration
        self.eval_strategy = cfg.buffer.get('eval_strategy', 'cached')
        self.n_eval_batches = cfg.buffer.get('n_eval_batches', 10)
        
        # Auto-calculate batches per eval call for on-demand strategy
        if self.eval_strategy == 'on_demand':
            total_steps = cfg.sae.get('steps', 1000)
            eval_every = cfg.experiment.get('eval_every', 100)
            n_eval_calls = max(1, total_steps // eval_every)
            self.n_eval_batches_per_call = max(1, self.n_eval_batches // n_eval_calls)
            log.info(
                f"On-demand eval: {self.n_eval_batches_per_call} batches/call "
                f"(~{n_eval_calls} calls total)"
            )
        
        # Cached strategy storage
        self.eval_batches_original = []
        self.eval_cache_initialized = False
        
    def _setup_activation_hooks(self):
        """Setup hooks via adapter to extract activations at specified layer"""
        self.adapter.setup_hook()
    
    def _init_buffer_storage(self):
        """Initialize activation storage tensors"""
        self.activations = torch.empty(
            self.buffer_size, 
            self.activation_dim, 
            device='cpu'
        )
        self.read_mask = torch.ones(self.buffer_size, dtype=torch.bool, device='cpu')
        self.overflow = torch.empty(0, self.activation_dim, device='cpu')
        self.is_exhausted = False
    
    def _init_multiprocessing(self):
        """Initialize multiprocessing primitives"""
        self.n_producers = len(self.device_buffer)
        
        # Share tensors across processes
        self.activations.share_memory_()
        self.read_mask.share_memory_()
        
        # Synchronization primitives
        self.lock = mp.Lock()
        self.not_empty = mp.Condition(self.lock)
        self.not_full = mp.Condition(self.lock)
        self.unread_count = mp.Value('i', 0)
        self.n_producers_done = mp.Value('i', 0)
        
        # Shared eval batch storage
        #self.eval_batch_genes = torch.empty(
        #    self.refresh_batch_size,
        #    self.cfg.data.preprocess.max_len,
        #    dtype=torch.long
        #).share_memory_()
        #self.eval_batch_values = torch.empty(
        #    self.refresh_batch_size,
        #    self.cfg.data.preprocess.max_len,
        #    dtype=torch.float32
        #).share_memory_()
    
    # ========================================================================
    # Iterator Interface (Consumer Side)
    # ========================================================================
    
    def __iter__(self):
        return self
    
    def __next__(self) -> torch.Tensor:
        """
        Get next batch of activations for SAE training.
        
        Returns:
            Tensor of shape (out_batch_size, activation_dim)
        """
        if self.use_multiprocessing:
            return self._next_multiprocess()
        else:
            return self._next_singleprocess()
    
    def _next_singleprocess(self) -> torch.Tensor:
        """Single-process mode: Refresh when buffer is half empty"""
        # Refresh buffer if needed
        if not self.is_exhausted:
            if self.read_mask.sum() >= self.buffer_size // 2:
                self._refresh_buffer()
        
        # Check if we have enough data
        unread_count = (~self.read_mask).sum().item()
        if unread_count < self.out_batch_size:
            raise StopIteration
        
        # Sample and return batch
        return self._sample_batch()
    
    def _next_multiprocess(self) -> torch.Tensor:
        """Multi-process mode: Wait for producers to fill buffer"""
        with self.not_empty:
            # Wait until buffer is sufficiently full or all producers done
            while self.unread_count.value < self.buffer_size // 1.2:
                if self.n_producers_done.value == self.n_producers:
                    if self.unread_count.value < self.out_batch_size:
                        raise StopIteration
                    break
                self.not_empty.wait()
            
            # Sample batch
            batch = self._sample_batch()
            
            # Update count and signal producers
            self.unread_count.value -= self.out_batch_size
            self.not_full.notify_all()
            
            return batch
    
    def _sample_batch(self) -> torch.Tensor:
        """Sample random unread activations from buffer"""
        unread_indices = (~self.read_mask).nonzero(as_tuple=False).squeeze()
        
        # Random sampling
        perm = torch.randperm(len(unread_indices), device='cpu')
        selected = unread_indices[perm[:self.out_batch_size]]
        
        # Mark as read
        self.read_mask[selected] = True
        
        # Return batch on training device
        return self.activations[selected].clone().to(self.device_training)
    
    # ========================================================================
    # Buffer Refresh (Producer Side)
    # ========================================================================
    
    def _refresh_buffer(self):
        """Refill buffer with new activations (single-process mode)"""
        # Initialize eval cache on first refresh (cached strategy only)
        if not self.eval_cache_initialized and self.eval_strategy == 'cached':
            self._initialize_eval_cache_from_loader()
        
        # Compact buffer: remove read activations
        self.activations = self.activations[~self.read_mask]
        current_size = len(self.activations)
        
        # Add overflow from previous refresh
        if len(self.overflow) > 0:
            self.activations = torch.cat([self.activations, self.overflow], dim=0)
            current_size += len(self.overflow)
            self.overflow = torch.empty(0, self.activation_dim, device='cpu')
        
        # Reallocate full buffer
        new_buffer = torch.empty(self.buffer_size, self.activation_dim, device='cpu')
        new_buffer[:current_size] = self.activations
        self.activations = new_buffer
        
        # Fill remaining space with new activations
        while current_size < self.buffer_size:
            try:
                batch = next(self.data_loader)
            except StopIteration:
                self.is_exhausted = True
                break
            
            # Extract and sample activations from batch
            sampled_acts = self._extract_activations(batch)
            
            # Handle overflow
            remaining = self.buffer_size - current_size
            if len(sampled_acts) > remaining:
                self.overflow = sampled_acts[remaining:]
                sampled_acts = sampled_acts[:remaining]
            
            # Add to buffer
            self.activations[current_size:current_size + len(sampled_acts)] = sampled_acts
            current_size += len(sampled_acts)
        
        # Reset read mask
        self.read_mask = torch.zeros(current_size, dtype=torch.bool, device='cpu')
    
    def _extract_activations(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract activations from a data batch via adapter.
        
        Args:
            batch: Data batch (model-specific format)
        
        Returns:
            Sampled activations (n_samples, activation_dim)
        """
        # Run forward pass through adapter
        activations = self.adapter.generate_activations(batch, self.device_buffer)
        sampled = self._sample_tokens(activations, batch)
        return sampled.detach().cpu()
    
    def _sample_tokens(self, activations: torch.Tensor, batch: Dict) -> torch.Tensor:
        """
        Sample ctx_len tokens from each sequence, avoiding padding.
        
        Args:
            activations: (batch_size, seq_len, activation_dim)
            batch: Original batch 
        
        Returns:
            (batch_size * ctx_len, activation_dim)
        """
        samples = []
        mask = self.adapter.generate_activation_mask(batch)
        for seq_acts, mask_i in zip(activations, mask):
            # Use adapter-specific method to filter out padding tokens
            #valid_tokens = self.adapter.filter_padding_tokens(seq_acts, batch, i)
            valid_tokens = seq_acts[mask_i]
            n_valid = len(valid_tokens)
            
            if n_valid > self.ctx_len:
                indices = torch.randperm(n_valid)[:self.ctx_len]
                samples.append(valid_tokens[indices])
            else:
                samples.append(valid_tokens)
        
        return torch.cat(samples, dim=0) #TODO: may throw error if n_valid < ctx_len
    
    # ========================================================================
    # Multiprocessing Producer Workers
    # ========================================================================
    
    def producer_worker(self, worker_id: int, n_workers: int):
        """
        Producer worker process for multi-GPU buffer population.
        Each worker processes different partitions in parallel.
        
        Args:
            worker_id: ID of this worker (0 to n_workers-1)
            n_workers: Total number of workers
        """
        from .data_loader import create_multiprocess_data_loader
        
        # Setup worker-specific state
        self.device_buffer = self.device_buffer[worker_id]
        self.adapter.model.to(self.device_buffer)
        
        # Create worker-specific data loader (only for this worker's partitions)
        worker_data_loader = create_multiprocess_data_loader(
            cfg=self.cfg,
            worker_id=worker_id,
            n_workers=n_workers
        )
        
        log.info(f"Worker {worker_id}: Created data loader for partitions {worker_id}, {worker_id + n_workers}, ...")
        
        # Continuous production loop
        for batch in worker_data_loader:
            # Extract activations
            sampled_acts = self._extract_activations(batch)
            
            # Wait for buffer space and write
            with self.not_full:
                while self.unread_count.value > self.buffer_size * 0.8:
                    self.not_full.wait()
                
                # Find free slots and write
                free_indices = self.read_mask.nonzero(as_tuple=False).squeeze()
                n_acts = len(sampled_acts)
                
                # Handle case where we need more slots than available
                if len(free_indices) < n_acts:
                    n_acts = len(free_indices)
                    sampled_acts = sampled_acts[:n_acts]
                
                target_slots = free_indices[:n_acts]
                self.activations[target_slots] = sampled_acts
                self.read_mask[target_slots] = False
                self.unread_count.value += n_acts
                
                self.not_empty.notify()
        
        # Signal completion
        with self.lock:
            self.n_producers_done.value += 1
            self.not_empty.notify_all()
        
        log.info(f"Producer worker {worker_id} finished")
    
    # ========================================================================
    # Evaluation Batch Management
    # ========================================================================
    
    def _initialize_eval_cache_from_loader(self):
        """
        Cache first N batches from data loader for evaluation (cached strategy).
        Called during first buffer refresh to reuse already-initialized loader.
        """
        if self.eval_cache_initialized:
            return
        
        log.info(f"Caching {self.n_eval_batches} eval batches from data loader...")
        
        for i in range(self.n_eval_batches):
            try:
                batch = next(self.data_loader)
                
                # Store original batch (CPU)
                original_batch = {
                    k: v.clone().cpu() if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()
                }
                self.eval_batches_original.append(original_batch)
                
            except StopIteration:
                log.warning(
                    f"Data exhausted after {i}/{self.n_eval_batches} eval batches. "
                    f"Consider reducing n_eval_batches."
                )
                break
        
        self.eval_cache_initialized = True
        log.info(f"Cached {len(self.eval_batches_original)} eval batches")
    
    def get_eval_data(self) -> tuple:
        """
        Get evaluation data based on strategy.
        
        Returns:
            For 'cached': (list of original batches, list of activations)
            For 'on_demand': (list of original batches, list of activations)
                             - consumes next N batches from training data
        """
        if self.eval_strategy == 'cached':
            if not self.eval_cache_initialized:
                raise RuntimeError(
                    "Eval cache not initialized. This should happen during first buffer refresh."
                )
            return self.eval_batches_original
        
        elif self.eval_strategy == 'on_demand':
            # Consume next N batches from data loader
            original_batches = []
            
            n_batches = getattr(self, 'n_eval_batches_per_call', 1)
            
            for i in range(n_batches):
                try:
                    batch = next(self.data_loader)
                    
                    # Store original
                    original_batch = {
                        k: v.clone().cpu() if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()
                    }
                    original_batches.append(original_batch)
                    
                except StopIteration:
                    if i == 0:
                        log.warning("Data exhausted, no eval batches available")
                    break
            
            return original_batches
        
        else:
            raise ValueError(f"Unknown eval_strategy: {self.eval_strategy}")
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    @property
    def config(self) -> Dict[str, Any]:
        """Return buffer configuration"""
        return {
            'activation_dim': self.activation_dim,
            'buffer_size': self.buffer_size,
            'ctx_len': self.ctx_len,
            'n_ctxs': self.n_ctxs,
            'out_batch_size': self.out_batch_size,
            'refresh_batch_size': self.refresh_batch_size,
            'device_training': self.device_training,
            'device_buffer': self.device_buffer,
            'use_multiprocessing': self.use_multiprocessing
        }
    
    def close(self):
        """Cleanup resources"""
        self.adapter.clear_hooks()
        if self.use_multiprocessing:
            # TODO: Cleanup multiprocessing resources
            pass

# ============================================================================

def create_buffer(
    cfg: DictConfig,
    adapter,
    device_training: str,
    device_buffer: Optional[List[str]] = None
) -> tuple:
    """
    Factory function to create appropriate buffer and start producer processes if needed.
    
    Args:
        cfg: Hydra configuration
        adapter: ModelAdapter instance
        device_training: Device for SAE training
        device_buffer: Device(s) for buffer population
    
    Returns:
        (buffer, worker_processes)
    """
    from .data_loader import create_data_loader
    
    # Determine if multiprocessing needed
    use_multiprocessing = (
        len(device_buffer) > 1
    )
    
    # Create data loader (only for single-process mode)
    # In multiprocessing mode, each worker creates its own loader
    data_loader = None if use_multiprocessing else create_data_loader(cfg, adapter)
    
    # Create buffer
    buffer = ActivationBuffer(
        cfg=cfg,
        adapter=adapter,
        data_loader=data_loader,
        device_training=device_training,
        device_buffer=device_buffer,
        use_multiprocessing=use_multiprocessing
    )
    
    # Start producer workers if multiprocessing
    worker_processes = []
    if use_multiprocessing:
        n_workers = len(device_buffer)
        for worker_id in range(n_workers):
            process = mp.Process(
                target=buffer.producer_worker,
                args=(worker_id, n_workers)
            )
            process.start()
            worker_processes.append(process)
        
        log.info(f"Started {n_workers} producer workers")
    
    # Evaluation will be initialized based on strategy:
    # - cached: during first buffer refresh
    # - on_demand: when get_eval_data() is called
    log.info(f"Using '{buffer.eval_strategy}' evaluation strategy")
    
    return buffer, worker_processes
