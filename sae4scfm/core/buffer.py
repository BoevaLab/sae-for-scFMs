"""
Activation Buffer for SAE Training
"""

import gc
import torch
import torch.multiprocessing as mp
import logging
import sys
import time
import math
from pathlib import Path
from typing import Optional, Dict, Any, List
from omegaconf import DictConfig, OmegaConf
import queue

from sae4scfm.core.data_loader import DataLoader
from sae4scfm.core.utils import set_seed

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
        
        # Initialize buffer storage
        self._init_buffer_storage()
        
        # Evaluation configuration
        self.eval_strategy = cfg.buffer.get('eval_strategy', 'cached')
        self.n_eval_batches = cfg.buffer.get('n_eval_batches', 10)

        # Multiprocessing setup
        if use_multiprocessing:
            self.device_buffer = device_buffer
            self._init_multiprocessing()
        else:
            self.device_buffer = device_buffer[0]
            self._ensure_model_and_hook(self.device_buffer)
                       
        # Auto-calculate batches per eval call for on-demand strategy
        if self.eval_strategy == 'on_demand':
            total_steps = cfg.sae.get('steps', 1000)
            eval_every = cfg.experiment.get('eval_steps', 100)
            n_eval_calls = max(1, total_steps // eval_every)
            self.n_eval_batches_per_call = max(1, self.n_eval_batches // n_eval_calls)
            log.info(
                f"On-demand eval: {self.n_eval_batches_per_call} batches/call "
                f"(~{n_eval_calls} calls total)"
            )
        
        # Cached strategy storage
        self.eval_batches_original = []
        self.eval_cache_initialized = False
        

    def _ensure_model_and_hook(self, device: Optional[str] = None):
        """Load the model if needed and make sure exactly one save hook is registered."""
        if getattr(self.adapter, "model", None) is None:
            self.adapter.load_model()
        self.adapter.clear_hooks()
        self.adapter.setup_hook()
        if device is not None:
            self.adapter.model.to(device)

    def release_eval_model(self):
        """Free the eval-only model in the main process after multiprocessing eval."""
        if not self.use_multiprocessing:
            return
        self.adapter.clear_hooks()
        model = getattr(self.adapter, "model", None)
        if model is not None:
            model.to("cpu")
            self.adapter.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _init_buffer_storage(self):
        """Initialize activation storage tensors"""
        # Single-process backing storage (kept for backwards compatibility).
        # In multi-process mode, per-producer shard buffers are created in _init_multiprocessing.
        self.activations = torch.empty(self.buffer_size, self.activation_dim, device='cpu')
        self.read_mask = torch.ones(self.buffer_size, dtype=torch.bool, device='cpu')
        self.overflow = torch.empty(0, self.activation_dim, device='cpu')
        self.is_exhausted = False
    
    def _init_multiprocessing(self):
        """Initialize multiprocessing primitives"""
        self.n_producers = len(self.device_buffer)

        # Superbatch prefetch (consumer-side): how many SAE batches to fetch at once
        self.superbatch_batches = int(getattr(self.cfg.buffer, "superbatch_batches", 128))
        self.superbatch_size = int(self.superbatch_batches * self.out_batch_size)

        # Per-producer shard capacity (in samples).
        # Default is sized relative to the superbatch so each refill can draw without stalling:
        #   per_producer_take = ceil(superbatch_size / n_producers)
        #   shard_capacity ≈ 2x per_producer_take (headroom for producer/consumer overlap)
        per_producer_take = int(math.ceil(self.superbatch_size / self.n_producers))
        default_shard_capacity = int(per_producer_take * 2)
        shard_capacity_cfg = getattr(self.cfg.buffer, "shard_capacity", None)
        shard_capacity = int(shard_capacity_cfg) if shard_capacity_cfg is not None else default_shard_capacity
        shard_capacity = max(1, shard_capacity)
        self.shard_capacity = shard_capacity

        # Per-producer shard tensors + ring indices (shared across processes)
        self.shards: List[torch.Tensor] = []
        self.shard_write_idx = []
        self.shard_read_idx = []
        self.shard_filled = []  # number of valid unread samples
        self.shard_locks = []
        self.shard_not_empty = []
        self.shard_not_full = []
        self.producer_done = []

        shard_dtype_cfg = str(getattr(self.cfg.buffer, "shard_dtype", "float16")).lower()
        shard_dtype = torch.float16 if shard_dtype_cfg in ("fp16", "float16", "half") else torch.float32

        for _ in range(self.n_producers):
            shard = torch.empty(
                self.shard_capacity,
                self.activation_dim,
                device="cpu",
                dtype=shard_dtype,
            )
            shard.share_memory_()
            self.shards.append(shard)

            self.shard_write_idx.append(mp.Value("i", 0))
            self.shard_read_idx.append(mp.Value("i", 0))
            self.shard_filled.append(mp.Value("i", 0))

            lock = mp.Lock()
            self.shard_locks.append(lock)
            self.shard_not_empty.append(mp.Condition(lock))
            self.shard_not_full.append(mp.Condition(lock))

            self.producer_done.append(mp.Value("i", 0))

        # Consumer-side GPU cache (not shared)
        self._gpu_cache: Optional[torch.Tensor] = None
        self._gpu_cache_perm: Optional[torch.Tensor] = None
        self._gpu_cache_pos = 0

        # Consumer-side pinned CPU staging buffer (allocated lazily)
        self._cpu_pinned_stage: Optional[torch.Tensor] = None

        # Lightweight profiling/heartbeat for hang debugging
        self.profile_enabled = bool(getattr(self.cfg.buffer, "profile", True))
        self._prof_last_log_t = mp.Value("d", 0.0)
        self._producer_heartbeat = mp.Array("d", [0.0] * self.n_producers, lock=True)
        self._producer_batches = mp.Array("i", [0] * self.n_producers, lock=True)
        self._producer_wait_full_s = mp.Array("d", [0.0] * self.n_producers, lock=True)
        self._consumer_wait_empty_s = mp.Value("d", 0.0)
        
        # Shared eval batch storage
        self.eval_batch_queue = mp.Queue(maxsize=max(1, int(self.n_eval_batches)))
        self.eval_cached_count = mp.Value("i", 0)   
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
        """
        Multi-process mode: serve SAE batches from a prefetched, shuffled GPU superbatch.

        The consumer refills a GPU cache by reading contiguous chunks from each producer shard,
        concatenating on CPU pinned memory, transferring once to GPU (async), and shuffling on-GPU.
        """
        if (self._gpu_cache is None) or (self._gpu_cache_pos + self.out_batch_size > self._gpu_cache.shape[0]):
            log.info("_refill_gpu_cache")
            self._refill_gpu_cache()

        if self._gpu_cache is None or self._gpu_cache.shape[0] < self.out_batch_size:
            raise StopIteration

        start = self._gpu_cache_pos
        end = start + self.out_batch_size
        self._gpu_cache_pos = end

        if self._gpu_cache_perm is None:
            return self._gpu_cache[start:end]
        idx = self._gpu_cache_perm[start:end]
        return self._gpu_cache.index_select(0, idx)
    
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
    # Multiprocessing Consumer: Prefetch + Shuffle
    # ========================================================================

    def _maybe_log_profile(self, force: bool = False):
        if not self.profile_enabled:
            return
        now = time.time()
        with self._prof_last_log_t.get_lock():
            last = self._prof_last_log_t.value
            if (not force) and (now - last < 10.0):
                return
            self._prof_last_log_t.value = now

        # Snapshot shard fill levels and heartbeats
        fills = []
        read_idxs = []
        write_idxs = []
        dones = []
        for i in range(self.n_producers):
            fills.append(self.shard_filled[i].value)
            read_idxs.append(self.shard_read_idx[i].value)
            write_idxs.append(self.shard_write_idx[i].value)
            dones.append(self.producer_done[i].value)

        log.info(
            "Buffer(profile): fills=%s read=%s write=%s done=%s consumer_wait_empty_s=%.3f",
            fills,
            read_idxs,
            write_idxs,
            dones,
            float(self._consumer_wait_empty_s.value),
        )

    def _refill_gpu_cache(self):
        """
        Read `superbatch_size` samples across producer shards, shuffle on GPU, and reset cache position.
        """
        
        # Drop the previous GPU superbatch before allocating the next one.
        if self._gpu_cache is not None or self._gpu_cache_perm is not None:
            self._gpu_cache = None
            self._gpu_cache_perm = None
            self._gpu_cache_pos = 0
            gc.collect()
            if str(self.device_training).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_needed = self.superbatch_size
        per_prod = total_needed // self.n_producers
        remainder = total_needed - (per_prod * self.n_producers)

        # Allocate pinned staging buffer (consumer-only)
        if (self._cpu_pinned_stage is None) or (self._cpu_pinned_stage.shape[0] != total_needed):
            # float32 by default is wasteful; keep buffer in fp16 unless explicitly requested.
            stage_dtype = torch.float16 if getattr(self.cfg.buffer, "stage_dtype", "float16") == "float16" else torch.float32
            self._cpu_pinned_stage = torch.empty(
                total_needed,
                self.activation_dim,
                device="cpu",
                dtype=stage_dtype,
                pin_memory=True,
            )

        # Read from shards into pinned stage (contiguous segments per producer)
        offset = 0
        got_total = 0
        for pid in range(self.n_producers):
            need = per_prod + (1 if pid < remainder else 0)
            if need == 0:
                continue
            got = self._shard_read_into(pid, need, self._cpu_pinned_stage, offset)
            got_total += got
            offset += got  # keep the stage contiguous when a producer returns fewer rows

        if got_total < self.out_batch_size:
            # Not enough to form even one SAE batch
            self._gpu_cache = None
            self._gpu_cache_perm = None
            self._gpu_cache_pos = 0
            self._maybe_log_profile(force=True)
            return

        # Transfer to GPU once, then shuffle on-GPU
        device = self.device_training
        # Only transfer the valid prefix; remainder slots may be uninitialized if producers are done.
        cpu_view = self._cpu_pinned_stage[:got_total]
        self._gpu_cache = cpu_view.to(device=device, non_blocking=True)

        # Shuffle on GPU
        self._gpu_cache_perm = torch.randperm(got_total, device=device)
        self._gpu_cache_pos = 0
        self._maybe_log_profile()

    def _shard_read_into(self, pid: int, n: int, dst: torch.Tensor, dst_offset: int) -> int:
        """
        Read up to `n` samples contiguously from shard `pid` into `dst[dst_offset:dst_offset+n]`.
        Blocks until enough data is available OR the producer is done.
        Returns number of samples actually read.
        """
        start_wait = time.perf_counter()
        with self.shard_not_empty[pid]:
            while self.shard_filled[pid].value < n and self.producer_done[pid].value == 0:
                self.shard_not_empty[pid].wait(timeout=1.0)
                if self.profile_enabled:
                    with self._consumer_wait_empty_s.get_lock():
                        self._consumer_wait_empty_s.value += (time.perf_counter() - start_wait)
                    start_wait = time.perf_counter()

            available = self.shard_filled[pid].value
            take = min(n, available)
            if take <= 0:
                return 0

            read_idx = self.shard_read_idx[pid].value
            cap = self.shard_capacity

            first = min(take, cap - read_idx)
            second = take - first

            shard = self.shards[pid]
            # Copy into pinned dst; allow dtype conversion if stage_dtype != shard dtype.
            dst_slice = dst[dst_offset:dst_offset + take]
            dst_slice[:first].copy_(shard[read_idx:read_idx + first])
            if second > 0:
                dst_slice[first:first + second].copy_(shard[0:second])

            # Update ring pointers
            self.shard_read_idx[pid].value = (read_idx + take) % cap
            self.shard_filled[pid].value = available - take
            self.shard_not_full[pid].notify()
            return take
    
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
        
        return torch.cat(samples, dim=0)
    
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
        _configure_producer_process_logging(worker_id, self.cfg)
        log.info(f"producer_worker {worker_id} started")
        set_seed(self.cfg.seed + worker_id)        

        # Setup worker-specific state
        self.device_buffer = self.device_buffer[worker_id]
        # Ensure model is loaded in this process and hooks are registered.
        self._ensure_model_and_hook(self.device_buffer)
        # Create worker-specific data loader (only for this worker's partitions)
        worker_data_loader = create_multiprocess_data_loader(
            cfg=self.cfg,
            worker_id=worker_id,
            n_workers=n_workers,
            adapter=self.adapter
        )
        
        log.info(f"Worker {worker_id}: Created data loader for partitions {worker_id}, {worker_id + n_workers}, ...")
       
        if worker_id == 0 and self.eval_strategy == 'cached':
            self._initialize_eval_cache_from_loader(worker_data_loader, sink=self.eval_batch_queue.put)

        # Continuous production loop. Use next() rather than a for loop: iter() on the
        # partition loader restarts from its first partition, which would feed the cached
        # eval batches back into training.
        while True:
            try:
                batch = next(worker_data_loader)
            except StopIteration:
                break
            # Extract activations
            sampled_acts = self._extract_activations(batch)

            # Write into this producer's shard ring buffer
            self._shard_write_from(worker_id, sampled_acts)

            # Heartbeat / counters
            if self.profile_enabled:
                with self._producer_heartbeat.get_lock():
                    self._producer_heartbeat[worker_id] = time.time()
                with self._producer_batches.get_lock():
                    self._producer_batches[worker_id] += 1
        
        # Signal completion
        with self.shard_not_empty[worker_id]:
            self.producer_done[worker_id].value = 1
            self.shard_not_empty[worker_id].notify_all()
        
        log.info(f"Producer worker {worker_id} finished")

    def _shard_write_from(self, pid: int, acts_cpu: torch.Tensor):
        """
        Write a CPU tensor of shape (n, activation_dim) into shard `pid`.
        Blocks if shard is full.
        """
        if acts_cpu.numel() == 0:
            return
        # Ensure dtype matches shard (avoid implicit reallocs in shared memory)
        shard = self.shards[pid]
        if acts_cpu.dtype != shard.dtype:
            acts_cpu = acts_cpu.to(dtype=shard.dtype)

        n_total = int(acts_cpu.shape[0])
        written = 0
        cap = self.shard_capacity

        while written < n_total:
            start_wait = time.perf_counter()
            with self.shard_not_full[pid]:
                while self.shard_filled[pid].value >= cap and self.producer_done[pid].value == 0:
                    self.shard_not_full[pid].wait(timeout=1.0)
                    if self.profile_enabled:
                        with self._producer_wait_full_s.get_lock():
                            self._producer_wait_full_s[pid] += (time.perf_counter() - start_wait)
                        start_wait = time.perf_counter()

                filled = self.shard_filled[pid].value
                free = cap - filled
                if free <= 0:
                    continue

                write_idx = self.shard_write_idx[pid].value
                take = min(free, n_total - written)

                first = min(take, cap - write_idx)
                second = take - first

                shard[write_idx:write_idx + first].copy_(acts_cpu[written:written + first])
                if second > 0:
                    shard[0:second].copy_(acts_cpu[written + first:written + first + second])

                self.shard_write_idx[pid].value = (write_idx + take) % cap
                self.shard_filled[pid].value = filled + take
                written += take
                self.shard_not_empty[pid].notify()
    
    # ========================================================================
    # Evaluation Batch Management
    # ========================================================================
    
    def _initialize_eval_cache_from_loader(self, loader=None, sink=None):
        """
        Cache first N batches from data loader for evaluation (cached strategy).
        Called during first buffer refresh to reuse already-initialized loader.
        """
        if self.eval_cache_initialized:
            return

        if loader is None:
            loader = self.data_loader
        if sink is None:
            sink = self.eval_batches_original.append
        
        log.info(f"Caching {self.n_eval_batches} eval batches from data loader...")
       
        cached=0 
        for i in range(self.n_eval_batches):
            try:
                batch = next(loader)
                
                # Store original batch (CPU) with a deep clone so cached eval batches
                # can't be mutated by later collation/padding (e.g. Geneformer lists of tensors).
                def _clone_to_cpu(obj):
                    if isinstance(obj, torch.Tensor):
                        return obj.clone().cpu()
                    if isinstance(obj, dict):
                        return {kk: _clone_to_cpu(vv) for kk, vv in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        out = [_clone_to_cpu(vv) for vv in obj]
                        return out if isinstance(obj, list) else tuple(out)
                    return obj

                original_batch = _clone_to_cpu(batch)
                sink(original_batch)
                cached += 1
                if self.use_multiprocessing:
                    with self.eval_cached_count.get_lock():
                        self.eval_cached_count.value += 1
                
            except StopIteration:
                log.warning(
                    f"Data exhausted after {i}/{self.n_eval_batches} eval batches. "
                    f"Consider reducing n_eval_batches."
                )
                break
        
        self.eval_cache_initialized = True
        log.info(f"Cached {cached} eval batches")

    
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
                if self.use_multiprocessing:
                    target_batches = self.eval_cached_count.value
                    for _ in range(target_batches):
                        try:
                            self.eval_batches_original.append(self.eval_batch_queue.get(timeout=30))
                        except queue.Empty as exc:
                            raise RuntimeError("Timed out waiting for cached eval batches from worker 0.") from exc
                    self.eval_cache_initialized = True
                else:
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
    
    # Determine if multiprocessing needed (single source of truth: cfg.buffer.use_multiprocessing)
    use_multiprocessing = cfg.buffer.use_multiprocessing
    if use_multiprocessing and (device_buffer is None or len(device_buffer) <= 1):
        raise ValueError(
            "cfg.buffer.use_multiprocessing=true requires >1 CUDA devices so producers can run on separate GPUs. "
            f"Got device_buffer={device_buffer}."
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

# ============================================================================

def _configure_producer_process_logging(worker_id: int, cfg: DictConfig) -> None:
    """
    Spawned workers do not inherit the parent's logging handlers (e.g. after mp.spawn).
    Configure stderr (and Hydra run dir when available) so producer logs are visible.
    """
    if logging.root.handlers:
        return
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    out_dir = OmegaConf.select(cfg, "hydra.runtime.output_dir")
    if out_dir:
        try:
            log_path = Path(out_dir) / f"producer_worker_{worker_id}.log"
            handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
        except OSError:
            pass
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


# ============================================================================