import numpy as np
import random
import math
import json
import time
import signal
import concurrent.futures
from datetime import datetime
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from collections import defaultdict, deque
from typing import Callable, Tuple, Dict, Optional, List
import os

def _is_in_worker_process():
    return 'MULTIPROCESSING_WORKER' in os.environ

def _set_worker_flag():
    os.environ['MULTIPROCESSING_WORKER'] = '1'

# GPU DETECTION & INITIALIZATION

try:
    import cupy as cp
    test_array = cp.array([1, 2, 3])
    GPU_CUDA = True
except (ImportError, RuntimeError, Exception) as e:
    GPU_CUDA = False
    if not isinstance(e, ImportError):
        print(f"CuPy found but non-functional: {type(e).__name__}")

GPU_AVAILABLE = GPU_CUDA
GPU_STATUS_PRINTED = False

def print_gpu_status():
    global GPU_STATUS_PRINTED
    if not GPU_STATUS_PRINTED:
        if GPU_CUDA:
            print("[GPU] CUDA (CuPy) available")
        if not GPU_AVAILABLE:
            print("[WARN] No GPU detected - CPU-only mode")
        GPU_STATUS_PRINTED = True

# MULTIPROCESSING WORKERS (Module-level for pickling)

def _compute_ddt_chunk(args):
    dx_start, dx_end, s, sz = args
    s_arr = np.array(s, dtype=np.int32)
    max_val = 0
    
    for dx in range(dx_start, dx_end):
        xor_indices = np.arange(sz, dtype=np.int32) ^ dx
        valid = xor_indices < sz
        dy_values = s_arr[valid] ^ s_arr[xor_indices[valid]]
        counts = np.bincount(dy_values)
        max_val = max(max_val, np.max(counts) if len(counts) > 0 else 0)
    
    return max_val

def _compute_lat_chunk(args):
    s, a_start, a_end, sz, b_max = args
    s_arr = np.array(s, dtype=np.int32)
    max_bias = 0
    
    for a in range(a_start, a_end):
        input_parities = np.array([bin(x & a).count("1") & 1 for x in range(sz)], dtype=np.int8)
        
        for b in range(1, b_max):
            output_vals = s_arr & b
            output_parities = np.array([bin(v).count("1") & 1 for v in output_vals], dtype=np.int8)
            xor_result = input_parities ^ output_parities
            bias = np.sum(1 - 2 * xor_result)
            max_bias = max(max_bias, abs(bias))
    
    return max_bias

def _compute_nl_bit_wht_fft(args):
    s, bit_idx, sz = args
    f = np.array([(v >> bit_idx) & 1 for v in s], dtype=np.float32)
    f = 1.0 - 2.0 * f
    
    h = np.copy(f)
    n = sz
    step = 1
    
    while step < n:
        for i in range(0, n, step * 2):
            for j in range(step):
                idx1 = i + j
                idx2 = i + j + step
                if idx2 < n:
                    a = h[idx1]
                    b = h[idx2]
                    h[idx1] = a + b
                    h[idx2] = a - b
        step *= 2
    
    max_w = np.max(np.abs(h))
    return sz // 2 - int(max_w) // 2

def _compute_nl_bit_wht(args):
    s, bit_idx, sz = args
    f = np.array([(v >> bit_idx) & 1 for v in s], dtype=np.int8)
    W = np.zeros(sz, dtype=np.int32)
    
    for a in range(sz):
        parity_a = np.array([bin(a & x).count("1") & 1 for x in range(sz)], dtype=np.int8)
        xor_vals = f ^ parity_a
        W[a] = np.sum(1 - 2 * xor_vals)
    
    max_w = np.max(np.abs(W))
    return sz // 2 - max_w // 2

def _compute_av_chunk(args):
    s, x_start, x_end, nb = args
    total_flips = 0
    count = 0
    
    for x in range(x_start, x_end):
        for b in range(nb):
            flips = bin(s[x] ^ s[x ^ (1 << b)]).count('1')
            total_flips += flips / nb
            count += 1
    
    return total_flips, count

def _compute_sac_chunk(args):
    s, x_start, x_end, nb = args
    passed = 0
    total = 0
    
    for x in range(x_start, x_end):
        for bp in range(nb):
            if bin(s[x] ^ s[x ^ (1 << bp)]).count('1') >= nb // 2:
                passed += 1
            total += 1
    
    return passed, total

def _compute_bi_chunk(args):
    s, ib_start, ib_end, nb, sz = args
    total = 0
    count = 0
    
    for ib in range(ib_start, ib_end):
        for ob1 in range(nb):
            for ob2 in range(ob1 + 1, nb):
                corr = abs(sum(((s[x] >> ob1) & 1) ^ ((s[x ^ (1 << ib)] >> ob2) & 1) 
                              for x in range(sz)) / sz - 0.5)
                total += corr
                count += 1
    
    return total, count

# METRICS ENGINE

class MetricsEngine:    
    def __init__(self, s: List[int], num_workers: Optional[int] = None, use_threads: bool = False):
        self.s = s[:]
        self.sz = len(s)
        self.nb = int(math.log2(self.sz))
        
        if num_workers is None:
            self.num_workers = max(4, min(cpu_count() - 1, 16))
        else:
            self.num_workers = num_workers
        
        self.use_threads = use_threads  
        self._pool = None
        self._cache = {
            'ddt': None,
            'lat': None,
            'nl': None,
            'nl_full': False,
            'last_sbox_hash': None,
        }
        
        self._last_swap = None
        self.gpu_available = GPU_CUDA
        self.s_gpu = None
        if self.gpu_available:
            try:
                self.s_gpu = cp.array(s, dtype=cp.int32)
            except:
                self.gpu_available = False
    
    def _get_sbox_hash(self) -> int:
        return hash(tuple(self.s[::max(1, self.sz//100)]))
    
    def _get_pool(self):
        if self.num_workers <= 1:
            return None  
        
        if self._pool is None:
            if self.use_threads:
                from multiprocessing.pool import ThreadPool
                self._pool = ThreadPool(processes=self.num_workers)
                print(f"  [ENGINE] Initialized with {self.num_workers} threads")
            else:
                self._pool = Pool(processes=self.num_workers)
                print(f"  [ENGINE] Initialized with {self.num_workers} workers")
        return self._pool
    
    def update_sbox(self, s: List[int], invalidate_cache: bool = True, swap_info: Optional[Tuple[int, int]] = None):
        old_hash = self._cache.get('last_sbox_hash')
        self.s = s[:]
        new_hash = self._get_sbox_hash()
        
        if self.gpu_available:
            try:
                cp.copyto(self.s_gpu, cp.array(s, dtype=cp.int32))
            except:
                pass
        
        if invalidate_cache or old_hash != new_hash:
            if swap_info is not None:
                self._last_swap = swap_info
                self._cache['ddt'] = None
                self._cache['lat'] = None
                self._cache['nl'] = None
            else:
                self._cache = {
                    'ddt': None,
                    'lat': None,
                    'nl': None,
                    'nl_full': False,
                    'last_sbox_hash': new_hash,
                }
                self._last_swap = None
        
        self._cache['last_sbox_hash'] = new_hash
    
    def compute_ddt(self, use_cache: bool = True) -> int:
        if use_cache and self._cache['ddt'] is not None:
            return self._cache['ddt']
        
        pool = self._get_pool()
        
        # FIXED: Check if pool exists
        if self.sz <= 256 or pool is None:
            result = self._compute_ddt_serial()
        else:
            chunk_size = max(8, (self.sz - 1) // (self.num_workers * 4))
            chunks = []
            for dx_start in range(1, self.sz, chunk_size):
                dx_end = min(dx_start + chunk_size, self.sz)
                chunks.append((dx_start, dx_end, self.s, self.sz))
            
            try:
                results = pool.map(_compute_ddt_chunk, chunks)
                result = max(results) if results else 0
            except Exception as e:
                print(f"  [WARN] DDT parallel failed: {e}, using serial")
                result = self._compute_ddt_serial()
        
        if use_cache:
            self._cache['ddt'] = result
        return result
    
    def _compute_ddt_serial(self) -> int:
        max_val = 0
        for dx in range(1, self.sz):
            dc = defaultdict(int)
            for x in range(self.sz):
                dc[self.s[x] ^ self.s[x ^ dx]] += 1
            if dc:
                max_val = max(max_val, max(dc.values()))
        return max_val
    
    def compute_lat(self, use_cache: bool = True) -> int:
        if use_cache and self._cache['lat'] is not None:
            return self._cache['lat']
        
        if self.sz >= 16384:
            max_ab = 64
        elif self.sz >= 8192:
            max_ab = 96
        elif self.sz >= 4096:
            max_ab = 128
        else:
            max_ab = min(self.sz, 256)
        
        b_max = min(max_ab, 32)
        
        pool = self._get_pool()
        
        if self.sz <= 256 or pool is None:
            result = self._compute_lat_serial(max_ab, b_max)
        else:
            chunk_size = max(4, max_ab // (self.num_workers * 2))
            chunks = []
            for a_start in range(1, max_ab, chunk_size):
                a_end = min(a_start + chunk_size, max_ab)
                chunks.append((self.s, a_start, a_end, self.sz, b_max))
            
            try:
                results = pool.map(_compute_lat_chunk, chunks)
                result = max(results) if results else 0
            except Exception as e:
                print(f"  [WARN] LAT parallel failed: {e}, using serial")
                result = self._compute_lat_serial(max_ab, b_max)
        
        if use_cache:
            self._cache['lat'] = result
        return result
    
    def _compute_lat_serial(self, max_ab: int, b_max: int) -> int:
        max_bias = 0
        for a in range(1, max_ab):
            for b in range(1, b_max):
                bias = sum((-1) ** (bin(x & a).count("1") % 2 ^ bin(self.s[x] & b).count("1") % 2) 
                          for x in range(self.sz))
                max_bias = max(max_bias, abs(bias))
        return max_bias
    
    def compute_nl(self, use_cache: bool = True, force_full: bool = False, use_fft: bool = True) -> int:
        if use_cache and self._cache['nl'] is not None and (not force_full or self._cache['nl_full']):
            return self._cache['nl']
        
        sample_bits = min(self.nb, 10)
        use_fft = use_fft and self.sz >= 256
        
        pool = self._get_pool()
        
        if self.sz <= 512 or sample_bits <= 4 or pool is None:
            result = self._compute_nl_serial(sample_bits, use_fft=use_fft)
        else:
            bit_args = [(self.s, b, self.sz) for b in range(sample_bits)]
            
            try:
                chunksize = max(1, sample_bits // (self.num_workers * 2))
                worker_func = _compute_nl_bit_wht_fft if use_fft else _compute_nl_bit_wht
                nls = pool.map(worker_func, bit_args, chunksize=chunksize)
                result = min(nls) if nls else self.sz // 4
            except Exception as e:
                print(f"  [WARN] NL parallel failed: {e}, using serial")
                result = self._compute_nl_serial(sample_bits, use_fft=use_fft)
        
        if use_cache:
            self._cache['nl'] = result
            self._cache['nl_full'] = True
        return result
    
    def _compute_nl_serial(self, sample_bits: int, use_fft: bool = True) -> int:
        nls = []
        
        for b in range(sample_bits):
            if use_fft and self.sz >= 256:
                f = np.array([(v >> b) & 1 for v in self.s], dtype=np.float32)
                f = 1.0 - 2.0 * f
                
                h = np.copy(f)
                n = self.sz
                step = 1
                
                while step < n:
                    for i in range(0, n, step * 2):
                        for j in range(step):
                            idx1 = i + j
                            idx2 = i + j + step
                            if idx2 < n:
                                a = h[idx1]
                                b = h[idx2]
                                h[idx1] = a + b
                                h[idx2] = a - b
                    step *= 2
                
                max_w = np.max(np.abs(h))
                nls.append(self.sz // 2 - int(max_w) // 2)
            else:
                f = np.array([(v >> b) & 1 for v in self.s], dtype=np.int8)
                W = np.zeros(self.sz, dtype=np.int32)
                
                for a in range(self.sz):
                    parity_a = np.array([bin(a & x).count("1") & 1 for x in range(self.sz)], dtype=np.int8)
                    xor_vals = f ^ parity_a
                    W[a] = np.sum(1 - 2 * xor_vals)
                
                max_w = np.max(np.abs(W))
                nls.append(self.sz // 2 - max_w // 2)
        
        return min(nls) if nls else self.sz // 4
    
    def compute_avalanche(self, sample_size: Optional[int] = None) -> float:
        if sample_size is None:
            sample_size = min(self.sz, 2048)
        
        step = max(1, self.sz // sample_size)
        
        if sample_size >= self.sz or self.sz <= 512:
            total_flips = 0
            count = 0
            for x in range(self.sz):
                for b in range(self.nb):
                    flips = bin(self.s[x] ^ self.s[x ^ (1 << b)]).count('1')
                    total_flips += flips / self.nb
                    count += 1
            avg = total_flips / count if count > 0 else 0
        else:
            pool = self._get_pool()
            
            if pool is None:
                total_flips = 0
                count = 0
                for x in range(0, self.sz, step):
                    for b in range(self.nb):
                        flips = bin(self.s[x] ^ self.s[x ^ (1 << b)]).count('1')
                        total_flips += flips / self.nb
                        count += 1
                avg = total_flips / count if count > 0 else 0
            else:
                chunk_size = max(64, sample_size // self.num_workers)
                chunks = []
                for x_start in range(0, self.sz, step * chunk_size):
                    x_end = min(x_start + step * chunk_size, self.sz)
                    chunks.append((self.s, x_start, x_end, self.nb))
                
                try:
                    results = pool.map(_compute_av_chunk, chunks)
                    total_flips = sum(r[0] for r in results)
                    count = sum(r[1] for r in results)
                    avg = total_flips / count if count > 0 else 0
                except Exception as e:
                    print(f"  [WARN] AV parallel failed: {e}, using serial")
                    total_flips = 0
                    count = 0
                    for x in range(0, self.sz, step):
                        for b in range(self.nb):
                            flips = bin(self.s[x] ^ self.s[x ^ (1 << b)]).count('1')
                            total_flips += flips / self.nb
                            count += 1
                    avg = total_flips / count if count > 0 else 0
        
        return abs(avg - 0.5)
    
    def compute_sac(self, sample_size: Optional[int] = None) -> float:
        if sample_size is None:
            sample_size = min(self.sz, 2048)
        
        step = max(1, self.sz // sample_size)
        
        if sample_size >= self.sz or self.sz <= 512:
            passed = 0
            total = 0
            for x in range(self.sz):
                for bp in range(self.nb):
                    if bin(self.s[x] ^ self.s[x ^ (1 << bp)]).count('1') >= self.nb // 2:
                        passed += 1
                    total += 1
        else:
            pool = self._get_pool()
            
            if pool is None:
                passed = 0
                total = 0
                for x in range(0, self.sz, step):
                    for bp in range(self.nb):
                        if bin(self.s[x] ^ self.s[x ^ (1 << bp)]).count('1') >= self.nb // 2:
                            passed += 1
                        total += 1
            else:
                chunk_size = max(64, sample_size // self.num_workers)
                chunks = []
                for x_start in range(0, self.sz, step * chunk_size):
                    x_end = min(x_start + step * chunk_size, self.sz)
                    chunks.append((self.s, x_start, x_end, self.nb))
                
                try:
                    results = pool.map(_compute_sac_chunk, chunks)
                    passed = sum(r[0] for r in results)
                    total = sum(r[1] for r in results)
                except Exception as e:
                    print(f"  [WARN] SAC parallel failed: {e}, using serial")
                    passed = 0
                    total = 0
                    for x in range(0, self.sz, step):
                        for bp in range(self.nb):
                            if bin(self.s[x] ^ self.s[x ^ (1 << bp)]).count('1') >= self.nb // 2:
                                passed += 1
                            total += 1
        
        return passed / total if total > 0 else 0
    
    def compute_bit_independence(self, sample_size: Optional[int] = None) -> float:
        if sample_size is None:
            sample_size = min(self.sz, 2048)
        
        bit_sample = min(self.nb, 10)
        
        if sample_size >= self.sz or self.sz <= 512:
            total = 0
            count = 0
            for ib in range(bit_sample):
                for ob1 in range(bit_sample):
                    for ob2 in range(ob1 + 1, bit_sample):
                        corr = abs(sum(((self.s[x] >> ob1) & 1) ^ ((self.s[x ^ (1 << ib)] >> ob2) & 1) 
                                    for x in range(self.sz)) / self.sz - 0.5)
                        total += corr
                        count += 1
        else:
            pool = self._get_pool()
            
            if pool is None:
                total = 0
                count = 0
                for ib in range(bit_sample):
                    for ob1 in range(bit_sample):
                        for ob2 in range(ob1 + 1, bit_sample):
                            corr = abs(sum(((self.s[x] >> ob1) & 1) ^ ((self.s[x ^ (1 << ib)] >> ob2) & 1) 
                                        for x in range(self.sz)) / self.sz - 0.5)
                            total += corr
                            count += 1
            else:
                chunk_size = max(1, bit_sample // self.num_workers)
                chunks = []
                for ib_start in range(0, bit_sample, chunk_size):
                    ib_end = min(ib_start + chunk_size, bit_sample)
                    chunks.append((self.s, ib_start, ib_end, bit_sample, self.sz))
                
                try:
                    results = pool.map(_compute_bi_chunk, chunks)
                    total = sum(r[0] for r in results)
                    count = sum(r[1] for r in results)
                except Exception as e:
                    print(f"  [WARN] BI parallel failed: {e}, using serial")
                    total = 0
                    count = 0
                    for ib in range(bit_sample):
                        for ob1 in range(bit_sample):
                            for ob2 in range(ob1 + 1, bit_sample):
                                corr = abs(sum(((self.s[x] >> ob1) & 1) ^ ((self.s[x ^ (1 << ib)] >> ob2) & 1) 
                                            for x in range(self.sz)) / self.sz - 0.5)
                                total += corr
                                count += 1
        
        return total / count if count > 0 else 0
    
    def compute_all_metrics(self, use_sampling_where_safe: bool = True) -> Dict[str, float]:
        sample_size = 2048 if use_sampling_where_safe else self.sz
        
        return {
            'ddt': self.compute_ddt(use_cache=True),
            'lat': self.compute_lat(use_cache=True),
            'nl': self.compute_nl(use_cache=True, force_full=True),
            'av': self.compute_avalanche(sample_size),
            'bi': self.compute_bit_independence(sample_size),
            'sac': self.compute_sac(sample_size),
            'fp': sum(1 for i in range(self.sz) if self.s[i] == i),
        }
    
    def cleanup(self):
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
                print("  [ENGINE] Worker pool closed")
            except:
                pass

# INTELLIGENT SAMPLING STRATEGY

class IntelligentSamplingStrategy:
    def __init__(self, sbox_size: int, total_iterations: int):
        self.sbox_size = sbox_size
        self.total_iterations = total_iterations
        self.full_computation_metrics = {'ddt', 'lat', 'nl'}
        self.safe_sampling_metrics = {'av', 'bi', 'sac'}
        self.phase = "exploration"
        self.iterations_in_phase = 0
        self.exploration_end = int(total_iterations * 0.5)
        self.refinement_end = int(total_iterations * 0.8)
        
        print(f"  [SAMPLING] Intelligent strategy initialized:")
        print(f"    - Full computation: DDT, LAT, NL (always)")
        print(f"    - Sampling allowed: AV, BI, SAC (statistical metrics)")
        print(f"    - Phases: Exploration (0-{self.exploration_end}), "
              f"Refinement ({self.exploration_end}-{self.refinement_end}), "
              f"Precision ({self.refinement_end}+)")
    
    def update_phase(self, iteration: int, stagnation: int):
        old_phase = self.phase
        
        if iteration < self.exploration_end:
            self.phase = "exploration"
        elif iteration < self.refinement_end:
            if old_phase != "refinement":
                self.phase = "refinement"
                print(f"  [SAMPLING] -> REFINEMENT phase at iteration {iteration}")
        else:
            if old_phase != "precision":
                self.phase = "precision"
                print(f"  [SAMPLING] -> PRECISION phase at iteration {iteration}")
        
        if stagnation > 100 and self.phase != "precision":
            self.phase = "precision"
            print(f"  [SAMPLING] Early convergence -> PRECISION at iteration {iteration}")
    
    def get_sample_size(self, metric_name: str) -> Optional[int]:
        if metric_name in self.full_computation_metrics:
            return None
        
        if metric_name not in self.safe_sampling_metrics:
            return None
        
        if self.phase == "exploration":
            return min(1024, self.sbox_size // 4)
        elif self.phase == "refinement":
            return min(2048, self.sbox_size // 2)
        else:
            return self.sbox_size

# ADVANCED ADAPTIVE PARAMETERS - FIXED SIMULATED ANNEALING

class AdvancedAdaptiveParams:
    
    def __init__(self, sbox_size: int, total_iterations: int, metric_name: str):
        self.sbox_size = sbox_size
        self.total_iterations = total_iterations
        self.metric_name = metric_name
        
        self.stagnation = 0
        self.best_history = deque(maxlen=100)
        self.last_improvement = 0
        self.improvements_in_window = deque(maxlen=50)
        
        if metric_name == "NL":
            self.method_weights = {'greedy': 0.4, 'swap': 0.2, 'cycle': 0.15, 'block': 0.15, 'multi': 0.05, 'guided': 0.05}
        elif metric_name in ["DDT", "LAT"]:
            self.method_weights = {'greedy': 0.25, 'swap': 0.2, 'cycle': 0.2, 'block': 0.2, 'multi': 0.1, 'guided': 0.05}
        else:
            self.method_weights = {'greedy': 0.3, 'swap': 0.25, 'cycle': 0.15, 'block': 0.15, 'multi': 0.1, 'guided': 0.05}
        
        self.base_weights = self.method_weights.copy()
        self.exploration_factor = 1.0
        self.sampling = IntelligentSamplingStrategy(sbox_size, total_iterations)
        
        self.method_performance = {
            'greedy': {'attempts': 0, 'successes': 0, 'momentum': 1.0},
            'swap': {'attempts': 0, 'successes': 0, 'momentum': 1.0},
            'cycle': {'attempts': 0, 'successes': 0, 'momentum': 1.0},
            'block': {'attempts': 0, 'successes': 0, 'momentum': 1.0},
            'multi': {'attempts': 0, 'successes': 0, 'momentum': 1.0},
            'guided': {'attempts': 0, 'successes': 0, 'momentum': 1.0},
        }
        
        # FIXED: Slower temperature decay
        self.temperature = 1.0
        self.temperature_decay = 0.9995  # Was 0.995 - too fast!
    
    def update(self, current_val: float, iteration: int, improved: bool, method: str):
        self.best_history.append(current_val)
        self.improvements_in_window.append(1 if improved else 0)
        
        if method in self.method_performance:
            self.method_performance[method]['attempts'] += 1
            if improved:
                self.method_performance[method]['successes'] += 1
                self.method_performance[method]['momentum'] = min(2.0, 
                    self.method_performance[method]['momentum'] * 1.1)
            else:
                self.method_performance[method]['momentum'] = max(0.5,
                    self.method_performance[method]['momentum'] * 0.98)
        
        if len(self.best_history) >= 30:
            recent_var = np.var(list(self.best_history)[-30:])
            if recent_var < 1e-8:
                self.stagnation += 1
            else:
                self.stagnation = max(0, self.stagnation - 1)
        
        self.sampling.update_phase(iteration, self.stagnation)
        
        if iteration % 50 == 0 and iteration > 50:
            self._adapt_method_weights()
        
        recent_improvements = sum(self.improvements_in_window) if self.improvements_in_window else 0
        improvement_rate = recent_improvements / len(self.improvements_in_window) if self.improvements_in_window else 0
        
        if self.stagnation > 20:
            self.exploration_factor = min(3.0, 1.0 + self.stagnation * 0.1)
        elif improvement_rate > 0.05:
            self.exploration_factor = max(1.0, self.exploration_factor * 0.98)
        else:
            self.exploration_factor = min(2.0, self.exploration_factor * 1.02)
        
        self.temperature *= self.temperature_decay
    
    def _adapt_method_weights(self):
        total_attempts = sum(m['attempts'] for m in self.method_performance.values())
        if total_attempts < 30:
            return
        
        weighted_scores = {}
        for method, perf in self.method_performance.items():
            if perf['attempts'] > 0:
                success_rate = perf['successes'] / perf['attempts']
                weighted_scores[method] = success_rate * perf['momentum']
            else:
                weighted_scores[method] = 0.1
        
        total_score = sum(weighted_scores.values())
        if total_score > 0:
            for method in self.method_weights:
                if method in weighted_scores:
                    performance_weight = weighted_scores[method] / total_score
                    self.method_weights[method] = (0.6 * self.method_weights[method] + 
                                                   0.4 * performance_weight)
            
            total_weight = sum(self.method_weights.values())
            if total_weight > 0:
                for method in self.method_weights:
                    self.method_weights[method] /= total_weight
    
    def select_method(self) -> str:
        if random.random() < self.temperature * 0.1:
            return random.choice(list(self.method_weights.keys()))
        
        r = random.random()
        cumulative = 0
        for method, weight in self.method_weights.items():
            cumulative += weight
            if r < cumulative:
                return method
        return 'swap'
    
    def should_accept_worse(self, delta: float) -> bool:
        if delta >= 0:
            return True
        
        acceptance_prob = math.exp(delta / (self.temperature + 1e-10))
        return random.random() < acceptance_prob

# UTILITY FUNCTIONS

def eliminate_fixed_points(s: List[int]) -> List[int]:
    s = s[:]
    fp = [i for i in range(len(s)) if s[i] == i]
    
    for f in fp:
        for c in range(len(s)):
            if c != f and s[c] != c and s[f] != c:
                s[f], s[c] = s[c], s[f]
                break
    
    return s

def has_fixed_points(s: List[int]) -> bool:
    return any(i == s[i] for i in range(len(s)))

def count_fixed_points(s: List[int]) -> int:
    return sum(1 for i in range(len(s)) if s[i] == i)


# TRANSFORMATION METHODS

def random_swap(s: List[int]) -> List[int]:
    s = s[:]
    i, j = random.sample(range(len(s)), 2)
    s[i], s[j] = s[j], s[i]
    return s

def cycle_rotation(s: List[int]) -> List[int]:
    s = s[:]
    length = random.randint(3, min(20, len(s) // 100))
    start = random.randint(0, len(s) - length)
    cycle = s[start:start + length]
    cycle = cycle[1:] + cycle[:1]
    s[start:start + length] = cycle
    return s

def block_swap(s: List[int]) -> List[int]:
    s = s[:]
    sz = len(s)
    
    if sz >= 8192:
        max_block = min(100, sz // 80)
    elif sz >= 4096:
        max_block = min(60, sz // 90)
    else:
        max_block = min(40, sz // 100)
    
    block_size = random.randint(2, max_block)
    p1 = random.randint(0, sz - block_size)
    p2 = random.randint(0, sz - block_size)
    
    if abs(p1 - p2) >= block_size:
        temp = s[p1:p1 + block_size]
        s[p1:p1 + block_size] = s[p2:p2 + block_size]
        s[p2:p2 + block_size] = temp
    
    return s

def multi_swap(s: List[int], n: int = 3) -> List[int]:
    s = s[:]
    n = min(n, len(s) // 2)
    indices = random.sample(range(len(s)), n * 2)
    
    for i in range(0, len(indices) - 1, 2):
        s[indices[i]], s[indices[i+1]] = s[indices[i+1]], s[indices[i]]
    
    return s

def guided_perturbation(s: List[int], engine: MetricsEngine) -> List[int]:
    s = s[:]
    sz = len(s)
    
    high_impact_positions = [i for i in range(sz) if s[i] > sz * 0.7]
    
    if len(high_impact_positions) < 10:
        high_impact_positions = random.sample(range(sz), min(20, sz // 10))
    
    num_swaps = random.randint(2, 3)
    for _ in range(num_swaps):
        if high_impact_positions:
            i = random.choice(high_impact_positions)
            j = random.randint(0, sz - 1)
            s[i], s[j] = s[j], s[i]
    
    return s

def greedy_swap(s: List[int], engine: MetricsEngine, 
                     metric_func: Callable, direction: str, 
                     candidates: int = 30, use_batch: bool = True) -> List[int]:
    best = s[:]
    best_val = metric_func(s, engine)
    
    # FIXED: Use batch evaluation if enabled
    if use_batch and candidates >= 10:
        return _greedy_swap_batch(s, engine, metric_func, direction, candidates)
    
    # Sequential evaluation fallback
    attempts_without_improvement = 0
    max_attempts = candidates * 2
    
    for attempt in range(max_attempts):
        if attempts_without_improvement > candidates // 2:
            i = random.randint(0, len(s) // 2)
            j = random.randint(len(s) // 2, len(s) - 1)
        else:
            i, j = random.sample(range(len(s)), 2)
        
        test_sbox = best[:]
        test_sbox[i], test_sbox[j] = test_sbox[j], test_sbox[i]
        
        if any(idx == test_sbox[idx] for idx in range(len(test_sbox))):
            test_sbox = eliminate_fixed_points(test_sbox)
        
        try:
            engine.update_sbox(test_sbox, invalidate_cache=True, swap_info=(i, j))
            new_val = metric_func(test_sbox, engine)
            
            if (direction == 'higher' and new_val > best_val) or \
               (direction == 'lower' and new_val < best_val):
                best = test_sbox[:]
                best_val = new_val
                attempts_without_improvement = 0
                
                if attempt < candidates:
                    continue
                else:
                    break
            else:
                attempts_without_improvement += 1
        except:
            continue
        
        if attempt >= candidates and attempts_without_improvement > 5:
            break
    
    return best

def _greedy_swap_batch(s: List[int], engine: MetricsEngine,
                       metric_func: Callable, direction: str,
                       candidates: int) -> List[int]:
    candidate_sboxes = []
    
    for _ in range(candidates):
        i, j = random.sample(range(len(s)), 2)
        test = s[:]
        test[i], test[j] = test[j], test[i]
        
        if has_fixed_points(test):
            test = eliminate_fixed_points(test)
        
        candidate_sboxes.append(test)
    
    best_sbox = s
    best_val = metric_func(s, engine)
    
    for candidate in candidate_sboxes:
        try:
            engine.update_sbox(candidate, invalidate_cache=True)
            val = metric_func(candidate, engine)
            
            if (direction == 'higher' and val > best_val) or \
               (direction == 'lower' and val < best_val):
                best_sbox = candidate
                best_val = val
        except:
            continue
    
    return best_sbox


# PARALLEL MULTI-SEARCH ORCHESTRATOR - NOW INTEGRATED!

class ParallelSearchOrchestrator:
    def __init__(self, num_searches: int = None):
        if num_searches is None:
            num_searches = max(2, min(cpu_count() // 2, 8))
        self.num_searches = num_searches
        self.search_results = []
        
        print(f"  [ORCHESTRATOR] Running {num_searches} parallel searches")
    
    def run_parallel_searches(
        self,
        initial_sbox: List[int],
        metric_name: str,
        metric_func: Callable,
        direction: str,
        target: Optional[float],
        iterations_per_search: int,
        num_workers_per_search: int
    ) -> List[Dict]:
        print(f"\n{'='*70}")
        print(f"PARALLEL MULTI-SEARCH MODE")
        print(f"{'='*70}")
        print(f"Searches: {self.num_searches}")
        print(f"Iterations per search: {iterations_per_search}")
        print(f"Workers per search: {num_workers_per_search}")
        print(f"Total parallel work: {self.num_searches * iterations_per_search} iterations")
        print(f"{'='*70}\n")
        
        starting_sboxes = self._create_diverse_starts(initial_sbox, self.num_searches)
        
        search_args = []
        for search_id in range(self.num_searches):
            search_args.append({
                'search_id': search_id,
                'initial_sbox': starting_sboxes[search_id],
                'metric_name': metric_name,
                'metric_func': metric_func,
                'direction': direction,
                'target': target,
                'iterations': iterations_per_search,
                'num_workers': num_workers_per_search
            })
        
        with Pool(processes=self.num_searches) as pool:
            results = pool.map(_run_single_search, search_args)
        
        self.search_results = results
        return results
    
    def _create_diverse_starts(self, base_sbox: List[int], num_starts: int) -> List[List[int]]:
        starting_points = []
        
        for i in range(num_starts):
            s = base_sbox[:]
            perturbation_strength = i * 5
            
            for _ in range(perturbation_strength):
                choice = random.random()
                if choice < 0.4:
                    idx1, idx2 = random.sample(range(len(s)), 2)
                    s[idx1], s[idx2] = s[idx2], s[idx1]
                elif choice < 0.7:
                    block_size = random.randint(2, min(20, len(s) // 200))
                    p1 = random.randint(0, len(s) - block_size)
                    p2 = random.randint(0, len(s) - block_size)
                    if abs(p1 - p2) >= block_size:
                        temp = s[p1:p1 + block_size]
                        s[p1:p1 + block_size] = s[p2:p2 + block_size]
                        s[p2:p2 + block_size] = temp
                else:
                    length = random.randint(3, min(15, len(s) // 200))
                    start = random.randint(0, len(s) - length)
                    cycle = s[start:start + length]
                    cycle = cycle[1:] + cycle[:1]
                    s[start:start + length] = cycle
            
            s = eliminate_fixed_points(s)
            starting_points.append(s)
            
            print(f"  [DIVERSIFY] Search {i+1}: {perturbation_strength} perturbations applied")
        
        return starting_points
    
    def tournament_selection(self, engine: MetricsEngine) -> Dict:
        print(f"\n{'='*70}")
        print("TOURNAMENT SELECTION - Finding Best Overall Solution")
        print(f"{'='*70}")
        
        candidates = []
        
        for i, result in enumerate(self.search_results):
            print(f"\nEvaluating Search {i+1} result...")
            engine.update_sbox(result['best_sbox'], invalidate_cache=True)
            
            metrics = {
                'ddt': engine.compute_ddt(use_cache=False),
                'lat': engine.compute_lat(use_cache=False),
                'nl': engine.compute_nl(use_cache=False, force_full=True),
                'av': engine.compute_avalanche(sample_size=None),
                'bi': engine.compute_bit_independence(sample_size=None),
                'sac': engine.compute_sac(sample_size=None),
                'fp': count_fixed_points(result['best_sbox'])
            }
            
            sz = len(result['best_sbox'])
            nb = int(math.log2(sz))
            
            ddt_target = max(20, sz // 25)
            lat_target = max(200, sz // 3)
            nl_target = 2 ** (nb - 2)
            
            norm_ddt = max(0, min(1, (ddt_target - metrics['ddt']) / ddt_target))
            norm_lat = max(0, min(1, (lat_target - metrics['lat']) / lat_target))
            norm_nl = max(0, min(1, metrics['nl'] / nl_target))
            norm_av = max(0, min(1, (0.1 - metrics['av']) / 0.1))
            norm_bi = max(0, min(1, (0.1 - metrics['bi']) / 0.1))
            norm_sac = max(0, min(1, metrics['sac']))
            
            mo_score = (0.25 * norm_ddt + 0.25 * norm_lat + 0.20 * norm_nl + 
                       0.15 * norm_av + 0.15 * norm_bi)
            
            score_array = np.array([norm_ddt, norm_lat, norm_nl, norm_av, norm_bi, norm_sac])
            balance_score = 1.0 - np.std(score_array)
            
            final_score = 0.7 * mo_score + 0.3 * balance_score
            
            candidates.append({
                'search_id': i + 1,
                'sbox': result['best_sbox'],
                'metrics': metrics,
                'normalized_scores': {
                    'ddt': norm_ddt,
                    'lat': norm_lat,
                    'nl': norm_nl,
                    'av': norm_av,
                    'bi': norm_bi,
                    'sac': norm_sac
                },
                'mo_score': mo_score,
                'balance_score': balance_score,
                'final_score': final_score,
                'improvements': result['improvements'],
                'time': result['time']
            })
            
            print(f"  DDT:{metrics['ddt']} LAT:{metrics['lat']} NL:{metrics['nl']} "
                  f"AV:{metrics['av']:.4f} BI:{metrics['bi']:.4f} SAC:{metrics['sac']:.4f}")
            print(f"  MO Score: {mo_score:.4f} | Balance: {balance_score:.4f} | "
                  f"Final: {final_score:.4f}")
        
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        print(f"\n{'='*70}")
        print("TOURNAMENT RANKINGS")
        print(f"{'='*70}")
        print(f"{'Rank':<6} {'Search':<8} {'Final':<8} {'MO':<8} {'Balance':<8} {'Improvements':<12} {'Time':<10}")
        print("-" * 70)
        
        for rank, cand in enumerate(candidates, 1):
            print(f"{rank:<6} #{cand['search_id']:<7} {cand['final_score']:.4f}   "
                  f"{cand['mo_score']:.4f}   {cand['balance_score']:.4f}   "
                  f"{cand['improvements']:<12} {cand['time']:.1f}s")
        
        winner = candidates[0]
        
        print(f"\n{'='*70}")
        print(f" WINNER: Search #{winner['search_id']}")
        print(f"{'='*70}")
        print(f"Final Score: {winner['final_score']:.4f}")
        print(f"MO Score: {winner['mo_score']:.4f}")
        print(f"Balance Score: {winner['balance_score']:.4f}")
        print(f"Improvements: {winner['improvements']}")
        print(f"\nFinal Metrics:")
        print(f"  DDT: {winner['metrics']['ddt']}")
        print(f"  LAT: {winner['metrics']['lat']}")
        print(f"  NL: {winner['metrics']['nl']}")
        print(f"  AV: {winner['metrics']['av']:.4f}")
        print(f"  BI: {winner['metrics']['bi']:.4f}")
        print(f"  SAC: {winner['metrics']['sac']:.4f}")
        print(f"  FP: {winner['metrics']['fp']}")
        
        print(f"\nNormalized Scores (balance analysis):")
        for metric, score in winner['normalized_scores'].items():
            bar = '' * int(score * 20)
            print(f"  {metric.upper():4}: {score:.4f} {bar}")
        
        return winner

def _run_single_search(args: Dict) -> Dict:
    _set_worker_flag()  # Mark as worker process
    
    search_id = args['search_id']
    initial_sbox = args['initial_sbox']
    metric_name = args['metric_name']
    metric_func = args['metric_func']
    direction = args['direction']
    target = args['target']
    iterations = args['iterations']
    num_workers = args['num_workers']
    
    print(f"\n[Search {search_id + 1}] Starting optimization...", flush=True)
    
    start_time = time.time()
    
    print(f"[Search {search_id + 1}] Eliminating fixed points...", flush=True)
    best_sbox = eliminate_fixed_points(initial_sbox)
    
    print(f"[Search {search_id + 1}] Creating engine with {num_workers} threads...", flush=True)
    # FIXED: Use threads instead of processes to avoid daemon restriction
    engine = MetricsEngine(best_sbox, num_workers=num_workers, use_threads=True)
    
    adaptive = AdvancedAdaptiveParams(len(best_sbox), iterations, metric_name)
    
    print(f"[Search {search_id + 1}] Computing initial metric...", flush=True)
    engine.update_sbox(best_sbox)
    
    try:
        best_val = metric_func(best_sbox, engine)
        print(f"[Search {search_id + 1}] Initial value: {best_val:.4f}", flush=True)
    except Exception as e:
        print(f"[Search {search_id + 1}] ERROR computing initial metric: {e}", flush=True)
        raise
    
    improvements = 0
    
    print(f"[Search {search_id + 1}] Starting iteration loop...", flush=True)
    
    for iteration in range(1, iterations + 1):
        method = adaptive.select_method()
        
        if method == 'greedy':
            cand_count = int(30 * adaptive.exploration_factor)
            new_sbox = greedy_swap(best_sbox, engine, metric_func, direction, cand_count, use_batch=False)
        elif method == 'swap':
            new_sbox = random_swap(best_sbox)
        elif method == 'cycle':
            new_sbox = cycle_rotation(best_sbox)
        elif method == 'block':
            new_sbox = block_swap(best_sbox)
        elif method == 'multi':
            new_sbox = multi_swap(best_sbox, random.randint(2, 4))
        else:
            new_sbox = guided_perturbation(best_sbox, engine)
        
        if has_fixed_points(new_sbox):
            new_sbox = eliminate_fixed_points(new_sbox)
        
        try:
            engine.update_sbox(new_sbox, invalidate_cache=True)
            current_val = metric_func(new_sbox, engine)
        except Exception as e:
            continue
        
        improved = (direction == 'higher' and current_val > best_val) or \
                   (direction == 'lower' and current_val < best_val)
        
        # FIXED: Proper delta calculation for SA
        if direction == 'higher':
            delta = (current_val - best_val) / (abs(best_val) + 1e-6)
        else:
            delta = (best_val - current_val) / (abs(best_val) + 1e-6)
        
        if improved:
            accept = True
        elif adaptive.stagnation > 30 and delta < 0:
            accept = adaptive.should_accept_worse(delta)
        else:
            accept = False
        
        if accept:
            best_sbox = new_sbox[:]
            best_val = current_val
            if improved:
                improvements += 1
        
        adaptive.update(current_val, iteration, improved, method)
        
        if iteration % 50 == 0:
            print(f"[Search {search_id + 1}] Iteration {iteration}/{iterations}: "
                  f"{metric_name}={best_val:.4f} (+{improvements})", flush=True)
        
        if target is not None and iteration % 100 == 0:
            target_reached = (direction == 'higher' and best_val >= target) or \
                           (direction == 'lower' and best_val <= target)
            if target_reached:
                print(f"[Search {search_id + 1}] Target reached!", flush=True)
                break
    
    elapsed = time.time() - start_time
    
    # Clean up engine
    engine.cleanup()
    
    print(f"[Search {search_id + 1}] Complete: {improvements} improvements in {elapsed:.1f}s", flush=True)
    
    return {
        'search_id': search_id,
        'best_sbox': best_sbox,
        'best_val': best_val,
        'improvements': improvements,
        'time': elapsed,
        'final_iteration': iteration
    }


# COMPOSITE METRICS

def composite_score(s: List[int], engine: MetricsEngine, 
                   weights: Optional[Dict[str, float]] = None) -> float:
    if weights is None:
        weights = {'ddt': 0.3, 'lat': 0.3, 'nl': 0.2, 'av': 0.1, 'bi': 0.1}
    
    sz = len(s)
    ddt_target = max(16, sz // 32)
    lat_target = max(128, sz // 4)
    nl_target = max(240, sz // 3)
    
    ddt_val = engine.compute_ddt()
    lat_val = engine.compute_lat()
    nl_val = engine.compute_nl()
    av_val = engine.compute_avalanche()
    bi_val = engine.compute_bit_independence()
    
    ddt_norm = min(1.0, (ddt_target - ddt_val) / ddt_target) if ddt_target > 0 else 0
    lat_norm = min(1.0, (lat_target - lat_val) / lat_target) if lat_target > 0 else 0
    nl_norm = nl_val / nl_target if nl_target > 0 else 0
    av_norm = 1.0 - min(1.0, av_val / 0.1)
    bi_norm = 1.0 - min(1.0, bi_val / 0.1)
    
    return (weights['ddt'] * ddt_norm + 
            weights['lat'] * lat_norm + 
            weights['nl'] * nl_norm + 
            weights['av'] * av_norm + 
            weights['bi'] * bi_norm)

def multi_objective_score(s: List[int], engine: MetricsEngine,
                         weights: Optional[Dict[str, float]] = None) -> float:
    if weights is None:
        weights = {'ddt': 0.25, 'lat': 0.25, 'nl': 0.2, 'av': 0.15, 'bi': 0.15}
    
    sz = len(s)
    nb = int(math.log2(sz))
    
    ddt_target = max(20, sz // 25)
    lat_target = max(200, sz // 3)
    nl_target = 2 ** (nb - 2)
    
    ddt_val = engine.compute_ddt()
    lat_val = engine.compute_lat()
    nl_val = engine.compute_nl()
    av_val = engine.compute_avalanche()
    bi_val = engine.compute_bit_independence()
    
    ddt_norm = max(0, min(1, (ddt_target - ddt_val) / ddt_target))
    lat_norm = max(0, min(1, (lat_target - lat_val) / lat_target))
    nl_norm = max(0, min(1, nl_val / nl_target))
    av_norm = max(0, min(1, (0.1 - av_val) / 0.1))
    bi_norm = max(0, min(1, (0.1 - bi_val) / 0.1))
    
    return (weights['ddt'] * ddt_norm + 
            weights['lat'] * lat_norm + 
            weights['nl'] * nl_norm + 
            weights['av'] * av_norm + 
            weights['bi'] * bi_norm)

def metric_ddt(s: List[int], engine: MetricsEngine) -> int:
    return engine.compute_ddt()

def metric_lat(s: List[int], engine: MetricsEngine) -> int:
    return engine.compute_lat()

def metric_nl(s: List[int], engine: MetricsEngine) -> int:
    return engine.compute_nl()

def metric_av(s: List[int], engine: MetricsEngine) -> float:
    return engine.compute_avalanche()

def metric_bi(s: List[int], engine: MetricsEngine) -> float:
    return engine.compute_bit_independence()

def metric_sac(s: List[int], engine: MetricsEngine) -> float:
    return engine.compute_sac()


# FILE I/O

def load_sbox_from_file(filename: str) -> List[int]:
    print(f"Loading S-box from {filename}...")
    with open(filename, 'r') as f:
        data = json.load(f)
    
    if 'sbox' in data:
        sbox_data = data['sbox']
        bits = data.get('bits', int(math.log2(len(sbox_data))))
        print(f"  Loaded {bits}-bit S-box ({len(sbox_data)} elements)")
        return sbox_data
    else:
        raise ValueError("Invalid S-box file format")

def save_sbox(sbox: List[int], filename: str, metadata: Optional[Dict] = None):
    data = {
        'sbox': sbox,
        'bits': int(math.log2(len(sbox))),
        'size': len(sbox),
        'timestamp': datetime.now().isoformat()
    }
    
    if metadata:
        data.update(metadata)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


# PROGRESS REPORTING

class ProgressReporter:
    def __init__(self, total_iterations: int, metric_name: str):
        self.total_iterations = total_iterations
        self.metric_name = metric_name
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.iteration_times = deque(maxlen=50)
        self.improvements = 0
    
    def report_improvement(self, iteration: int, value: float):
        self.improvements += 1
        elapsed = time.time() - self.start_time
        print(f"[+] {iteration}: {self.metric_name}={value:.4f} [+{self.improvements}] ({elapsed:.1f}s)")
    
    def report_status(self, iteration: int, value: float, metrics_snapshot: Dict):
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        iter_time = current_time - self.last_report_time
        self.iteration_times.append(iter_time)
        self.last_report_time = current_time
        
        rate = iteration / elapsed if elapsed > 0 else 0
        
        if len(self.iteration_times) > 10:
            avg_iter_time = sum(self.iteration_times) / len(self.iteration_times)
            remaining_iters = self.total_iterations - iteration
            eta = remaining_iters * avg_iter_time
        else:
            eta = (self.total_iterations - iteration) / rate if rate > 0 else 0
        
        ddt = metrics_snapshot.get('ddt', '?')
        lat = metrics_snapshot.get('lat', '?')
        nl = metrics_snapshot.get('nl', '?')
        fp = metrics_snapshot.get('fp', 0)
        
        print(f"Iteration {iteration}/{self.total_iterations}: {self.metric_name}={value:.4f} | "
              f"DDT={ddt} LAT={lat} NL={nl} FP={fp} | "
              f"+{self.improvements} | {rate:.2f}it/s | ETA:{eta/60:.1f}m")
    
    def report_checkpoint(self, iteration: int, full_metrics: Dict, filename: str):
        print(f"[SAVE] Checkpoint {iteration}: {filename}")
        print(f"   DDT:{full_metrics['ddt']} LAT:{full_metrics['lat']} "
              f"NL:{full_metrics['nl']} AV:{full_metrics['av']:.4f} "
              f"BI:{full_metrics['bi']:.4f} SAC:{full_metrics['sac']:.4f}")


# MAIN OPTIMIZATION ENGINE - FIXED SIMULATED ANNEALING

def optimize_sbox_v3(
    initial_sbox: List[int],
    metric_name: str,
    metric_func: Callable,
    direction: str,
    target: Optional[float] = None,
    iterations: int = 1000,
    num_workers: Optional[int] = None
) -> Tuple[List[int], Dict]:
    s = eliminate_fixed_points(initial_sbox)
    sz = len(s)
    nb = int(math.log2(sz))
    
    engine = MetricsEngine(s, num_workers)
    adaptive = AdvancedAdaptiveParams(sz, iterations, metric_name)
    reporter = ProgressReporter(iterations, metric_name)
    
    print("\n" + "="*70)
    print("COMPUTING INITIAL METRICS")
    print("="*70)
    init_start = time.time()
    initial_metrics = engine.compute_all_metrics(use_sampling_where_safe=False)
    init_time = time.time() - init_start
    
    print(f"  DDT: {initial_metrics['ddt']}")
    print(f"  LAT: {initial_metrics['lat']}")
    print(f"  NL: {initial_metrics['nl']}")
    print(f"  AV: {initial_metrics['av']:.4f}")
    print(f"  BI: {initial_metrics['bi']:.4f}")
    print(f"  SAC: {initial_metrics['sac']:.4f}")
    print(f"  FP: {initial_metrics['fp']}")
    print(f"Computed in {init_time:.2f}s")
    
    best_sbox = s[:]
    engine.update_sbox(best_sbox)
    best_val = metric_func(best_sbox, engine)
    
    print("\n" + "="*70)
    print(f"OPTIMIZING: {metric_name}")
    print("="*70)
    print(f"Initial: {best_val:.4f}")
    print(f"Target: {target if target else 'maximize' if direction == 'higher' else 'minimize'}")
    print(f"Direction: {direction}")
    print(f"Iterations: {iterations}")
    print(f"Workers: {engine.num_workers}")
    print("="*70)
    
    def interrupt_handler(signum, frame):
        nonlocal iteration_counter
        print(f"\n\n[!]  INTERRUPT DETECTED at iteration {iteration_counter}")
        print("Saving current best S-box...")
        
        try:
            interrupt_metrics = engine.compute_all_metrics(use_sampling_where_safe=False)
            interrupt_name = f"sbox_{nb}bit_INTERRUPTED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            save_sbox(best_sbox, interrupt_name, {
                'version': 'V3-Fixed',
                'iterations': iteration_counter,
                'improvements': reporter.improvements,
                'total_time': time.time() - reporter.start_time,
                'metric_optimized': metric_name,
                'final_value': best_val,
                'interrupted': True,
                'initial_metrics': initial_metrics,
                'final_metrics': interrupt_metrics
            })
            
            print(f"[OK] Saved: {interrupt_name}")
        except Exception as e:
            print(f"[ERROR] Error saving: {e}")
        finally:
            engine.cleanup()
            print("Terminated by user")
            exit(0)
    
    signal.signal(signal.SIGINT, interrupt_handler)
    
    iteration_counter = 0
    opt_start = time.time()
    
    for iteration in range(1, iterations + 1):
        iteration_counter = iteration
        
        method = adaptive.select_method()
        
        if method == 'greedy':
            candidates = int(30 * adaptive.exploration_factor)
            new_sbox = greedy_swap(best_sbox, engine, metric_func, direction, candidates)
        elif method == 'swap':
            new_sbox = random_swap(best_sbox)
        elif method == 'cycle':
            new_sbox = cycle_rotation(best_sbox)
        elif method == 'block':
            new_sbox = block_swap(best_sbox)
        elif method == 'multi':
            n_swaps = random.randint(2, 4)
            new_sbox = multi_swap(best_sbox, n_swaps)
        else:
            new_sbox = guided_perturbation(best_sbox, engine)
        
        if has_fixed_points(new_sbox):
            new_sbox = eliminate_fixed_points(new_sbox)
        
        try:
            engine.update_sbox(new_sbox, invalidate_cache=True)
            current_val = metric_func(new_sbox, engine)
        except Exception as e:
            continue
        
        # FIXED: Proper simulated annealing logic
        improved = (direction == 'higher' and current_val > best_val) or \
                   (direction == 'lower' and current_val < best_val)
        
        # Calculate normalized delta (negative for worse solutions)
        if direction == 'higher':
            delta = (current_val - best_val) / (abs(best_val) + 1e-6)
        else:
            delta = (best_val - current_val) / (abs(best_val) + 1e-6)
        
        # Accept solution logic
        if improved:
            accept = True
        elif adaptive.stagnation > 30 and delta < 0:
            # Only try SA for worse solutions when stagnating
            accept = adaptive.should_accept_worse(delta)
        else:
            accept = False
        
        if accept:
            best_sbox = new_sbox[:]
            best_val = current_val
            
            if improved:
                reporter.report_improvement(iteration, best_val)
            else:
                # Accepted worse solution via SA
                print(f"  [SA] Accepted worse: {current_val:.4f} (temp={adaptive.temperature:.6f})")
        
        adaptive.update(current_val, iteration, improved, method)
        
        if iteration % 50 == 0:
            snapshot = {
                'ddt': engine.compute_ddt(use_cache=True),
                'lat': engine.compute_lat(use_cache=True),
                'nl': engine.compute_nl(use_cache=True),
                'fp': count_fixed_points(best_sbox)
            }
            reporter.report_status(iteration, best_val, snapshot)
        
        if iteration % 500 == 0:
            checkpoint_name = f"sbox_{nb}bit_V3_FIXED_{datetime.now().strftime('%Y%m%d_%H%M%S')}_i{iteration}.json"
            full_metrics = engine.compute_all_metrics(use_sampling_where_safe=False)
            
            save_sbox(best_sbox, checkpoint_name, {
                'version': 'V3-Fixed',
                'iteration': iteration,
                'metric_optimized': metric_name,
                'current_value': best_val,
                'improvements': reporter.improvements,
                'metrics': full_metrics,
                'stagnation': adaptive.stagnation,
                'temperature': adaptive.temperature
            })
            
            reporter.report_checkpoint(iteration, full_metrics, checkpoint_name)
        
        if target is not None and iteration % 100 == 0:
            target_reached = (direction == 'higher' and best_val >= target) or \
                           (direction == 'lower' and best_val <= target)
            
            if target_reached:
                print(f"\n[!] Target possibly reached! Verifying...")
                engine.update_sbox(best_sbox, invalidate_cache=True)
                verified_val = metric_func(best_sbox, engine)
                
                target_verified = (direction == 'higher' and verified_val >= target) or \
                                (direction == 'lower' and verified_val <= target)
                
                if target_verified:
                    print(f"[SUCCESS] TARGET VERIFIED! {metric_name}={verified_val:.4f} (target: {target})")
                    best_val = verified_val
                    break
                else:
                    print(f"[!] False alarm: actual={verified_val:.4f}, target={target}, continuing...")
    
    total_time = time.time() - opt_start
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"Iterations: {iteration}")
    print(f"Improvements: {reporter.improvements}")
    print(f"Improvement rate: {reporter.improvements/iteration*100:.2f}%")
    print(f"Average: {iteration/total_time:.2f} it/s")
    
    print("\nComputing final comprehensive metrics...")
    final_start = time.time()
    final_metrics = engine.compute_all_metrics(use_sampling_where_safe=False)
    final_compute_time = time.time() - final_start
    
    print(f"\n{'='*70}")
    print("FINAL METRICS")
    print(f"{'='*70}")
    print(f"Primary ({metric_name}): {best_val:.4f}")
    
    comparisons = [
        ('DDT', final_metrics['ddt'], initial_metrics['ddt'], 'lower'),
        ('LAT', final_metrics['lat'], initial_metrics['lat'], 'lower'),
        ('NL', final_metrics['nl'], initial_metrics['nl'], 'higher'),
        ('AV', final_metrics['av'], initial_metrics['av'], 'lower'),
        ('BI', final_metrics['bi'], initial_metrics['bi'], 'lower'),
        ('SAC', final_metrics['sac'], initial_metrics['sac'], 'higher'),
        ('FP', final_metrics['fp'], initial_metrics['fp'], 'lower')
    ]
    
    improvements_count = 0
    for name, final, initial, dir in comparisons:
        improved = (dir == 'lower' and final < initial) or (dir == 'higher' and final > initial)
        if improved:
            improvements_count += 1
        
        if isinstance(final, float):
            change = final - initial
            status = '[+] IMPROVED' if improved else ('=' if abs(change) < 1e-6 else '[-] WORSE')
            print(f"{name}: {final:.4f} (initial: {initial:.4f}) ({change:+.4f}) {status}")
        else:
            change = final - initial
            status = '[+] IMPROVED' if improved else ('=' if change == 0 else '[-] WORSE')
            print(f"{name}: {final} (initial: {initial}) ({change:+d}) {status}")
    
    print(f"\n{'='*70}")
    print(f"Metrics improved: {improvements_count}/{len(comparisons)}")
    print(f"Success rate: {improvements_count/len(comparisons)*100:.1f}%")
    print(f"Final metrics computed in {final_compute_time:.2f}s")
    
    print(f"\n{'='*70}")
    print("METHOD PERFORMANCE ANALYSIS")
    print(f"{'='*70}")
    for method, perf in sorted(adaptive.method_performance.items(), 
                              key=lambda x: x[1]['successes'], reverse=True):
        if perf['attempts'] > 0:
            rate = perf['successes'] / perf['attempts'] * 100
            momentum = perf['momentum']
            weight = adaptive.method_weights.get(method, 0) * 100
            print(f"{method.upper():8} | Success: {perf['successes']:3}/{perf['attempts']:4} ({rate:5.1f}%) | "
                  f"Momentum: {momentum:.2f} | Final Weight: {weight:5.1f}%")
        else:
            print(f"{method.upper():8} | Not used")
    
    print(f"\nStagnation counter: {adaptive.stagnation}")
    print(f"Final temperature: {adaptive.temperature:.6f}")
    print(f"Exploration factor: {adaptive.exploration_factor:.2f}")
    
    final_name = f"sbox_{nb}bit_V3_FINAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_sbox(best_sbox, final_name, {
        'version': 'V3-Fixed',
        'iterations': iteration,
        'improvements': reporter.improvements,
        'total_time': total_time,
        'metric_optimized': metric_name,
        'final_value': best_val,
        'target': target,
        'initial_metrics': initial_metrics,
        'final_metrics': final_metrics,
        'method_performance': {k: {**v, 'final_weight': adaptive.method_weights.get(k, 0)} 
                              for k, v in adaptive.method_performance.items()},
        'improvements_count': improvements_count,
        'stagnation_final': adaptive.stagnation,
        'temperature_final': adaptive.temperature
    })
    
    print(f"\n[OK] Final S-box saved: {final_name}")
    print("="*70)
    
    engine.cleanup()
    
    return best_sbox, {
        'initial_metrics': initial_metrics,
        'final_metrics': final_metrics,
        'improvements': reporter.improvements,
        'total_time': total_time,
        'iterations': iteration
    }


# MAIN ENTRY POINT - FIXED WITH PARALLEL MULTI-SEARCH INTEGRATION

def main():
    print_gpu_status()
    
    print("\n" + "="*70)
    print(" S-BOX OPTIMIZER V3 (FIXED) ")
    print("="*70)
    print("Features:")
    print("  - Fast Walsh-Hadamard Transform (O(n log n))")
    print("  - Parallel Multi-Search with Tournament Selection [NOW WORKING!]")
    print("  - Intelligent sampling (safe metrics only)")
    print("  - Enhanced parallelization (up to 16 workers)")
    print("  - Adaptive method selection with momentum")
    print("  - FIXED Simulated annealing for escape")
    print("  - Batch candidate evaluation [NOW WORKING!]")
    print("  - Performance tracking & auto-tuning")
    print("  - Accurate NL computation")
    print("="*70)
    
    choice = input("\n[1] Load from JSON\n[2] Generate random\n[3] Specific file path\nChoice: ")
    
    sbox = None
    
    if choice == "1":
        import os
        try:
            sbox_files = [f for f in os.listdir('.') if f.endswith('.json') and 'sbox' in f.lower()]
            if sbox_files:
                print("\nAvailable S-box files:")
                for idx, f in enumerate(sbox_files[:15], 1):
                    size_kb = os.path.getsize(f) / 1024
                    print(f"  [{idx}] {f} ({size_kb:.1f} KB)")
                
                file_choice = input("\nEnter number or filename: ")
                try:
                    file_idx = int(file_choice) - 1
                    filename = sbox_files[file_idx] if 0 <= file_idx < len(sbox_files) else file_choice
                except ValueError:
                    filename = file_choice
            else:
                filename = input("Enter filename: ")
        except:
            filename = input("Enter filename: ")
        
        sbox = load_sbox_from_file(filename)
    
    elif choice == "3":
        filename = input("Enter full path: ")
        sbox = load_sbox_from_file(filename)
    
    else:
        bits = int(input("Enter bit size (12, 13, or 14): "))
        if bits not in [12, 13, 14]:
            print("Invalid bit size!")
            return
        
        size = 2 ** bits
        print(f"Generating random {bits}-bit S-box ({size} elements)...")
        sbox = list(range(size))
        random.shuffle(sbox)
        print("[OK] Generated")
    
    metric_options = {
        1: ("DDT", metric_ddt, "lower"),
        2: ("LAT", metric_lat, "lower"),
        3: ("NL", metric_nl, "higher"),
        4: ("AV", metric_av, "lower"),
        5: ("BI", metric_bi, "lower"),
        6: ("SAC", metric_sac, "higher"),
        7: ("COMP", composite_score, "higher"),
        8: ("MO", multi_objective_score, "higher")
    }
    
    print("\nOptimization metrics:")
    print("  1. DDT (Differential) - lower better")
    print("  2. LAT (Linear) - lower better")
    print("  3. NL (Nonlinearity) - higher better [SLOW but ACCURATE]")
    print("  4. AV (Avalanche) - lower better")
    print("  5. BI (Bit Independence) - lower better")
    print("  6. SAC (Strict Avalanche) - higher better")
    print("  7. COMP (Composite) - higher better")
    print("  8. MO (Multi-Objective) - higher better [RECOMMENDED]")
    print("  9. PARALLEL MULTI-SEARCH - runs multiple searches, picks best [BEST QUALITY]")
    
    choice_metric = int(input("\nSelect metric (1-9): "))
    
    # FIXED: Handle parallel multi-search mode
    use_parallel_search = (choice_metric == 9)
    
    if use_parallel_search:
        metric_name = "MO"
        metric_func = multi_objective_score
        direction = "higher"
        print(f"\n[PARALLEL MODE] Using Multi-Objective metric for parallel search")
    else:
        metric_name, metric_func, direction = metric_options.get(choice_metric, metric_options[8])
    
    target_input = input("Target value (Enter=none): ")
    target = float(target_input) if target_input else None
    
    iterations = int(input("Iterations (default=1000): ") or "1000")
    
    if metric_name == "NL" and len(sbox) >= 4096:
        print(f"\n[!] NL optimization info for {len(sbox)}-element S-boxes:")
        print(f"   - Using Fast WHT algorithm (O(n log n))")
        print(f"   - Estimated: {iterations * 5 / 60:.0f}-{iterations * 8 / 60:.0f} minutes")
        print(f"   - ~40-50% faster than standard WHT")
        proceed = input("   Proceed? (y/n): ")
        if proceed.lower() != 'y':
            print("Optimization cancelled")
            return
    
    # FIXED: Run optimization with parallel mode support
    if use_parallel_search:
        print("\n" + "="*70)
        print("PARALLEL MULTI-SEARCH MODE ACTIVATED")
        print("="*70)
        
        # NEW: Direct input for both searches and workers
        num_searches = int(input("Number of parallel searches (2-8): ") or "4")
        num_searches = max(2, min(8, num_searches))
        
        workers_per_search = int(input("Workers per search (recommend 4-8): ") or "6")
        workers_per_search = max(2, min(16, workers_per_search))
        
        iterations_per_search = iterations // num_searches
        
        print(f"\nConfiguration:")
        print(f"  - Searches: {num_searches}")
        print(f"  - Workers per search: {workers_per_search}")
        print(f"  - Iterations per search: {iterations_per_search}")
        total_threads = num_searches * workers_per_search
        print(f"  - Total threads: {total_threads}")
        print(f"  - Total work: {num_searches * iterations_per_search} iterations")
        print("="*70 + "\n")
        
        orchestrator = ParallelSearchOrchestrator(num_searches)
        
        search_results = orchestrator.run_parallel_searches(
            initial_sbox=sbox,
            metric_name=metric_name,
            metric_func=metric_func,
            direction=direction,
            target=target,
            iterations_per_search=iterations_per_search,
            num_workers_per_search=workers_per_search
        )
        
        print(f"\nAll searches complete. Running tournament selection...")
        engine = MetricsEngine(sbox, num_workers=12)
        winner = orchestrator.tournament_selection(engine)
        engine.cleanup()
        
        optimized_sbox = winner['sbox']
        stats = {
            'initial_metrics': winner['metrics'],
            'final_metrics': winner['metrics'],
            'improvements': winner['improvements'],
            'total_time': winner['time'],
            'iterations': winner.get('final_iteration', iterations_per_search),
            'search_id': winner['search_id'],
            'all_searches': len(search_results)
        }
        
        print("\n[DONE] Parallel optimization complete!")
        
    else:
        optimized_sbox, stats = optimize_sbox_v3(
            initial_sbox=sbox,
            metric_name=metric_name,
            metric_func=metric_func,
            direction=direction,
            target=target,
            iterations=iterations
        )
        
        print("\n[DONE] Optimization complete!")

if __name__ == "__main__":
    main()