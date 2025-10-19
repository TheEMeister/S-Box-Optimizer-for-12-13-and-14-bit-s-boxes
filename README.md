Project Status: Work in Progress -- Stable enough to use though

Overview
A high-performance optimizer for substitution boxes (S-boxes) used in cryptographic algorithms. S-boxes are fundamental to block ciphers like AES, they provide the nonlinearity that makes encryption resistant to attacks.

Why It's Interesting
Finding optimal S-boxes requires evaluating multiple conflicting cryptographic properties simultaneously over a massive search space (4096! permutations for 12-bit S-boxes).

Technical Highlights
- Hybrid multi-processing/threading system (up to 16 workers)
- Fast Walsh-Hadamard Transform for nonlinearity (O(n log n) vs standard O(n²))
- Simulated annealing with adaptive temperature decay
- Adaptive method selection that learns which techniques work best
- Parallel multi-search mode that runs independent optimizations and picks the winner
- GPU acceleration through CuPy for CUDA-enabled hardware -- Note: Not implemented right now. 
- Intelligent caching system to avoid redundant computations

Key Challenges
- Performance: Naive metric computation is prohibitively slow; solved with NumPy vectorization and Fast WHT 
- Daemon restrictions: Worked around Python's multiprocessing limitations using ThreadPools for nested parallelization
- Annealing tuning: Found 0.9995 decay rate balances exploration vs convergence

Results
Handles 12/13/14-bit S-boxes. Produces measurably better cryptographic properties than random permutations; parallel search consistently outperforms single-search approaches.

===========================================================================

Requirements: numpy
-
Optional: cupy (for GPU acceleration)

===========================================================================
## Usage
```bash
pip install numpy
python 12-13-14-bit_s-box_improver_V3.py
```

Then follow the interactive prompts to:
1. Load or generate an S-box
2. Select optimization metric (DDT, LAT, NL, etc.)
3. Choose number of iterations
4. Results saved to JSON
