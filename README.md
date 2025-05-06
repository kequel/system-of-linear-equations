# Linear Equations: Iterative vs Direct Methods
## Note: Code variables, comments, and report (sprawozdanie.pdf) are in Polish.

Analysis of Jacobi, Gauss-Seidel, and LU methods for banded matrices with report made using LaTeX. 
Developed as part of a Numerical Methods course at Gdańsk University of Technology.

## Author
- Karolina Glaza [GitHub](https://github.com/kequel)

## Key Features  
- **Banded Matrix Generator**: Creates 5-diagonal matrices (size: 1293×1293) with configurable coefficients.  
- **Three Solvers**:  
  - Iterative: Jacobi and Gauss-Seidel.  
  - Direct: LU Decomposition.  
- **Convergence Analysis**: Tests for diagonal dominance (`a1=6` vs `a1=3`).  
- **Performance Benchmarking**: Time comparisons for `N = 100 → 3000`.  
- **Technical Report**: Full analysis in Polish (`sprawozdanie.pdf`).  

## Files  
| File | Description | 
|------|-------------| 
| [`main.py`](main.py) | Core implementation (matrix generation, solvers, tests) |
| [`sprawozdanie.pdf`](sprawozdanie.pdf) | Technical report with analysis |   
| [`sprawozdanie_source.tex`](sprawozdanie_source.tex) | LaTeX source for the report |  

## Results  
### Case 1: Diagonal Dominance (`a1=6`)  
- **Jacobi**: Converged in **37 iterations** (Residual: `1e-9`).  
- **Gauss-Seidel**: Converged in **23 iterations** (2x faster than Jacobi).  

### Case 2: Non-Diagonal Dominance (`a1=3`)  
- **Iterative Methods Failed**: Residual diverged exponentially.  
- **LU Decomposition**: Solved with residual `2.42e-13` (machine precision).

### Conclusion (from sprawozdanie.pdf)
1. Iterative Methods Work for diagonally dominant matrices (`a1=6`).

2. Avoid Iterative Methods for non-diagonally dominant cases (`a1=3`).

3. LU is Precise but Slow: Use only for small-to-medium matrices (N < 2000).

4. Gauss-Seidel > Jacobi: 2x faster convergence thanks to in-place updates.

## Installation  
1. Clone the repository:  
```bash  
git clone https://github.com/kequel/linear-equations-solver.git
 ```
Install dependencies:
 ```bash
pip install numpy matplotlib
 ```
Run all analyses:
 ```bash
python main.py
 ```
Generated charts are saved in the wykresy/ folder.
