// hypre BoomerAMG + hypre Krylov bench server (GPU, device memory, 1 MPI rank in v1).
// Config lines (written by sma/deck.py):
//   solver <pcg|gmres|fgmres|bicgstab>
//   solver_param  <Name> <value>   -> HYPRE_ParCSR<Solver>Set<Name>
//   precond_param <Name> <value>   -> HYPRE_BoomerAMGSet<Name>   ("Name:k" for (value, k) setters)
//   backend_param <Name> <value>   -> HYPRE_Set<Name> (global, e.g. SpGemmUseVendor)
#include <mpi.h>

#include <functional>

#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_ls.h"  // AMG hierarchy internals, for complexities
#include "sma_common.hpp"

using namespace sma;

namespace {

void check(HYPRE_Int ierr, const char* what) {
  if (ierr) {
    char msg[256];
    HYPRE_DescribeError(ierr, msg);
    HYPRE_ClearAllErrors();
    throw std::runtime_error(std::string(what) + ": " + msg);
  }
}

// ---------------------------------------------------------------- BoomerAMG setter table
using AmgSetter = std::function<void(HYPRE_Solver, const std::string&, int k)>;

#define AMG_INT(N) {#N, [](HYPRE_Solver s, const std::string& v, int) { check(HYPRE_BoomerAMGSet##N(s, std::stoi(v)), #N); }}
#define AMG_DBL(N) {#N, [](HYPRE_Solver s, const std::string& v, int) { check(HYPRE_BoomerAMGSet##N(s, std::stod(v)), #N); }}
#define AMG_INT_K(N) {#N, [](HYPRE_Solver s, const std::string& v, int k) { check(HYPRE_BoomerAMGSet##N(s, std::stoi(v), k), #N); }}

const std::map<std::string, AmgSetter>& amg_setters() {
  static const std::map<std::string, AmgSetter> t = {
      // hierarchy
      AMG_INT(MaxLevels), AMG_INT(MaxCoarseSize), AMG_INT(MinCoarseSize),
      // strength / coarsening
      AMG_DBL(StrongThreshold), AMG_DBL(StrongThresholdR), AMG_DBL(FilterThresholdR), AMG_DBL(MaxRowSum),
      AMG_INT(CoarsenType), AMG_INT(MeasureType), AMG_INT(AggNumLevels), AMG_INT(NumPaths),
      AMG_INT(CGCIts), AMG_DBL(NonGalerkinTol),
      // interpolation
      AMG_INT(InterpType), AMG_DBL(TruncFactor), AMG_INT(PMaxElmts), AMG_INT(AggInterpType),
      AMG_DBL(AggTruncFactor), AMG_DBL(AggP12TruncFactor), AMG_INT(AggPMaxElmts), AMG_INT(AggP12MaxElmts),
      // cycle / smoothing
      AMG_INT(CycleType), AMG_INT(FCycle), AMG_INT(NumSweeps), AMG_INT_K(CycleNumSweeps),
      AMG_INT(RelaxType), AMG_INT_K(CycleRelaxType), AMG_INT(RelaxOrder), AMG_DBL(RelaxWt), AMG_DBL(OuterWt),
      AMG_INT(ChebyOrder), AMG_DBL(ChebyFraction), AMG_INT(ChebyEigEst), AMG_INT(ChebyVariant),
      AMG_INT(ChebyScale), AMG_INT(SmoothType), AMG_INT(SmoothNumLevels), AMG_INT(SmoothNumSweeps),
      AMG_INT(ILUType), AMG_INT(ILULevel), AMG_INT(ILUMaxIter), AMG_DBL(ILUDroptol),
      AMG_INT(FSAIMaxSteps), AMG_INT(FSAIMaxStepSize), AMG_DBL(FSAIKapTolerance),
      // systems
      AMG_INT(NumFunctions), AMG_INT(Nodal), AMG_INT(NodalDiag),
      // Galerkin product / misc
      AMG_INT(KeepTranspose), AMG_INT(RAP2), AMG_INT(ModuleRAP2), AMG_DBL(ADropTol), AMG_INT(ADropType),
      AMG_INT(PrintLevel), AMG_INT(MaxIter), AMG_DBL(Tol),
  };
  return t;
}

// ---------------------------------------------------------------- Krylov wrappers
struct Krylov {
  std::string name;
  HYPRE_Solver s = nullptr;
  void create() {
    if (name == "pcg") check(HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &s), "PCGCreate");
    else if (name == "gmres") check(HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &s), "GMRESCreate");
    else if (name == "fgmres") check(HYPRE_ParCSRFlexGMRESCreate(MPI_COMM_WORLD, &s), "FlexGMRESCreate");
    else if (name == "bicgstab") check(HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &s), "BiCGSTABCreate");
    else throw std::runtime_error("unknown hypre solver " + name);
  }
  void destroy() {
    if (!s) return;
    if (name == "pcg") HYPRE_ParCSRPCGDestroy(s);
    else if (name == "gmres") HYPRE_ParCSRGMRESDestroy(s);
    else if (name == "fgmres") HYPRE_ParCSRFlexGMRESDestroy(s);
    else HYPRE_ParCSRBiCGSTABDestroy(s);
    s = nullptr;
  }
  void set(const std::string& k, const std::string& v) {
    auto I = [&] { return (HYPRE_Int)std::stoi(v); };
    auto D = [&] { return (HYPRE_Real)std::stod(v); };
#define KSET(SOLVER, N, CONV) if (k == #N) { check(HYPRE_ParCSR##SOLVER##Set##N(s, CONV()), #SOLVER "Set" #N); return; }
    if (name == "pcg") {
      KSET(PCG, Tol, D) KSET(PCG, AbsoluteTol, D) KSET(PCG, MaxIter, I) KSET(PCG, TwoNorm, I)
      KSET(PCG, RelChange, I) KSET(PCG, Logging, I) KSET(PCG, PrintLevel, I)
    } else if (name == "gmres") {
      KSET(GMRES, Tol, D) KSET(GMRES, AbsoluteTol, D) KSET(GMRES, MaxIter, I) KSET(GMRES, KDim, I)
      KSET(GMRES, MinIter, I) KSET(GMRES, Logging, I) KSET(GMRES, PrintLevel, I)
    } else if (name == "fgmres") {
      KSET(FlexGMRES, Tol, D) KSET(FlexGMRES, AbsoluteTol, D) KSET(FlexGMRES, MaxIter, I)
      KSET(FlexGMRES, KDim, I) KSET(FlexGMRES, MinIter, I) KSET(FlexGMRES, Logging, I) KSET(FlexGMRES, PrintLevel, I)
    } else {
      KSET(BiCGSTAB, Tol, D) KSET(BiCGSTAB, AbsoluteTol, D) KSET(BiCGSTAB, MaxIter, I)
      KSET(BiCGSTAB, MinIter, I) KSET(BiCGSTAB, Logging, I) KSET(BiCGSTAB, PrintLevel, I)
    }
#undef KSET
    throw std::runtime_error("unknown hypre " + name + " solver_param '" + k + "'");
  }
  void set_precond(HYPRE_Solver amg) {
    if (name == "pcg") HYPRE_ParCSRPCGSetPrecond(s, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, amg);
    else if (name == "gmres") HYPRE_ParCSRGMRESSetPrecond(s, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, amg);
    else if (name == "fgmres") HYPRE_ParCSRFlexGMRESSetPrecond(s, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, amg);
    else HYPRE_ParCSRBiCGSTABSetPrecond(s, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, amg);
  }
  void setup(HYPRE_ParCSRMatrix A, HYPRE_ParVector b, HYPRE_ParVector x) {
    if (name == "pcg") check(HYPRE_ParCSRPCGSetup(s, A, b, x), "PCGSetup");
    else if (name == "gmres") check(HYPRE_ParCSRGMRESSetup(s, A, b, x), "GMRESSetup");
    else if (name == "fgmres") check(HYPRE_ParCSRFlexGMRESSetup(s, A, b, x), "FlexGMRESSetup");
    else check(HYPRE_ParCSRBiCGSTABSetup(s, A, b, x), "BiCGSTABSetup");
  }
  // Solve errors (e.g. max iterations reached) are reported via convergence, not thrown.
  void solve(HYPRE_ParCSRMatrix A, HYPRE_ParVector b, HYPRE_ParVector x) {
    if (name == "pcg") HYPRE_ParCSRPCGSolve(s, A, b, x);
    else if (name == "gmres") HYPRE_ParCSRGMRESSolve(s, A, b, x);
    else if (name == "fgmres") HYPRE_ParCSRFlexGMRESSolve(s, A, b, x);
    else HYPRE_ParCSRBiCGSTABSolve(s, A, b, x);
    HYPRE_ClearAllErrors();
  }
  int iters() {
    HYPRE_Int n = -1;
    if (name == "pcg") HYPRE_ParCSRPCGGetNumIterations(s, &n);
    else if (name == "gmres") HYPRE_ParCSRGMRESGetNumIterations(s, &n);
    else if (name == "fgmres") HYPRE_ParCSRFlexGMRESGetNumIterations(s, &n);
    else HYPRE_ParCSRBiCGSTABGetNumIterations(s, &n);
    return n;
  }
  bool converged() {
    HYPRE_Int c = 0;
    if (name == "pcg") HYPRE_PCGGetConverged(s, &c);
    else if (name == "gmres") HYPRE_GMRESGetConverged(s, &c);
    else if (name == "fgmres") HYPRE_FlexGMRESGetConverged(s, &c);
    else return true;  // BiCGSTAB has no getter: rely on the true residual
    return c != 0;
  }
};

struct HypreDriver {
  HYPRE_IJMatrix ijA = nullptr;
  HYPRE_IJVector ijb = nullptr, ijx = nullptr;
  HYPRE_ParCSRMatrix A = nullptr;
  HYPRE_ParVector b = nullptr, x = nullptr;
  HYPRE_Int n = 0;

  void load(const Csr& h, const std::vector<double>& hb) {
    if (h.nrows > INT32_MAX || h.nnz > INT32_MAX) throw std::runtime_error("hypre built without int64");
    n = (HYPRE_Int)h.nrows;
    // Device copies of the CSR in IJ form (ncols per row, row ids, col ids, values).
    std::vector<HYPRE_Int> ncols(n), rows(n), cols(h.nnz);
    for (HYPRE_Int i = 0; i < n; ++i) { rows[i] = i; ncols[i] = (HYPRE_Int)(h.rowptr[i + 1] - h.rowptr[i]); }
    for (int64_t k = 0; k < h.nnz; ++k) cols[k] = (HYPRE_Int)h.colind[k];
    HYPRE_Int *d_ncols, *d_rows, *d_cols;
    HYPRE_Real *d_vals, *d_b, *d_zero;
    cudaMalloc(&d_ncols, n * sizeof(HYPRE_Int));
    cudaMalloc(&d_rows, n * sizeof(HYPRE_Int));
    cudaMalloc(&d_cols, h.nnz * sizeof(HYPRE_Int));
    cudaMalloc(&d_vals, h.nnz * sizeof(HYPRE_Real));
    cudaMalloc(&d_b, n * sizeof(HYPRE_Real));
    cudaMalloc(&d_zero, n * sizeof(HYPRE_Real));
    cudaMemcpy(d_ncols, ncols.data(), n * sizeof(HYPRE_Int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rows, rows.data(), n * sizeof(HYPRE_Int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, cols.data(), h.nnz * sizeof(HYPRE_Int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, h.values.data(), h.nnz * sizeof(HYPRE_Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, hb.data(), n * sizeof(HYPRE_Real), cudaMemcpyHostToDevice);
    cudaMemset(d_zero, 0, n * sizeof(HYPRE_Real));

    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, n - 1, 0, n - 1, &ijA);
    HYPRE_IJMatrixSetObjectType(ijA, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize_v2(ijA, HYPRE_MEMORY_DEVICE);
    check(HYPRE_IJMatrixSetValues(ijA, n, d_ncols, d_rows, d_cols, d_vals), "IJMatrixSetValues");
    check(HYPRE_IJMatrixAssemble(ijA), "IJMatrixAssemble");
    HYPRE_IJMatrixGetObject(ijA, (void**)&A);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, n - 1, &ijb);
    HYPRE_IJVectorSetObjectType(ijb, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize_v2(ijb, HYPRE_MEMORY_DEVICE);
    HYPRE_IJVectorSetValues(ijb, n, d_rows, d_b);
    HYPRE_IJVectorAssemble(ijb);
    HYPRE_IJVectorGetObject(ijb, (void**)&b);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, n - 1, &ijx);
    HYPRE_IJVectorSetObjectType(ijx, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize_v2(ijx, HYPRE_MEMORY_DEVICE);
    HYPRE_IJVectorSetValues(ijx, n, d_rows, d_zero);
    HYPRE_IJVectorAssemble(ijx);
    HYPRE_IJVectorGetObject(ijx, (void**)&x);
    cudaDeviceSynchronize();
    cudaFree(d_ncols); cudaFree(d_rows); cudaFree(d_cols); cudaFree(d_vals); cudaFree(d_b); cudaFree(d_zero);
  }

  void zero_x() { HYPRE_ParVectorSetConstantValues(x, 0.0); }

  std::vector<double> x_host() {
    std::vector<double> hx(n);
    std::vector<HYPRE_Int> idx(n);
    for (HYPRE_Int i = 0; i < n; ++i) idx[i] = i;
    // GetValues with host index/value arrays copies device data back.
    HYPRE_Int* d_idx;
    HYPRE_Real* d_val;
    cudaMalloc(&d_idx, n * sizeof(HYPRE_Int));
    cudaMalloc(&d_val, n * sizeof(HYPRE_Real));
    cudaMemcpy(d_idx, idx.data(), n * sizeof(HYPRE_Int), cudaMemcpyHostToDevice);
    HYPRE_IJVectorGetValues(ijx, n, d_idx, d_val);
    cudaMemcpy(hx.data(), d_val, n * sizeof(HYPRE_Real), cudaMemcpyDeviceToHost);
    cudaFree(d_idx); cudaFree(d_val);
    return hx;
  }

  static void complexities(HYPRE_Solver amg, RunStats& st) {
    auto* data = (hypre_ParAMGData*)amg;
    int L = hypre_ParAMGDataNumLevels(data);
    hypre_ParCSRMatrix** As = hypre_ParAMGDataAArray(data);
    double nnz0 = 0, nnz = 0, rows0 = 0, rows = 0;
    for (int l = 0; l < L; ++l) {
      hypre_ParCSRMatrixSetDNumNonzeros(As[l]);
      double z = hypre_ParCSRMatrixDNumNonzeros(As[l]);
      double r = (double)hypre_ParCSRMatrixGlobalNumRows(As[l]);
      if (l == 0) { nnz0 = z; rows0 = r; }
      nnz += z; rows += r;
    }
    st.levels = L;
    st.op_complexity = nnz0 > 0 ? nnz / nnz0 : NAN;
    st.grid_complexity = rows0 > 0 ? rows / rows0 : NAN;
  }

  Json run(const Request& rq, const Csr& hA, const std::vector<double>& hb) {
    double tol = std::stod(rq.need("tol"));
    int repeats = std::stoi(rq.get("repeats", "3"));
    int warmup = std::stoi(rq.get("warmup", "1"));
    auto cfg = read_config_lines(rq.need("config"));
    std::string solver_name;
    for (auto& c : cfg) if (c.section == "solver") solver_name = c.value;
    if (solver_name.empty()) throw std::runtime_error("config has no 'solver' line");

    // Global (backend) settings apply to the whole library: set them before building solvers.
    for (auto& c : cfg) {
      if (c.section != "backend_param") continue;
      int v = std::stoi(c.value);
      if (c.key == "SpGemmUseVendor") check(HYPRE_SetSpGemmUseVendor(v), "SetSpGemmUseVendor");
      else if (c.key == "SpMVUseVendor") check(HYPRE_SetSpMVUseVendor(v), "SetSpMVUseVendor");
      else if (c.key == "UseGpuRand") check(HYPRE_SetUseGpuRand(v), "SetUseGpuRand");
      else throw std::runtime_error("unknown hypre backend_param '" + c.key + "'");
    }

    RunStats st;
    MemSampler mem;
    mem.start();
    st.mem_baseline = mem.baseline();
    for (int r = -warmup; r < repeats; ++r) {
      Krylov k{solver_name};
      k.create();
      HYPRE_Solver amg;
      check(HYPRE_BoomerAMGCreate(&amg), "BoomerAMGCreate");
      HYPRE_BoomerAMGSetTol(amg, 0.0);  // one V-cycle per application (override with MaxIter/Tol)
      HYPRE_BoomerAMGSetMaxIter(amg, 1);
      for (auto& c : cfg) {
        if (c.section == "solver_param") k.set(c.key, c.value);
        if (c.section != "precond_param") continue;
        std::string name = c.key;
        int lvl = 0;
        auto colon = name.find(':');
        if (colon != std::string::npos) { lvl = std::stoi(name.substr(colon + 1)); name = name.substr(0, colon); }
        auto it = amg_setters().find(name);
        if (it == amg_setters().end()) throw std::runtime_error("unknown BoomerAMG parameter '" + name + "'");
        it->second(amg, c.value, lvl);
      }
      k.set_precond(amg);
      zero_x();
      cudaDeviceSynchronize();
      double t0 = now();
      k.setup(A, b, x);
      cudaDeviceSynchronize();
      double ts = now() - t0;
      t0 = now();
      k.solve(A, b, x);
      cudaDeviceSynchronize();
      double tv = now() - t0;
      if (r >= 0) {
        st.t_setup.push_back(ts);
        st.t_solve.push_back(tv);
        st.iters = k.iters();
        st.lib_converged = k.converged();
        st.relres_true.push_back(true_relres(hA, x_host(), hb));
        if (r == 0) complexities(amg, st);
      }
      HYPRE_BoomerAMGDestroy(amg);
      k.destroy();
    }
    mem.stop();
    st.mem_peak = mem.peak();
    Json j;
    fill_result(j, rq, "hypre", HYPRE_RELEASE_VERSION, st, tol);
    j.str("solver", solver_name);
    return j;
  }
};

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  HYPRE_Initialize();
  HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
  HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
  int rc;
  {
    HypreDriver d;
    rc = serve(d, argc, argv);
  }
  HYPRE_Finalize();
  MPI_Finalize();
  return rc;
}
