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

// ---------------------------------------------------------------- parameter tables
// Generated at build time from the installed headers (gen_params.py -> hypre_params.inc): every
// HYPRE_BoomerAMGSet<Name> and Krylov Set<Name> taking plain values. Deck value syntax:
// "3", "0.25", list "1,2,3" (arrays: copied into hypre-allocated host memory, hypre frees them).
// Indexed setters (level / cycle part) are written "Name:k".
struct ParamSetter {
  bool indexed;
  HYPRE_Int (*set)(HYPRE_Solver, const std::string&, HYPRE_Int);
};
using ParamTable = std::map<std::string, ParamSetter>;

HYPRE_Int to_int(const std::string& v) {
  size_t pos;
  long x = std::stol(v, &pos);
  if (pos != v.size()) throw std::runtime_error("not an integer: '" + v + "'");
  return (HYPRE_Int)x;
}
HYPRE_Real to_real(const std::string& v) {
  size_t pos;
  double x = std::stod(v, &pos);
  if (pos != v.size()) throw std::runtime_error("not a number: '" + v + "'");
  return (HYPRE_Real)x;
}
std::vector<std::string> split_list(const std::string& v) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : v) {
    if (c == ',') { out.push_back(cur); cur.clear(); }
    else if (c != '[' && c != ']' && c != ' ') cur += c;
  }
  if (!cur.empty()) out.push_back(cur);
  if (out.empty()) throw std::runtime_error("empty list");
  return out;
}
size_t list_len(const std::string& v) { return split_list(v).size(); }
HYPRE_Int* int_array(const std::string& v) {
  auto e = split_list(v);
  HYPRE_Int* a = hypre_CTAlloc(HYPRE_Int, e.size(), HYPRE_MEMORY_HOST);
  for (size_t i = 0; i < e.size(); ++i) a[i] = to_int(e[i]);
  return a;
}
HYPRE_Real* real_array(const std::string& v) {
  auto e = split_list(v);
  HYPRE_Real* a = hypre_CTAlloc(HYPRE_Real, e.size(), HYPRE_MEMORY_HOST);
  for (size_t i = 0; i < e.size(); ++i) a[i] = to_real(e[i]);
  return a;
}

#include "hypre_params.inc"

// "Name" or "Name:k" -> table lookup + call. Errors name the parameter.
void set_param(const ParamTable& t, const char* what, HYPRE_Solver s, const std::string& key,
               const std::string& value) {
  std::string name = key;
  HYPRE_Int k = 0;
  auto colon = key.find(':');
  if (colon != std::string::npos) { name = key.substr(0, colon); k = to_int(key.substr(colon + 1)); }
  auto it = t.find(name);
  if (it == t.end()) throw std::runtime_error(std::string("unknown ") + what + " parameter '" + name + "'");
  if (it->second.indexed != (colon != std::string::npos))
    throw std::runtime_error(std::string(what) + " parameter '" + name + "' " +
                             (it->second.indexed ? "needs an index: '" + name + ":k'" : "takes no index"));
  try {
    check(it->second.set(s, value, k), name.c_str());
  } catch (std::invalid_argument&) {
    throw std::runtime_error(std::string(what) + " parameter '" + key + "': bad value '" + value + "'");
  }
}
void set_amg_param(HYPRE_Solver amg, const std::string& key, const std::string& value) {
  set_param(amg_table(), "BoomerAMG", amg, key, value);
}

// ---------------------------------------------------------------- Krylov wrappers
// Every ParCSR Krylov solver hypre provides: deck `solver:` = pcg | gmres | fgmres | lgmres |
// cogmres | bicgstab. X(deck name, hypre name).
#define SMA_KRYLOV(X) X(pcg, PCG) X(gmres, GMRES) X(fgmres, FlexGMRES) X(lgmres, LGMRES) X(cogmres, COGMRES) \
  X(bicgstab, BiCGSTAB)

struct Krylov {
  std::string name;
  HYPRE_Solver s = nullptr;
  void create() {
#define X(n, H) if (name == #n) { check(HYPRE_ParCSR##H##Create(MPI_COMM_WORLD, &s), #H "Create"); return; }
    SMA_KRYLOV(X)
#undef X
    throw std::runtime_error("unknown hypre solver '" + name + "' (pcg gmres fgmres lgmres cogmres bicgstab)");
  }
  void destroy() {
    if (!s) return;
#define X(n, H) if (name == #n) HYPRE_ParCSR##H##Destroy(s);
    SMA_KRYLOV(X)
#undef X
    s = nullptr;
  }
  void set(const std::string& k, const std::string& v) {
#define X(n, H) if (name == #n) { set_param(n##_table(), #n, s, k, v); return; }
    SMA_KRYLOV(X)
#undef X
  }
  void set_precond(HYPRE_Solver amg) {
#define X(n, H) if (name == #n) HYPRE_ParCSR##H##SetPrecond(s, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, amg);
    SMA_KRYLOV(X)
#undef X
  }
  void setup(HYPRE_ParCSRMatrix A, HYPRE_ParVector b, HYPRE_ParVector x) {
#define X(n, H) if (name == #n) check(HYPRE_ParCSR##H##Setup(s, A, b, x), #H "Setup");
    SMA_KRYLOV(X)
#undef X
  }
  // Solve errors (e.g. max iterations reached) are reported via convergence, not thrown.
  void solve(HYPRE_ParCSRMatrix A, HYPRE_ParVector b, HYPRE_ParVector x) {
#define X(n, H) if (name == #n) HYPRE_ParCSR##H##Solve(s, A, b, x);
    SMA_KRYLOV(X)
#undef X
    HYPRE_ClearAllErrors();
  }
  int iters() {
    HYPRE_Int it = -1;
#define X(n, H) if (name == #n) HYPRE_ParCSR##H##GetNumIterations(s, &it);
    SMA_KRYLOV(X)
#undef X
    return it;
  }
  bool converged() {
    HYPRE_Int c = 1;  // BiCGSTAB has no getter: validity relies on the true residual anyway
    if (name == "pcg") HYPRE_PCGGetConverged(s, &c);
    else if (name == "gmres") HYPRE_GMRESGetConverged(s, &c);
    else if (name == "fgmres") HYPRE_FlexGMRESGetConverged(s, &c);
    else if (name == "lgmres") HYPRE_LGMRESGetConverged(s, &c);
    else if (name == "cogmres") HYPRE_COGMRESGetConverged(s, &c);
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
    auto cfg = read_config_lines(rq.need("config"));
    std::string solver_name;
    for (auto& c : cfg) if (c.section == "solver") solver_name = c.value;
    if (solver_name.empty()) throw std::runtime_error("config has no 'solver' line");

    // Global (backend) settings apply to the whole library and cannot be read back: set them before
    // building solvers, and ask the dispatcher for a fresh driver afterwards (result key
    // driver_restart) so that they never leak into the next request.
    bool global_set = false;
    for (auto& c : cfg)
      if (c.section == "backend_param") {
        global_set = true;
        set_param(backend_table(), "backend", nullptr, c.key, c.value);
      }

    RunStats st;
    Measure m(rq);
    MemSampler mem;
    mem.start();
    st.mem_baseline = mem.baseline();
    std::unique_ptr<Krylov> k;
    HYPRE_Solver amg = nullptr;
    bool first = true;
    auto release = [&] {
      if (amg) { HYPRE_BoomerAMGDestroy(amg); amg = nullptr; }
      if (k) { k->destroy(); k.reset(); }
    };
    auto setup = [&] {
      k = std::make_unique<Krylov>(Krylov{solver_name});
      k->create();
      check(HYPRE_BoomerAMGCreate(&amg), "BoomerAMGCreate");
      HYPRE_BoomerAMGSetTol(amg, 0.0);  // one V-cycle per application (override with MaxIter/Tol)
      HYPRE_BoomerAMGSetMaxIter(amg, 1);
      for (auto& c : cfg) {
        if (c.section == "solver_param") k->set(c.key, c.value);
        if (c.section == "precond_param") set_amg_param(amg, c.key, c.value);
      }
      k->set_precond(amg);
      zero_x();
      cudaDeviceSynchronize();
      double t0 = now();
      k->setup(A, b, x);
      cudaDeviceSynchronize();
      return now() - t0;
    };
    auto solve = [&](int i) {
      zero_x();
      cudaDeviceSynchronize();
      double t0 = now();
      k->solve(A, b, x);
      cudaDeviceSynchronize();
      double tv = now() - t0;
      if (i >= 0) {
        st.iters_all.push_back(k->iters());
        st.lib_converged = k->converged();
        st.relres_true.push_back(true_relres(hA, x_host(), hb));
        if (first) { complexities(amg, st); first = false; }
      }
      return tv;
    };
    try {
      measure(m, st, setup, solve, release);
    } catch (std::exception& e) {
      st.error = e.what();
    }
    mem.stop();
    st.mem_peak = mem.peak();
    Json j;
    fill_result(j, rq, "hypre", HYPRE_RELEASE_VERSION, st, tol);
    j.str("solver", solver_name).boolean("driver_restart", global_set);
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
