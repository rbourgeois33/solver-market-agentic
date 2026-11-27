// PETSc KSP + PCGAMG bench server (GPU matrices: aijkokkos / aijcusparse, 1 MPI rank in v1).
// Config: one line of PETSc options (written by sma/deck.py), inserted into the global options
// database for the duration of one request. mat_type/vec_type choose the GPU backend
// (default aijkokkos/kokkos); one matrix per mat_type is cached so the CSR is uploaded once per type.
#include <petscksp.h>

#include <cstdarg>

#include "sma_common.hpp"

using namespace sma;

namespace {

void ck(PetscErrorCode e, const char* what) {
  if (e) throw std::runtime_error(std::string("PETSc error in ") + what + " (code " + std::to_string(e) + ")");
}

struct PetscDriver {
  std::map<std::string, Mat> mats;  // by mat_type
  std::vector<PetscInt> coo_i, coo_j;
  std::vector<PetscScalar> vals, hb;
  PetscInt n = 0;

  void load(const Csr& h, const std::vector<double>& b) {
    n = (PetscInt)h.nrows;
    coo_i.resize(h.nnz);
    coo_j.resize(h.nnz);
    vals.assign(h.values.begin(), h.values.end());
    for (int64_t i = 0; i < h.nrows; ++i)
      for (int64_t k = h.rowptr[i]; k < h.rowptr[i + 1]; ++k) { coo_i[k] = (PetscInt)i; coo_j[k] = (PetscInt)h.colind[k]; }
    hb.assign(b.begin(), b.end());
    get_mat("aijkokkos");  // default backend uploaded at load time
  }

  Mat get_mat(const std::string& type) {
    auto it = mats.find(type);
    if (it != mats.end()) return it->second;
    Mat A;
    ck(MatCreate(PETSC_COMM_WORLD, &A), "MatCreate");
    ck(MatSetSizes(A, n, n, PETSC_DETERMINE, PETSC_DETERMINE), "MatSetSizes");
    ck(MatSetType(A, type.c_str()), "MatSetType");
    ck(MatSetPreallocationCOO(A, (PetscCount)coo_i.size(), coo_i.data(), coo_j.data()), "MatSetPreallocationCOO");
    ck(MatSetValuesCOO(A, vals.data(), INSERT_VALUES), "MatSetValuesCOO");
    ck(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
    ck(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");
    mats[type] = A;
    return A;
  }

  static std::string opt_string(const char* name, const char* def) {
    char buf[256];
    PetscBool set = PETSC_FALSE;
    PetscOptionsGetString(nullptr, nullptr, name, buf, sizeof buf, &set);
    return set ? std::string(buf) : std::string(def);
  }

  Json run(const Request& rq, const Csr& hA, const std::vector<double>& hbv) {
    double tol = std::stod(rq.need("tol"));
    std::ifstream in(rq.need("config"));
    std::string opts((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    // PetscOptionsInsertString splits on spaces only: a newline would end up inside the last value.
    for (char& c : opts)
      if (c == '\n' || c == '\r' || c == '\t') c = ' ';

    ck(PetscOptionsClear(nullptr), "PetscOptionsClear");
    ck(PetscOptionsInsertString(nullptr, opts.c_str()), "PetscOptionsInsertString");
    std::string mat_type = opt_string("-mat_type", "aijkokkos");
    std::string vec_type = opt_string("-vec_type", "");
    Mat A = get_mat(mat_type);
    Vec x, b;
    ck(MatCreateVecs(A, &x, &b), "MatCreateVecs");
    if (!vec_type.empty()) { ck(VecSetType(x, vec_type.c_str()), "VecSetType"); ck(VecSetType(b, vec_type.c_str()), "VecSetType"); }
    {
      PetscScalar* p;
      ck(VecGetArrayWrite(b, &p), "VecGetArrayWrite");
      std::copy(hb.begin(), hb.end(), p);
      ck(VecRestoreArrayWrite(b, &p), "VecRestoreArrayWrite");
    }

    RunStats st;
    Measure m(rq);
    std::string ksp_type, pc_type, reason_str;
    MemSampler mem;
    mem.start();
    st.mem_baseline = mem.baseline();
    KSP ksp = nullptr;
    PC pc = nullptr;
    bool first = true;
    auto release = [&] {
      if (ksp) KSPDestroy(&ksp);
      ksp = nullptr;
      pc = nullptr;
    };
    auto setup = [&] {
      ck(KSPCreate(PETSC_COMM_WORLD, &ksp), "KSPCreate");
      ck(KSPSetOperators(ksp, A, A), "KSPSetOperators");
      ck(KSPSetTolerances(ksp, tol, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT), "KSPSetTolerances");
      ck(KSPSetFromOptions(ksp), "KSPSetFromOptions");  // options override the default rtol
      ck(VecSet(x, 0.0), "VecSet");
      cudaDeviceSynchronize();
      double t0 = now();
      ck(KSPSetUp(ksp), "KSPSetUp");
      ck(KSPGetPC(ksp, &pc), "KSPGetPC");
      ck(PCSetUp(pc), "PCSetUp");
      cudaDeviceSynchronize();
      return now() - t0;
    };
    auto solve = [&](int i) {
      ck(VecSet(x, 0.0), "VecSet");
      cudaDeviceSynchronize();
      double t0 = now();
      ck(KSPSolve(ksp, b, x), "KSPSolve");
      cudaDeviceSynchronize();
      double tv = now() - t0;
      if (i < 0) return tv;
      PetscInt it;
      KSPGetIterationNumber(ksp, &it);
      st.iters_all.push_back((int)it);
      KSPConvergedReason reason;
      KSPGetConvergedReason(ksp, &reason);
      st.lib_converged = reason > 0;
      reason_str = KSPConvergedReasons[reason];
      const PetscScalar* px;
      ck(VecGetArrayRead(x, &px), "VecGetArrayRead");
      std::vector<double> hx(px, px + n);
      ck(VecRestoreArrayRead(x, &px), "VecRestoreArrayRead");
      st.relres_true.push_back(true_relres(hA, hx, hbv));
      if (first) {
        first = false;
        KSPType kt;
        PCType pt;
        KSPGetType(ksp, &kt);
        PCGetType(pc, &pt);
        ksp_type = kt;
        pc_type = pt;
        PetscBool is_mg = PETSC_FALSE;
        PetscObjectTypeCompareAny((PetscObject)pc, &is_mg, PCGAMG, PCMG, "");
        if (is_mg) {
          PetscInt L;
          PCMGGetLevels(pc, &L);
          st.levels = (int)L;
          PetscReal gc, oc;
          if (PCMGGetGridComplexity(pc, &gc, &oc) == 0) { st.grid_complexity = gc; st.op_complexity = oc; }
          for (PetscInt l = L - 1; l >= 0; --l) {  // PCMG numbers the coarsest level 0: report finest first
            KSP lk;
            Mat Al;
            PetscInt rows;
            MatInfo info;
            if (PCMGGetSmoother(pc, l, &lk) || KSPGetOperators(lk, &Al, nullptr) || MatGetSize(Al, &rows, nullptr) ||
                MatGetInfo(Al, MAT_GLOBAL_SUM, &info))
              break;
            st.level_rows.push_back((double)rows);
            st.level_nnz.push_back(info.nz_used);
          }
        }
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

    // Options given but never queried: typos or options that do not apply (reported to the agent).
    PetscInt nleft;
    char** names;
    char** values;
    std::string unused;
    if (PetscOptionsLeftGet(nullptr, &nleft, &names, &values) == 0) {
      for (PetscInt i = 0; i < nleft; ++i) unused += (unused.empty() ? "-" : " -") + std::string(names[i]);
      PetscOptionsLeftRestore(nullptr, &nleft, &names, &values);
    }
    VecDestroy(&x);
    VecDestroy(&b);
    ck(PetscOptionsClear(nullptr), "PetscOptionsClear");

    char ver[256];
    PetscGetVersion(ver, sizeof ver);
    Json j;
    fill_result(j, rq, "petsc", ver, st, tol);
    j.str("ksp_type", ksp_type).str("pc_type", pc_type).str("mat_type", mat_type);
    j.str("converged_reason", reason_str).str("unused_options", unused);
    return j;
  }
};

// ------------------------------------------------------------------ parameter catalogue
// --catalog DIR: for each scenario (every registered KSP type; every registered PC type under
// richardson; GAMG variants with level smoothers), run KSPSetFromOptions + KSPSetUp with -help on a
// small 2D Laplacian and capture PETSc's help text into DIR/petsc_help.txt. Each scenario has its
// own options prefix "sN_" (PETSc prints a help block only once per title + prefix).
// sma/catalog.py parses the text into petsc.json.
std::string g_help;
PetscErrorCode capture_help(MPI_Comm, const char fmt[], ...) {
  va_list ap;
  va_start(ap, fmt);
  char buf[16384];
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  g_help += buf;
  return PETSC_SUCCESS;
}

std::vector<std::string> list_names(PetscFunctionList fl) {
  const char** names = nullptr;
  int n = 0;
  PetscFunctionListGet(fl, &names, &n);
  std::vector<std::string> out(names, names + n);
  PetscFree(names);
  return out;
}

int catalog(const std::string& dir) {
  ck(KSPInitializePackage(), "KSPInitializePackage");
  ck(PCInitializePackage(), "PCInitializePackage");
  const PetscInt m = 32, N = m * m;  // 2D 5-point Laplacian, enough rows for >= 2 GAMG levels
  std::vector<PetscInt> I, J;
  std::vector<PetscScalar> V;
  for (PetscInt r = 0; r < N; ++r) {
    PetscInt i = r / m, j = r % m;
    auto add = [&](PetscInt c, double v) { I.push_back(r); J.push_back(c); V.push_back(v); };
    add(r, 4.0);
    if (i > 0) add(r - m, -1.0);
    if (i < m - 1) add(r + m, -1.0);
    if (j > 0) add(r - 1, -1.0);
    if (j < m - 1) add(r + 1, -1.0);
  }
  Mat A;
  ck(MatCreate(PETSC_COMM_WORLD, &A), "MatCreate");
  ck(MatSetSizes(A, N, N, PETSC_DETERMINE, PETSC_DETERMINE), "MatSetSizes");
  ck(MatSetType(A, MATAIJKOKKOS), "MatSetType");
  ck(MatSetPreallocationCOO(A, (PetscCount)I.size(), I.data(), J.data()), "MatSetPreallocationCOO");
  ck(MatSetValuesCOO(A, V.data(), INSERT_VALUES), "MatSetValuesCOO");
  ck(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
  ck(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");

  std::vector<std::string> scen;
  for (auto& k : list_names(KSPList)) scen.push_back("-ksp_type " + k + " -pc_type jacobi");
  for (auto& p : list_names(PCList)) scen.push_back("-ksp_type richardson -pc_type " + p);
  for (std::string g : {"agg", "classical"})
    for (std::string lk : {"chebyshev", "richardson", "cg"})
      for (std::string lp : {"jacobi", "pbjacobi", "sor", "bjacobi"})
        scen.push_back("-ksp_type cg -pc_type gamg -pc_gamg_type " + g + " -mg_levels_ksp_type " + lk +
                       " -mg_levels_pc_type " + lp);

  PetscHelpPrintf = capture_help;
  std::ofstream out(dir + "/petsc_help.txt");
  int ok = 0;
  for (size_t i = 0; i < scen.size(); ++i) {
    std::string pre = "s" + std::to_string(i) + "_";
    std::string opts = "-help";  // prefix every scenario option
    std::istringstream ss(scen[i]);
    std::string tok;
    while (ss >> tok) opts += " " + (tok[0] == '-' ? "-" + pre + tok.substr(1) : tok);
    g_help.clear();
    std::string err;
    KSP ksp = nullptr;
    try {
      ck(PetscOptionsClear(nullptr), "PetscOptionsClear");
      ck(PetscOptionsInsertString(nullptr, opts.c_str()), "PetscOptionsInsertString");
      ck(KSPCreate(PETSC_COMM_WORLD, &ksp), "KSPCreate");
      ck(KSPSetOptionsPrefix(ksp, pre.c_str()), "KSPSetOptionsPrefix");
      ck(KSPSetOperators(ksp, A, A), "KSPSetOperators");
      ck(KSPSetFromOptions(ksp), "KSPSetFromOptions");
      ck(KSPSetUp(ksp), "KSPSetUp");
      ++ok;
    } catch (std::exception& e) {
      err = e.what();
    }
    if (ksp) KSPDestroy(&ksp);
    out << "### SCENARIO " << pre << " " << scen[i] << "\n";
    if (!err.empty()) out << "### ERROR " << err << "\n";
    out << g_help << "\n";
  }
  ck(PetscOptionsClear(nullptr), "PetscOptionsClear");
  MatDestroy(&A);
  fprintf(stderr, "petsc catalogue: %d/%zu scenarios set up\n", ok, scen.size());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  // No PETSc options on the command line: each request brings its own.
  int pargc = 1;
  if (PetscInitialize(&pargc, &argv, nullptr, nullptr)) return 1;
  int rc;
  if (argc == 3 && std::string(argv[1]) == "--catalog") {
    rc = catalog(argv[2]);
    PetscFinalize();
    return rc;
  }
  {
    PetscDriver d;
    rc = serve(d, argc, argv);
    for (auto& kv : d.mats) MatDestroy(&kv.second);
  }
  PetscFinalize();
  return rc;
}
