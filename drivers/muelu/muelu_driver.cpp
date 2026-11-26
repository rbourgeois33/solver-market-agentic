// MueLu + Belos bench server on Tpetra (Kokkos CUDA node; Epetra is deprecated and not used).
// Request keys: solver (Belos name), belos_xml, muelu_xml (written by sma/deck.py).
// CG-family solvers get MueLu as LEFT preconditioner, all others as RIGHT preconditioner
// (so GMRES's residual is the unpreconditioned one); request key "prec_side" overrides.
#include <BelosLinearProblem.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>
#include <Amesos2.hpp>
#include <Ifpack2_Parameters.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_MasterList.hpp>
#include <MueLu_Hierarchy.hpp>
#include <MueLu_Level.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <Xpetra_Matrix.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Trilinos_version.h>

#include "sma_common.hpp"

using namespace sma;

namespace {

using SC = double;
using map_t = Tpetra::Map<>;
using LO = map_t::local_ordinal_type;
using GO = map_t::global_ordinal_type;
using NO = map_t::node_type;
using crs_t = Tpetra::CrsMatrix<SC, LO, GO, NO>;
using mv_t = Tpetra::MultiVector<SC, LO, GO, NO>;
using op_t = Tpetra::Operator<SC, LO, GO, NO>;

struct MueLuDriver {
  Teuchos::RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
  Teuchos::RCP<crs_t> A;
  Teuchos::RCP<mv_t> X, B;

  void load(const Csr& h, const std::vector<double>& hb) {
    if (h.nrows > INT32_MAX) throw std::runtime_error("too many rows for LO=int");
    auto map = Teuchos::rcp(new map_t((Tpetra::global_size_t)h.nrows, 0, comm));
    using graph_t = crs_t::local_graph_device_type;
    typename graph_t::row_map_type::non_const_type rowptr("rowptr", h.nrows + 1);
    typename graph_t::entries_type::non_const_type colind("colind", h.nnz);
    typename crs_t::local_matrix_device_type::values_type::non_const_type values("values", h.nnz);
    auto hr = Kokkos::create_mirror_view(rowptr);
    auto hc = Kokkos::create_mirror_view(colind);
    auto hv = Kokkos::create_mirror_view(values);
    for (int64_t i = 0; i <= h.nrows; ++i) hr(i) = (size_t)h.rowptr[i];
    for (int64_t k = 0; k < h.nnz; ++k) { hc(k) = (LO)h.colind[k]; hv(k) = h.values[k]; }
    Kokkos::deep_copy(rowptr, hr);
    Kokkos::deep_copy(colind, hc);
    Kokkos::deep_copy(values, hv);
    // 1 rank: column map == row map, local column ids == global ids.
    A = Teuchos::rcp(new crs_t(map, map, rowptr, colind, values));
    A->expertStaticFillComplete(map, map);
    X = Teuchos::rcp(new mv_t(map, 1));
    B = Teuchos::rcp(new mv_t(map, 1));
    {
      auto bv = B->getLocalViewHost(Tpetra::Access::OverwriteAll);
      for (int64_t i = 0; i < h.nrows; ++i) bv(i, 0) = hb[i];
    }
    Kokkos::fence();
  }

  // Operator / grid complexity from the level matrices. (Hierarchy::GetOperatorComplexity returns 0
  // when coarse operators have no global constants computed, which happens in normal runs.)
  template <class H>
  static void complexities(H& h, RunStats& st) {
    using xmat_t = Xpetra::Matrix<SC, LO, GO, NO>;
    double nnz = 0, nnz0 = 0, rows = 0, rows0 = 0;
    int L = h.GetNumLevels();
    for (int l = 0; l < L; ++l) {
      auto lvl = h.GetLevel(l);
      if (!lvl->IsAvailable("A")) break;
      auto A = Teuchos::rcp_dynamic_cast<xmat_t>(lvl->template Get<Teuchos::RCP<Xpetra::Operator<SC, LO, GO, NO>>>("A"));
      if (A.is_null()) break;
      double z = (double)A->getLocalNumEntries(), r = (double)A->getLocalNumRows();  // 1 rank in v1
      if (l == 0) { nnz0 = z; rows0 = r; }
      nnz += z; rows += r;
    }
    st.levels = L;
    st.op_complexity = nnz0 > 0 ? nnz / nnz0 : NAN;
    st.grid_complexity = rows0 > 0 ? rows / rows0 : NAN;
  }

  std::vector<double> x_host() {
    auto xv = X->getLocalViewHost(Tpetra::Access::ReadOnly);
    std::vector<double> hx(xv.extent(0));
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = xv(i, 0);
    return hx;
  }

  Json run(const Request& rq, const Csr& hA, const std::vector<double>& hb) {
    double tol = std::stod(rq.need("tol"));
    std::string solver_name = rq.need("solver");
    auto belos_params = Teuchos::getParametersFromXmlFile(rq.need("belos_xml"));
    auto muelu_xml = rq.need("muelu_xml");
    std::string side = rq.get("prec_side", solver_name.find("CG") != std::string::npos ? "left" : "right");

    RunStats st;
    Measure m(rq);
    MemSampler mem;
    mem.start();
    st.mem_baseline = mem.baseline();
    Belos::SolverFactory<SC, mv_t, op_t> factory;
    Teuchos::RCP<MueLu::TpetraOperator<SC, LO, GO, NO>> prec;
    Teuchos::RCP<Belos::SolverManager<SC, mv_t, op_t>> solver;
    Teuchos::RCP<Belos::LinearProblem<SC, mv_t, op_t>> problem;
    bool first = true;
    auto release = [&] {
      solver = Teuchos::null;
      problem = Teuchos::null;
      prec = Teuchos::null;
      Kokkos::fence();
    };
    auto setup = [&] {
      // Fresh copy each time: MueLu may annotate/modify the parameter list it is given.
      auto muelu_params = Teuchos::getParametersFromXmlFile(muelu_xml);
      X->putScalar(0.0);
      Kokkos::fence();
      double t0 = now();
      Teuchos::RCP<op_t> Aop = A;
      prec = MueLu::CreateTpetraPreconditioner(Aop, *muelu_params);
      Kokkos::fence();
      double ts = now() - t0;
      // Belos objects are built outside the timers (no numerical work).
      auto params = Teuchos::rcp(new Teuchos::ParameterList(*belos_params));
      solver = factory.create(solver_name, params);
      problem = Teuchos::rcp(new Belos::LinearProblem<SC, mv_t, op_t>(A, X, B));
      if (side == "left") problem->setLeftPrec(prec);
      else problem->setRightPrec(prec);
      if (!problem->setProblem()) throw std::runtime_error("Belos setProblem failed");
      solver->setProblem(problem);
      return ts;
    };
    auto solve = [&](int i) {
      X->putScalar(0.0);
      Kokkos::fence();
      // Belos keeps the problem; after resetting X it must recompute the initial residual.
      if (!problem->setProblem()) throw std::runtime_error("Belos setProblem failed");
      solver->reset(Belos::Problem);
      double t0 = now();
      Belos::ReturnType ret = solver->solve();
      Kokkos::fence();
      double tv = now() - t0;
      if (i < 0) return tv;
      st.iters_all.push_back(solver->getNumIters());
      st.lib_converged = ret == Belos::Converged;
      st.relres_true.push_back(true_relres(hA, x_host(), hb));
      if (first) { complexities(*prec->GetHierarchy(), st); first = false; }
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
    fill_result(j, rq, "muelu", TRILINOS_VERSION_STRING, st, tol);
    j.str("solver", solver_name).str("prec_side", side);
    return j;
  }
};

// ------------------------------------------------------------------ parameter catalogue
// --catalog DIR: the libraries' own lists of valid parameters, with defaults and doc strings:
//   muelu_master.xml  MueLu::MasterList (every MueLu parameter and its default)
//   ifpack2.xml       Ifpack2::getValidParameters (smoother parameters, "smoother: params")
//   belos_<i>.xml     getValidParameters() of every registered Belos solver (names in belos.txt)
//   amesos2.txt       coarse direct solvers: name and whether this build provides it
// sma/catalog.py turns them into muelu.json.
int catalog(const std::string& dir) {
  Teuchos::writeParameterListToXmlFile(*MueLu::MasterList::List(), dir + "/muelu_master.xml");
  Teuchos::ParameterList ifp;
  Ifpack2::getValidParameters(ifp);
  Teuchos::writeParameterListToXmlFile(ifp, dir + "/ifpack2.xml");
  Belos::SolverFactory<SC, mv_t, op_t> factory;
  std::ofstream names(dir + "/belos.txt");
  int i = 0;
  for (auto& n : factory.supportedSolverNames()) {
    try {
      auto solver = factory.create(n, Teuchos::parameterList());
      Teuchos::writeParameterListToXmlFile(*solver->getValidParameters(), dir + "/belos_" + std::to_string(i) + ".xml");
      names << i << " " << n << "\n";
    } catch (std::exception& e) {
      std::string msg = e.what();
      for (char& c : msg)
        if (c == '\n') c = ' ';
      names << "- " << n << " ERROR " << msg << "\n";
    }
    ++i;
  }
  std::ofstream am(dir + "/amesos2.txt");
  for (const char* n : {"KLU2", "Basker", "ShyLU_Basker", "Tacho", "cuSOLVER", "SuperLU", "SuperLUMT", "SuperLUDist",
                        "Umfpack", "Cholmod", "MUMPS", "PardisoMKL", "Lapack", "STRUMPACK", "CSS_MKL"})
    am << n << " " << (Amesos2::query(n) ? 1 : 0) << "\n";
  fprintf(stderr, "muelu catalogue written to %s\n", dir.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Tpetra::ScopeGuard scope(&argc, &argv);
  if (argc == 3 && std::string(argv[1]) == "--catalog") return catalog(argv[2]);
  MueLuDriver d;
  int rc = serve(d, argc, argv);
  d.A = Teuchos::null; d.X = Teuchos::null; d.B = Teuchos::null;
  return rc;
}
