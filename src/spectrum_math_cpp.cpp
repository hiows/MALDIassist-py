// C++ acceleration for the KDE (Nadaraya-Watson kernel regression) hot paths.
//
// Ported from Rpkg/MALDIassist/src/spectrum_math.cpp (the original Rcpp source)
// with the Rcpp NumericVector interface replaced by nanobind ndarrays. The inner
// summation is a strict left-to-right accumulation (``sum += a[j]``), matching
// R's Rcpp loop and the pure-Python ``_seqsum`` reference so that tie-breaking in
// extremum selection is reproduced bit-for-bit.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace nb = nanobind;

namespace {

constexpr double SQRT_2PI = 2.50662827463100050242;
constexpr double DEFAULT_KDE_CUTOFF = 5.0;

using arr_in = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

inline double nan_val() { return std::numeric_limits<double>::quiet_NaN(); }

inline bool is_strictly_increasing(const double* v, std::size_t n) {
  if (n < 2) return true;
  for (std::size_t i = 1; i < n; ++i) {
    if (v[i] <= v[i - 1]) return false;
  }
  return true;
}

// Inclusive [j_lo, j_hi] index range of observations within [xi-radius, xi+radius].
// Equivalent to numpy searchsorted(left/right); j_lo > j_hi means an empty window.
inline void support_bounds(double xi, const double* x_obs, std::size_t n_obs,
                           double radius, std::ptrdiff_t& j_lo,
                           std::ptrdiff_t& j_hi) {
  const double lo = xi - radius;
  const double hi = xi + radius;
  j_lo = std::lower_bound(x_obs, x_obs + n_obs, lo) - x_obs;
  j_hi = (std::upper_bound(x_obs, x_obs + n_obs, hi) - x_obs) - 1;
}

inline void accumulate(double xi, const double* x_obs, const double* y_obs,
                       double bw, std::ptrdiff_t j_lo, std::ptrdiff_t j_hi,
                       double& n0, double& d0, double& n1, double& d1,
                       double& n2, double& d2, double& n3, double& d3) {
  const double bw2 = bw * bw;
  const double bw4 = bw2 * bw2;
  const double bw6 = bw4 * bw2;
  const double inv_bw2 = 1.0 / bw2;
  const double inv_bw4 = 1.0 / bw4;
  const double inv_bw6 = 1.0 / bw6;
  const double kernel_const = 1.0 / (bw * SQRT_2PI);

  n0 = 0.0; d0 = 0.0;
  n1 = 0.0; d1 = 0.0;
  n2 = 0.0; d2 = 0.0;
  n3 = 0.0; d3 = 0.0;

  for (std::ptrdiff_t j = j_lo; j <= j_hi; ++j) {
    const double diff = xi - x_obs[j];
    const double yj = y_obs[j];
    const double k = std::exp(-0.5 * diff * diff * inv_bw2) * kernel_const;
    const double k1 = -diff * inv_bw2 * k;
    const double k2 = (diff * diff * inv_bw4 - inv_bw2) * k;
    const double k3 = diff * (3.0 * bw2 - diff * diff) * inv_bw6 * k;

    n0 += k * yj;  d0 += k;
    n1 += k1 * yj; d1 += k1;
    n2 += k2 * yj; d2 += k2;
    n3 += k3 * yj; d3 += k3;
  }
}

inline double combine(int deriv_order, double n0, double d0, double n1, double d1,
                      double n2, double d2, double n3, double d3) {
  if (deriv_order == 0) {
    return n0 / d0;
  }
  if (deriv_order == 1) {
    return (n1 * d0 - n0 * d1) / (d0 * d0);
  }
  if (deriv_order == 2) {
    const double d0_2 = d0 * d0;
    const double d0_3 = d0_2 * d0;
    return n2 / d0 - n0 * d2 / d0_2 - 2.0 * n1 * d1 / d0_2 +
           2.0 * n0 * d1 * d1 / d0_3;
  }
  const double d0_2 = d0 * d0;
  const double d0_3 = d0_2 * d0;
  const double d0_4 = d0_3 * d0;
  return n3 / d0 - 3.0 * n2 * d1 / d0_2 +
         3.0 * n1 * (2.0 * d1 * d1 / d0_3 - d2 / d0_2) +
         n0 * (-d3 / d0_2 + 6.0 * d1 * d2 / d0_3 - 6.0 * d1 * d1 * d1 / d0_4);
}

inline double kde_deriv_scalar_core(double xi, const double* x_obs,
                                    const double* y_obs, std::size_t n_obs,
                                    double bw, int deriv_order, bool use_truncation,
                                    double cutoff) {
  const double denom_floor = std::numeric_limits<double>::min();
  std::ptrdiff_t j_lo = 0;
  std::ptrdiff_t j_hi = static_cast<std::ptrdiff_t>(n_obs) - 1;

  if (use_truncation) {
    support_bounds(xi, x_obs, n_obs, cutoff * bw, j_lo, j_hi);
    if (j_lo > j_hi) return nan_val();
  }

  double n0, d0, n1, d1, n2, d2, n3, d3;
  accumulate(xi, x_obs, y_obs, bw, j_lo, j_hi, n0, d0, n1, d1, n2, d2, n3, d3);
  if (d0 <= denom_floor) return nan_val();
  return combine(deriv_order, n0, d0, n1, d1, n2, d2, n3, d3);
}

inline bool close_to_existing_root(const std::vector<double>& roots,
                                   double candidate, double tol) {
  for (double r : roots) {
    if (std::abs(r - candidate) <= tol) return true;
  }
  return false;
}

inline double bisection_root_kde_deriv(const double* x_obs, const double* y_obs,
                                       std::size_t n_obs, double bw,
                                       int deriv_order, bool use_truncation,
                                       double cutoff, double left, double right,
                                       double yi, double tol, int max_iter) {
  double f_left =
      kde_deriv_scalar_core(left, x_obs, y_obs, n_obs, bw, deriv_order,
                            use_truncation, cutoff) - yi;
  double f_right =
      kde_deriv_scalar_core(right, x_obs, y_obs, n_obs, bw, deriv_order,
                            use_truncation, cutoff) - yi;

  if (!std::isfinite(f_left) || !std::isfinite(f_right)) {
    throw std::runtime_error(
        "KDE derivative returned non-finite values during bisection.");
  }
  if (std::abs(f_left) <= tol) return left;
  if (std::abs(f_right) <= tol) return right;
  if (f_left * f_right > 0.0) {
    throw std::runtime_error("Bisection requires a sign-changing interval.");
  }

  double mid = 0.5 * (left + right);
  for (int iter = 0; iter < max_iter; ++iter) {
    mid = 0.5 * (left + right);
    const double f_mid =
        kde_deriv_scalar_core(mid, x_obs, y_obs, n_obs, bw, deriv_order,
                              use_truncation, cutoff) - yi;
    if (!std::isfinite(f_mid)) {
      throw std::runtime_error(
          "KDE derivative returned non-finite values during bisection.");
    }
    if (std::abs(f_mid) <= tol || std::abs(right - left) <= tol) {
      return mid;
    }
    if (f_left * f_mid <= 0.0) {
      right = mid;
      f_right = f_mid;
    } else {
      left = mid;
      f_left = f_mid;
    }
  }
  return mid;
}

// Build a 1-D numpy array owning a freshly allocated buffer.
nb::object vec_to_numpy(const std::vector<double>& v) {
  const std::size_t n = v.size();
  double* data = new double[n ? n : 1];
  std::copy(v.begin(), v.end(), data);
  nb::capsule owner(data, [](void* p) noexcept {
    delete[] static_cast<double*>(p);
  });
  return nb::cast(nb::ndarray<nb::numpy, double, nb::ndim<1>>(data, {n}, owner));
}

nb::object scalar_nan_array() {
  return vec_to_numpy(std::vector<double>{nan_val()});
}

}  // namespace

// ---------------------------------------------------------------------------
// Exposed functions (thin wrappers over the raw-pointer cores).
// ---------------------------------------------------------------------------

static nb::object gauss_kde_eval(arr_in x, arr_in x_obs, arr_in y_obs, double bw,
                                 int deriv_order, bool use_truncation,
                                 double cutoff) {
  const std::size_t n = x.shape(0);
  const std::size_t n_obs = x_obs.shape(0);
  const double* xp = x.data();
  const double* xo = x_obs.data();
  const double* yo = y_obs.data();
  const double denom_floor = std::numeric_limits<double>::min();

  const bool sorted_obs = is_strictly_increasing(xo, n_obs);
  const bool truncate = use_truncation && sorted_obs;

  double* out = new double[n ? n : 1];
  nb::capsule owner(out, [](void* p) noexcept {
    delete[] static_cast<double*>(p);
  });

  for (std::size_t i = 0; i < n; ++i) {
    const double xi = xp[i];
    std::ptrdiff_t j_lo = 0;
    std::ptrdiff_t j_hi = static_cast<std::ptrdiff_t>(n_obs) - 1;
    if (truncate) {
      support_bounds(xi, xo, n_obs, cutoff * bw, j_lo, j_hi);
      if (j_lo > j_hi) {
        out[i] = nan_val();
        continue;
      }
    }
    double n0, d0, n1, d1, n2, d2, n3, d3;
    accumulate(xi, xo, yo, bw, j_lo, j_hi, n0, d0, n1, d1, n2, d2, n3, d3);
    if (d0 <= denom_floor) {
      out[i] = nan_val();
      continue;
    }
    out[i] = combine(deriv_order, n0, d0, n1, d1, n2, d2, n3, d3);
  }

  return nb::cast(nb::ndarray<nb::numpy, double, nb::ndim<1>>(out, {n}, owner));
}

static nb::object gauss_kde_all_eval(arr_in x, arr_in x_obs, arr_in y_obs,
                                     double bw, bool use_truncation,
                                     double cutoff) {
  const std::size_t n = x.shape(0);
  const std::size_t n_obs = x_obs.shape(0);
  const double* xp = x.data();
  const double* xo = x_obs.data();
  const double* yo = y_obs.data();
  const double denom_floor = std::numeric_limits<double>::min();

  const bool sorted_obs = is_strictly_increasing(xo, n_obs);
  const bool truncate = use_truncation && sorted_obs;

  double* out = new double[(n ? n : 1) * 4];
  nb::capsule owner(out, [](void* p) noexcept {
    delete[] static_cast<double*>(p);
  });

  for (std::size_t i = 0; i < n; ++i) {
    const double xi = xp[i];
    std::ptrdiff_t j_lo = 0;
    std::ptrdiff_t j_hi = static_cast<std::ptrdiff_t>(n_obs) - 1;
    double* row = out + i * 4;
    if (truncate) {
      support_bounds(xi, xo, n_obs, cutoff * bw, j_lo, j_hi);
      if (j_lo > j_hi) {
        row[0] = row[1] = row[2] = row[3] = nan_val();
        continue;
      }
    }
    double n0, d0, n1, d1, n2, d2, n3, d3;
    accumulate(xi, xo, yo, bw, j_lo, j_hi, n0, d0, n1, d1, n2, d2, n3, d3);
    if (d0 <= denom_floor) {
      row[0] = row[1] = row[2] = row[3] = nan_val();
      continue;
    }
    const double d0_2 = d0 * d0;
    const double d0_3 = d0_2 * d0;
    const double d0_4 = d0_3 * d0;
    row[0] = n0 / d0;
    row[1] = (n1 * d0 - n0 * d1) / d0_2;
    row[2] = n2 / d0 - n0 * d2 / d0_2 - 2.0 * n1 * d1 / d0_2 +
             2.0 * n0 * d1 * d1 / d0_3;
    row[3] = n3 / d0 - 3.0 * n2 * d1 / d0_2 +
             3.0 * n1 * (2.0 * d1 * d1 / d0_3 - d2 / d0_2) +
             n0 * (-d3 / d0_2 + 6.0 * d1 * d2 / d0_3 - 6.0 * d1 * d1 * d1 / d0_4);
  }

  return nb::cast(nb::ndarray<nb::numpy, double, nb::ndim<2>>(out, {n, 4}, owner));
}

static double kde_deriv_scalar(double xi, arr_in x_obs, arr_in y_obs, double bw,
                               int deriv_order, bool use_truncation,
                               double cutoff) {
  return kde_deriv_scalar_core(xi, x_obs.data(), y_obs.data(), x_obs.shape(0), bw,
                               deriv_order, use_truncation, cutoff);
}

static std::vector<double> find_roots_core(const double* xp, const double* yg,
                                           std::size_t n, const double* xo,
                                           const double* yo, std::size_t n_obs,
                                           double bw, double yi, double tol,
                                           int max_iter, bool use_truncation,
                                           double cutoff) {
  std::vector<double> roots;
  roots.reserve(n / 4 + 1);

  for (std::size_t i = 0; i < n; ++i) {
    const double fi = yg[i] - yi;
    if (std::abs(fi) <= tol && !close_to_existing_root(roots, xp[i], tol)) {
      roots.push_back(xp[i]);
    }
  }
  for (std::size_t i = 0; i + 1 < n; ++i) {
    const double f_left = yg[i] - yi;
    const double f_right = yg[i + 1] - yi;
    if (f_left == 0.0 || f_right == 0.0) continue;
    if (f_left * f_right < 0.0) {
      const double root = bisection_root_kde_deriv(
          xo, yo, n_obs, bw, 1, use_truncation, cutoff, xp[i], xp[i + 1], yi, tol,
          max_iter);
      if (!close_to_existing_root(roots, root, tol)) {
        roots.push_back(root);
      }
    }
  }
  std::sort(roots.begin(), roots.end());
  return roots;
}

static nb::object find_roots_on_grid(arr_in x, arr_in y_grid, arr_in x_obs,
                                     arr_in y_obs, double bw, double yi,
                                     double tol, int max_iter,
                                     bool use_truncation, double cutoff) {
  std::vector<double> roots =
      find_roots_core(x.data(), y_grid.data(), x.shape(0), x_obs.data(),
                      y_obs.data(), x_obs.shape(0), bw, yi, tol, max_iter,
                      use_truncation, cutoff);
  return vec_to_numpy(roots);
}

static nb::object find_extrema_from_grid(arr_in x, arr_in d1_grid, arr_in x_obs,
                                         arr_in y_obs, double bw, double tol,
                                         int max_iter, bool use_truncation,
                                         double cutoff) {
  const double* xo = x_obs.data();
  const double* yo = y_obs.data();
  const std::size_t n_obs = x_obs.shape(0);

  std::vector<double> x_roots =
      find_roots_core(x.data(), d1_grid.data(), x.shape(0), xo, yo, n_obs, bw,
                      0.0, tol, max_iter, use_truncation, cutoff);

  nb::dict result;
  if (x_roots.empty()) {
    result["local_min"] = scalar_nan_array();
    result["local_max"] = scalar_nan_array();
    result["plateau"] = scalar_nan_array();
    return result;
  }

  std::vector<double> local_max, local_min, plateau;
  for (double xr : x_roots) {
    const double d2 =
        kde_deriv_scalar_core(xr, xo, yo, n_obs, bw, 2, use_truncation, cutoff);
    if (!std::isfinite(d2)) {
      throw std::runtime_error(
          "Second KDE derivative non-finite at root positions.");
    }
    if (d2 < -tol) {
      local_max.push_back(xr);
    } else if (d2 > tol) {
      local_min.push_back(xr);
    } else {
      plateau.push_back(xr);
    }
  }

  result["local_min"] =
      local_min.empty() ? scalar_nan_array() : vec_to_numpy(local_min);
  result["local_max"] =
      local_max.empty() ? scalar_nan_array() : vec_to_numpy(local_max);
  result["plateau"] =
      plateau.empty() ? scalar_nan_array() : vec_to_numpy(plateau);
  return result;
}

NB_MODULE(_spectrum_math_cpp, m) {
  m.doc() = "C++ acceleration for MALDIassist KDE hot paths (nanobind).";

  m.def("gauss_kde_eval", &gauss_kde_eval, nb::arg("x"), nb::arg("x_obs"),
        nb::arg("y_obs"), nb::arg("bw"), nb::arg("deriv_order"),
        nb::arg("use_truncation") = true, nb::arg("cutoff") = DEFAULT_KDE_CUTOFF);

  m.def("gauss_kde_all_eval", &gauss_kde_all_eval, nb::arg("x"), nb::arg("x_obs"),
        nb::arg("y_obs"), nb::arg("bw"), nb::arg("use_truncation") = true,
        nb::arg("cutoff") = DEFAULT_KDE_CUTOFF);

  m.def("kde_deriv_scalar", &kde_deriv_scalar, nb::arg("xi"), nb::arg("x_obs"),
        nb::arg("y_obs"), nb::arg("bw"), nb::arg("deriv_order"),
        nb::arg("use_truncation"), nb::arg("cutoff"));

  m.def("find_roots_on_grid", &find_roots_on_grid, nb::arg("x"),
        nb::arg("y_grid"), nb::arg("x_obs"), nb::arg("y_obs"), nb::arg("bw"),
        nb::arg("yi"), nb::arg("tol"), nb::arg("max_iter"),
        nb::arg("use_truncation"), nb::arg("cutoff"));

  m.def("find_extrema_from_grid", &find_extrema_from_grid, nb::arg("x"),
        nb::arg("d1_grid"), nb::arg("x_obs"), nb::arg("y_obs"), nb::arg("bw"),
        nb::arg("tol"), nb::arg("max_iter"), nb::arg("use_truncation"),
        nb::arg("cutoff"));
}
