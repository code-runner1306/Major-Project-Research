// Placeholder pybind11 bindings for the fused GDN recurrence (OpenSpec 4.5).
//
// Exposes forward/backward entry points with the canonical
// (q, k, v, alpha, beta, initial_state, options) signature once the sm_70
// kernels land. No numeric fallback lives here: unavailable paths raise.
