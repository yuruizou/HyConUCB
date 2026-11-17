"""Certified query selection with deterministic numerical retries."""
from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np
from scipy.linalg import qr


def _solve_conic(problem, backend, precision):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Solution may be inaccurate.*', category=UserWarning)
        if backend == 'CLARABEL':
            problem.solve(solver=backend, warm_start=False, max_iter=350,
                          tol_gap_abs=precision, tol_gap_rel=precision, tol_feas=precision)
        else:
            problem.solve(solver=backend, warm_start=False, eps_abs=precision, eps_rel=precision,
                          max_iters=200000, acceleration_lookback=10)


class LogMinDesignSolver:
    def __init__(self, vectors, xi=.05, certificate_tol=1e-6):
        self.q = np.asarray(vectors, dtype=float).copy()
        self.xi, self.tol = float(xi), float(certificate_tol)
        gram = self.q.T @ self.q
        eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2.)
        threshold = max(1., float(eigenvalues[-1])) * 1e-12
        rank = int(np.sum(eigenvalues > threshold))
        if rank == 0:
            raise ValueError("A zero-dimensional query pool has no positive information rate")
        self.basis = np.eye(self.q.shape[1]) if rank == self.q.shape[1] else eigenvectors[:, -rank:]
        self.u = self.q if rank == self.q.shape[1] else self.q @ self.basis
        self.rank = rank
        _, _, pivots = qr(self.u.T, mode="economic", pivoting=True)
        self.initial_columns = np.sort(pivots[:rank]).astype(int)
        self.max_squared_norm = float(np.max(np.einsum('ij,ij->i', self.u, self.u)))
        self.warm_indices = self.initial_columns.copy()
        self.warm_probabilities = np.full(rank, 1. / rank)
        self.scalar_geometry_scale = self.max_squared_norm / self.rank

    def certificate(self, indices, probabilities, scores, dual):
        indices = np.asarray(indices, dtype=int)
        raw = np.asarray(probabilities, dtype=float)
        if not np.isfinite(raw).all() or raw.min() < -1e-6:
            raise RuntimeError("The conic primal solution has invalid probabilities")
        p = np.maximum(raw, 0.)
        if p.sum() <= 0:
            raise RuntimeError("The conic primal solution has zero probability mass")
        p /= p.sum()
        v = self.u[indices]
        V = (v.T * p) @ v
        V = (V + V.T) / 2.
        j_raw = float(np.linalg.eigvalsh(V)[0])
        eps = np.finfo(float).eps
        j_padding = (256 * eps * max(1, len(indices)) * self.rank
                     * max(1., self.max_squared_norm, float(np.linalg.norm(V, 2))))
        j_lower = j_raw - j_padding
        if not np.isfinite(j_lower) or j_lower <= 0:
            raise RuntimeError("The actual query distribution lacks certified positive information")
        Y = np.asarray(dual, dtype=float)
        if Y.shape != (self.rank, self.rank) or not np.isfinite(Y).all():
            raise RuntimeError("Invalid conic dual matrix")
        eig, basis = np.linalg.eigh((Y + Y.T) / 2.)
        dual_padding = 256 * eps * self.rank * max(1., float(np.max(np.abs(eig))))
        Y = (basis * np.maximum(eig, 0.)) @ basis.T + dual_padding * np.eye(self.rank)
        Y *= (self.xi / j_lower) / float(np.trace(Y))
        Y = (Y + Y.T) / 2.
        trace = float(np.trace(Y))
        prices = scores + np.einsum('ij,ij->i', self.u @ Y, self.u)
        maximizer = int(np.argmax(prices))
        maximum = float(prices[maximizer])
        pricing_padding = (256 * eps * self.rank
            * max(1., abs(maximum), self.max_squared_norm * float(np.linalg.norm(Y, 2))))
        expected = float(p @ scores[indices])
        normalized_gap = maximum + pricing_padding - expected - self.xi
        lower = expected + self.xi * np.log(j_lower)
        upper = maximum + pricing_padding + self.xi * np.log(self.xi / trace) - self.xi
        gap = max(normalized_gap, float(upper - lower), 0.)
        return {
            'distribution_indices': indices.tolist(), 'distribution_probabilities': p.tolist(),
            'j_lower': j_lower, 'primal_dual_gap': float(gap),
        }, prices, Y

    def _dual_problem(self, active, scores, scale):
        q = self.u[active]
        columns = np.einsum('ki,kj->kij', q, q).reshape(len(active), -1).T / scale
        Y = cp.Variable((self.rank, self.rank), PSD=True)
        bound = cp.Variable()
        inequalities = scores[active] - 1. + columns.T @ cp.vec(Y, order='F') <= bound
        problem = cp.Problem(cp.Minimize(bound-self.xi*cp.log(cp.trace(Y))), [inequalities])
        return problem, Y, inequalities, columns

    def _polish(self, active, scores, cert, columns, scale, backend):
        Y = cp.Variable((self.rank, self.rank), PSD=True)
        bound = cp.Variable()
        problem = cp.Problem(cp.Minimize(bound), [
            cp.trace(Y) == self.xi * scale / cert['j_lower'],
            scores[active] - 1. + columns.T @ cp.vec(Y, order='F') <= bound])
        _solve_conic(problem, backend, 1e-10 if backend == 'CLARABEL' else 1e-9)
        if problem.status not in ('optimal', 'optimal_inaccurate') or Y.value is None:
            raise RuntimeError('Linear certificate SDP status: '+str(problem.status))
        return self.certificate(active, cert['distribution_probabilities'], scores, Y.value/scale)

    def solve(self, scores):
        scores = np.asarray(scores, dtype=float)
        warm = self.warm_indices[self.warm_probabilities > 1e-7]
        active = np.unique(np.r_[self.initial_columns, warm, np.argmax(scores)]).astype(int)
        for _ in range(100):
            outside = np.ones(len(scores), dtype=bool)
            outside[active] = False
            usable = None
            best_gap = float('inf')
            settings = [('CLARABEL', 1e-9, self.scalar_geometry_scale),
                        ('CLARABEL', 1e-11, self.scalar_geometry_scale),
                        ('CLARABEL', 1e-10, 1.),
                        ('SCS', 1e-8, self.scalar_geometry_scale),
                        ('SCS', 1e-10, self.scalar_geometry_scale)]
            for backend, precision, scale in settings:
                try:
                    problem, Yvar, multiplier_constraint, columns = self._dual_problem(active, scores, scale)
                    _solve_conic(problem, backend, precision)
                    if (problem.status not in ('optimal', 'optimal_inaccurate')
                            or Yvar.value is None or multiplier_constraint.dual_value is None):
                        raise RuntimeError('Dual log-trace SDP status: '+str(problem.status))
                    candidate, prices, Y = self.certificate(active, multiplier_constraint.dual_value,
                                                           scores, Yvar.value/scale)
                    violation = outside & (prices > float(prices[active].max()) + self.tol*.05)
                    if candidate['primal_dual_gap'] > self.tol and not np.any(violation):
                        try:
                            refined = self._polish(active, scores, candidate, columns, scale, backend)
                            if refined[0]['primal_dual_gap'] < candidate['primal_dual_gap']:
                                candidate, prices, Y = refined
                                violation = outside & (prices > float(prices[active].max()) + self.tol*.05)
                        except (cp.error.SolverError, RuntimeError, ValueError):
                            pass
                    if candidate['primal_dual_gap'] < best_gap:
                        usable = candidate, prices, Y, violation, backend, problem.status
                        best_gap = candidate['primal_dual_gap']
                    if candidate['primal_dual_gap'] <= self.tol or np.any(violation):
                        usable = candidate, prices, Y, violation, backend, problem.status
                        break
                except (cp.error.SolverError, RuntimeError, ValueError):
                    continue
            if usable is None:
                raise RuntimeError('Fisher design failed: all equivalent conic solves failed')
            cert, prices, Y, violation, backend, status = usable
            if cert['primal_dual_gap'] <= self.tol:
                self.warm_indices = active.copy()
                self.warm_probabilities = np.asarray(cert['distribution_probabilities'])
                return cert
            candidates = np.flatnonzero(violation)
            if not len(candidates):
                raise RuntimeError('Fisher design failed: no valid full-pool certificate')
            order = np.lexsort((candidates, -prices[candidates]))
            active = np.union1d(active, candidates[order[:16]])
        raise RuntimeError('Fisher design failed: column generation iteration limit')
