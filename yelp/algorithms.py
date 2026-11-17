"""The five final algotithms."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize, root
from scipy.special import expit

from optimization import LogMinDesignSolver

ALPHA = 1.5
FISHER = 'hyconucb'
CDSPAN = 'conduel'
RANDOM = 'var_random'
AM = 'var_am'
LEE = 'ofuglb_ucb'
ALGORITHMS = (FISHER, RANDOM, AM, CDSPAN, LEE)
DISPLAY_NAMES = {FISHER: 'HyConUCB', CDSPAN: 'ConDuel', RANDOM: 'Var-Random',
                 AM: 'Var-AM', LEE: 'OFUGLB-UCB'}


def logistic_curvature(logit):
    z = np.asarray(logit, dtype=float)
    return expit(z) * expit(-z)


class LogisticModel:
    def __init__(self, d, S, delta=.05, ridge=0., tol=1e-8):
        self.d, self.S, self.delta = int(d), float(S), float(delta)
        self.ridge, self.tol = float(ridge), float(tol)
        self.n = 0
        self.theta = np.zeros(self.d)
        self.sum_norms = 0.0
        self._records = {}
        self._X = np.empty((0, self.d))
        self._counts = np.empty(0)
        self._ysums = np.empty(0)
        self._data_dirty = False
        self._fit_dirty = True
        self.mle_lower = self.mle_upper = self.mle_gap = 0.0
        self.mle_success = True
        self.fit()

    @property
    def lipschitz_bound(self) -> float:
        return self.sum_norms + self.ridge * self.S

    @property
    def beta2(self) -> float:
        a = 2.0 * self.S * self.lipschitz_bound
        penalty = a if a <= self.d else self.d * (1.0 + np.log(a / self.d))
        return float(np.log(2.0 / self.delta) + penalty)

    def add(self, x: np.ndarray, y: float) -> None:
        x = np.asarray(x, dtype=float)
        key = tuple(x)
        record = self._records.setdefault(key, [0.0, 0.0])
        record[0] += 1
        record[1] += float(y)
        self.n += 1
        self.sum_norms += float(np.linalg.norm(x))
        self._data_dirty = self._fit_dirty = True


    def _refresh_data(self) -> None:
        if self._data_dirty:
            self._X = np.asarray(list(self._records), dtype=float).reshape(-1, self.d)
            values = np.asarray(list(self._records.values()), dtype=float)
            self._counts, self._ysums = values[:, 0], values[:, 1]
            self._data_dirty = False

    def loss(self, theta: np.ndarray) -> float:
        self._refresh_data()
        theta = np.asarray(theta, dtype=float)
        z = self._X @ theta
        losses = ((self._counts - self._ysums) * np.logaddexp(0.0, z)
                  + self._ysums * np.logaddexp(0.0, -z))
        return float(losses.sum() + 0.5 * self.ridge * (theta @ theta))

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        self._refresh_data()
        theta = np.asarray(theta, dtype=float)
        z = self._X @ theta
        residual = ((self._counts - self._ysums) * expit(z)
                    - self._ysums * expit(-z))
        return self._X.T @ residual + self.ridge * theta

    def _hessian_at(self, theta: np.ndarray) -> np.ndarray:
        weights = self._counts * logistic_curvature(self._X @ theta)
        return self._X.T @ (weights[:, None] * self._X) + self.ridge * np.eye(self.d)

    def hessian(self) -> np.ndarray:
        self.fit()
        return self._hessian_at(self.theta)

    def _roundoff(self, multiplier: float = 1.0, extra: float = 0.0) -> float:
        scale = (1.0 + self.n + self.S * self.sum_norms
                 + self.ridge * self.S**2 + abs(extra))
        return float(64.0 * np.finfo(float).eps * (1.0 + abs(multiplier)) * scale)

    def _project_ball(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float).copy()
        norm = np.linalg.norm(theta)
        if norm > self.S:
            theta *= self.S / norm
            theta *= 1.0 - 2.0 * np.finfo(float).eps
        return theta

    def _mle_certificate(self, theta: np.ndarray) -> tuple[float, float]:
        value, grad = self.loss(theta), self.gradient(theta)
        dual_gap = max(0.0, float(grad @ theta + self.S * np.linalg.norm(grad)))
        pad = self._roundoff(extra=value)
        return value - dual_gap - pad, value + pad

    def _refine_mle(self, theta: np.ndarray) -> np.ndarray:
        g = self.gradient(theta)
        active = np.linalg.norm(theta) >= self.S * (1.0 - 1e-6) and g @ theta < 0
        if active:
            lam = max(0.0, float(-g @ theta / self.S**2))

            def fun(w):
                t, l = w[:-1], w[-1]
                return np.r_[self.gradient(t) + l * t, 0.5 * (t @ t - self.S**2)]

            def jac(w):
                t, l = w[:-1], w[-1]
                out = np.zeros((self.d + 1, self.d + 1))
                out[:-1, :-1] = self._hessian_at(t) + l * np.eye(self.d)
                out[:-1, -1] = out[-1, :-1] = t
                return out

            res = root(fun, np.r_[theta, lam], jac=jac, method="lm",
                       options={"ftol": 1e-13, "xtol": 1e-13, "gtol": 1e-13, "maxiter": 200})
            candidate = res.x[:-1]
            if res.x[-1] < -self.tol:
                return theta
        else:
            res = root(self.gradient, theta, jac=self._hessian_at, method="lm",
                       options={"ftol": 1e-13, "xtol": 1e-13, "gtol": 1e-13, "maxiter": 200})
            candidate = res.x
        if not np.all(np.isfinite(candidate)) or np.linalg.norm(candidate) > self.S * (1 + 1e-7):
            return theta
        candidate = self._project_ball(candidate)
        lo_old, hi_old = self._mle_certificate(theta)
        lo_new, hi_new = self._mle_certificate(candidate)
        if hi_new - lo_new < hi_old - lo_old and self.loss(candidate) <= self.loss(theta) + self._roundoff():
            return candidate
        return theta

    def _cvx_expression(self, cp, variable):
        if len(self._counts):
            z = self._X @ variable
            expr = cp.sum(cp.multiply(self._counts - self._ysums, cp.logistic(z))
                          + cp.multiply(self._ysums, cp.logistic(-z)))
        else:
            expr = cp.Constant(0.0)
        return expr + 0.5 * self.ridge * cp.sum_squares(variable)

    def _cvx_candidate(self):
        import cvxpy as cp
        variable = cp.Variable(self.d)
        loss = self._cvx_expression(cp, variable)
        constraints = [cp.norm(variable, 2) <= self.S]
        objective = cp.Minimize(loss)
        problem = cp.Problem(objective, constraints)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                problem.solve(solver="CLARABEL", tol_gap_abs=1e-10,
                              tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=300)
        except (cp.error.SolverError, ValueError):
            return None
        if variable.value is None or not np.all(np.isfinite(variable.value)):
            return None
        return self._project_ball(variable.value)

    def fit(self) -> np.ndarray:
        if not self._fit_dirty:
            return self.theta.copy()
        self._refresh_data()
        if self.n == 0:
            self.theta = np.zeros(self.d)
        else:
            result = minimize(self.loss, self.theta, jac=self.gradient, method="SLSQP",
                              constraints=[{"type": "ineq", "fun": lambda t: self.S**2 - t @ t,
                                            "jac": lambda t: -2.0 * t}],
                              options={"ftol": min(1e-12, self.tol * 0.01), "maxiter": 300})
            if np.all(np.isfinite(result.x)):
                candidate = self._project_ball(result.x)
                if self.loss(candidate) <= self.loss(self.theta) + self._roundoff():
                    self.theta = candidate
            self.theta = self._refine_mle(self.theta)
        self.mle_lower, self.mle_upper = self._mle_certificate(self.theta)
        self.mle_gap = self.mle_upper - self.mle_lower
        tolerance = self.tol * (1.0 + self.beta2) + 2 * self._roundoff()
        if self.mle_gap > tolerance:
            candidate = self._cvx_candidate()
            if candidate is not None:
                candidate = self._refine_mle(candidate)
                lo, hi = self._mle_certificate(candidate)
                if hi - lo < self.mle_gap:
                    self.theta = candidate
                    self.mle_lower, self.mle_upper, self.mle_gap = lo, hi, hi - lo
        self.mle_success = self.mle_gap <= tolerance
        self._fit_dirty = False
        return self.theta.copy()


class EmpiricalUCB(LogisticModel):
    def __init__(self, d, S, delta=.05, tol=1e-6, *, arm_only=False, alpha=ALPHA):
        self.arm_only = bool(arm_only)
        self.alpha = float(alpha)
        self.lambda_hessian = 1/(8*S*S*(1+S)) if arm_only else 1/(4*S**2)
        super().__init__(d, S, delta, ridge=0., tol=tol)

    @property
    def beta2(self):
        if not self.arm_only:
            return super().beta2
        a = 2*self.S*((1+self.S/2)*self.n)
        penalty = a if a <= self.d else self.d*(1+np.log(a/self.d))
        return float(np.log(1/self.delta)+penalty)

    @property
    def gamma2(self):
        return self.alpha**2

    def information_matrix(self):
        return self.hessian()+self.lambda_hessian*np.eye(self.d)

    def choose_arm(self, X, rng):
        X = np.asarray(X, dtype=float)
        self.fit()
        if not self.mle_success:
            raise RuntimeError('Constrained MLE failed its numerical certificate')
        H = self.information_matrix()
        whitened = solve_triangular(np.linalg.cholesky(H), X.T, lower=True, check_finite=False)
        variances = np.sum(whitened*whitened, axis=0)
        means = X @ self.theta
        bonuses = np.sqrt(self.gamma2)*np.sqrt(variances)
        scores = means+bonuses
        maximum = float(scores.max())
        tolerance = float(32*np.finfo(float).eps*(1+abs(maximum)))
        ties = np.flatnonzero(scores >= maximum-tolerance)
        selected = int(rng.choice(ties))
        return selected


class ConDuel:
    def __init__(self, d, S, delta, alpha=ALPHA):
        self.d, self.S = int(d), float(S)
        self.ridge, self.alpha = 5., float(alpha)
        self.kappa = float(expit(2*S)*expit(-2*S))
        self.model = LogisticModel(d, S, delta, ridge=self.ridge, tol=1e-8)
        self.Minv = np.eye(d)*(self.kappa/self.ridge)

    @property
    def theta(self):
        return np.asarray(self.model.fit())

    def fit(self):
        return self.model.fit()

    def add(self, x, y):
        x = np.asarray(x, dtype=float)
        self.model.add(x, y)
        v = self.Minv @ x
        updated = self.Minv-np.outer(v, v)/(1.+x @ v)
        self.Minv = (updated+updated.T)/2.

    def choose_arm(self, X, rng):
        X = np.asarray(X, dtype=float)
        variance = np.einsum('ij,ij->i', X @ self.Minv, X)
        scores = X @ self.theta+self.alpha*np.sqrt(np.maximum(variance, 0.))
        if not self.model.mle_success:
            raise RuntimeError('ConDuel MLE failed its numerical certificate')
        ties = np.flatnonzero(scores == np.max(scores))
        selected = int(rng.choice(ties))
        return selected


class UniformPairQuery:
    def choose(self, learner, key_features, pairs, rng):
        count = len(pairs)
        return int(rng.integers(count))


class AbsoluteMaxVarianceQuery:
    def choose(self, learner, key_features, rng):
        key = np.asarray(key_features, dtype=float)
        learner.fit()
        if not learner.mle_success:
            raise RuntimeError('Uncertified MLE before absolute query')
        H = learner.information_matrix()
        factor = np.linalg.cholesky((H + H.T) / 2.)
        whitened = solve_triangular(factor, key.T, lower=True, check_finite=False)
        values = np.maximum(np.sum(whitened * whitened, axis=0), 0.)
        maximum = float(values.max())
        tie_tolerance = float(32 * np.finfo(float).eps * (1 + abs(maximum)))
        ties = np.flatnonzero(values >= maximum - tie_tolerance)
        selected = int(rng.choice(ties))
        return selected


class FisherLogMinQuery:
    def __init__(self, xi=.05, certificate_tol=1e-6):
        self.xi, self.certificate_tol = float(xi), float(certificate_tol)
        self.solver = None

    def choose(self, learner, key_features, pairs, rng):
        key, pairs = np.asarray(key_features, dtype=float), np.asarray(pairs)
        if self.solver is None:
            self.solver = LogMinDesignSolver(key[pairs[:, 0]]-key[pairs[:, 1]],
                                            self.xi, self.certificate_tol)
        learner.fit()
        if not learner.mle_success:
            raise RuntimeError('Uncertified MLE before Fisher query')
        H = learner.information_matrix()
        inverse = np.linalg.solve(H, np.eye(learner.d))
        inverse = (inverse+inverse.T)/2.
        q = self.solver.q
        values = np.einsum('ij,ij->i', q @ inverse, q)
        maximum = float(values.max())
        cert = self.solver.solve(values/maximum)
        indices = np.asarray(cert['distribution_indices'], dtype=int)
        probability = np.asarray(cert['distribution_probabilities'])
        cdf = np.cumsum(probability)
        cdf[-1] = 1.
        uniform = float(rng.random())
        position = int(np.searchsorted(cdf, uniform, side='right'))
        selected = int(indices[position])
        return selected


def make_policy(algorithm, d=10, S=2., alpha=1.5, xi=.05):
    common = dict(d=d, S=S, delta=.05, alpha=alpha)
    if algorithm == CDSPAN:
        return ConDuel(**common), UniformPairQuery()
    learner = EmpiricalUCB(**common, tol=1e-6, arm_only=algorithm == LEE)
    if algorithm == LEE:
        return learner, None
    if algorithm == AM:
        return learner, AbsoluteMaxVarianceQuery()
    if algorithm == RANDOM:
        return learner, UniformPairQuery()
    if algorithm == FISHER:
        return learner, FisherLogMinQuery(xi=xi, certificate_tol=1e-6)
