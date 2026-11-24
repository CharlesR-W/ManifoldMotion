"""Fixture family 5: flow along a fixed curved support, and mixed tangential / normal flow whose two strain
terms partly cancel (SPEC item 8 tested term by term).

The support is the latitude sphere x = 2 (cos theta e0 + sin theta omega).  Tangential flow: the
conformal flow xi = c (e_p - h omega) on the omega-sphere (h = omega . e_p), whose exact time-t map is
tan(alpha'/2) = tan(alpha/2) exp(-c t) (alpha the angle from e_p).  Its strain rate is -c h T (grad xi =
c Hess h = -c h g, invariant under the scaling of the metric), and its finite stretches are both
sin(alpha') / sin(alpha) (a conformal map).  Points stay on the curved support, so the normal part of
the log displacement is O(dt^2) and the normal-motion term vanishes as a rate.  Mixed: add theta -> theta'
(normal term thetadot cot(theta) T, see family 4); the total is (thetadot cot theta - c h) T.
"""
import numpy as np
import pytest

from mm import sphere, step
from mm import fixtures as fx

R = sphere.R
M = 2


def flow_case(N, theta, theta1, c, dt, tau, k, seed=50):
    fr = fx.random_frame(4, np.random.default_rng(seed))
    om = fx.fibonacci_sphere(N, 0.0)
    X = fx.latitude_points(om, theta, fr)
    om1 = fx.conformal_flow(om, c, dt)
    Y = fx.latitude_points(om1, theta1, fr)
    o = step.measure_step(X, Y, dt, operator=dict(kind="fixed", tau=tau, k=k), mask=False)
    h = om[:, 0]
    a0, a1 = np.arccos(np.clip(h, -1, 1)), np.arccos(np.clip(om1[:, 0], -1, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        conf = np.sin(a1) / np.sin(a0)
    s_lat = np.sin(theta1) / np.sin(theta)
    return o, dict(h=h, conf=conf, s=conf * s_lat, flow=-c * h, normal=(theta1 - theta) / dt / np.tan(theta))


def flow_errors(o, ex):
    # interior of the omega-sphere away from the poles of the flow (where sin alpha -> 0)
    ok = (o["dim"] == M) & (np.abs(ex["h"]) < 0.9)
    T_tr = M
    scale = np.sqrt(np.mean((T_tr * (ex["flow"] + ex["normal"]))[ok] ** 2) + np.mean((T_tr * ex["flow"][ok]) ** 2))
    eflow = np.sqrt(np.mean((o["E_flow_trace"] - T_tr * ex["flow"])[ok] ** 2)) / scale
    enorm = np.sqrt(np.mean((o["E_normal_trace"] - T_tr * ex["normal"])[ok] ** 2)) / scale
    etot = np.sqrt(np.mean((o["E_trace"] - T_tr * (ex["flow"] + ex["normal"]))[ok] ** 2)) / scale
    st = o["stretches"][ok, :M]
    estr = np.median(np.abs(st / ex["s"][ok, None] - 1))
    return dict(flow=eflow, normal=enorm, total=etot, stretch=estr, skew=np.median(o["E_normal_skew"][ok]) / scale)


@pytest.mark.parametrize("thetadot", [0.0, 1.0])
def test_flow_terms(thetadot):
    """thetadot = 0: pure tangential flow on the fixed support; thetadot = 1: mixed, partly cancelling."""
    theta, c = 0.9, 1.5
    rows = []
    for dt in (0.02, 0.01):
        o, ex = flow_case(10000, theta, theta + thetadot * dt, c, dt, tau=0.003, k=160)
        rows.append((dt, flow_errors(o, ex)))
        print("flow thetadot", thetadot, dt, {k: round(float(v), 4) for k, v in rows[-1][1].items()})
    last = rows[-1][1]
    # rate terms converge as dt -> 0 (first-order step error) and reach the tolerance
    for key in ("flow", "normal", "total"):
        assert rows[-1][1][key] <= rows[0][1][key] + 2e-3, key
        assert last[key] < 0.03, key
    # finite stretches are exact for any step
    for dt, e in rows:
        assert e["stretch"] < 3e-3
    if thetadot:
        # the two terms are large and partly cancel: the fixture is informative
        ok = o["dim"] == M
        cancel = np.mean(np.abs(ex["flow"] + ex["normal"])[ok]) / np.mean((np.abs(ex["flow"]) + np.abs(ex["normal"]))[ok])
        assert cancel < 0.7
