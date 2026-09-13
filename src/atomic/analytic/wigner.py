
import math
from functools import lru_cache

__all__ = ["triangular", "wigner_3j", "wigner_6j"]

_CACHE_SIZE = 4096

def _doubled(j: float, name: str) -> int:
    two_j = round(2 * j)
    if abs(2 * j - two_j) > 1e-9:
        raise ValueError(f"{name} must be integer or half-integer, got {j}")
    if two_j < 0:
        raise ValueError(f"{name} must be non-negative, got {j}")
    return two_j

def _triangular_doubled(a2: int, b2: int, c2: int) -> bool:
    if (a2 + b2 + c2) % 2 != 0:
        return False
    return abs(a2 - b2) <= c2 <= a2 + b2

def triangular(a: float, b: float, c: float) -> bool:
    return _triangular_doubled(_doubled(a, "a"), _doubled(b, "b"), _doubled(c, "c"))

def _delta(a2: int, b2: int, c2: int) -> float:
    return math.sqrt(
        math.factorial((a2 + b2 - c2) // 2)
        * math.factorial((a2 - b2 + c2) // 2)
        * math.factorial((-a2 + b2 + c2) // 2)
        / math.factorial((a2 + b2 + c2) // 2 + 1)
    )

@lru_cache(maxsize=_CACHE_SIZE)
def wigner_6j(
    j1: float, j2: float, j3: float, j4: float, j5: float, j6: float
) -> float:
    a = [_doubled(j, n) for j, n in
         ((j1, "j1"), (j2, "j2"), (j3, "j3"), (j4, "j4"), (j5, "j5"), (j6, "j6"))]
    j1_, j2_, j3_, j4_, j5_, j6_ = a

    triads = ((j1_, j2_, j3_), (j1_, j5_, j6_), (j4_, j2_, j6_), (j4_, j5_, j3_))
    if not all(_triangular_doubled(*t) for t in triads):
        return 0.0

    prefactor = 1.0
    for t in triads:
        prefactor *= _delta(*t)

    lower = max(sum(t) for t in triads)
    upper = min(
        j1_ + j2_ + j4_ + j5_,
        j2_ + j3_ + j5_ + j6_,
        j1_ + j3_ + j4_ + j6_,
    )
    total = 0.0
    for t2 in range(lower, upper + 1, 2):
        t = t2 // 2
        denom = (
            math.factorial(t - sum(triads[0]) // 2)
            * math.factorial(t - sum(triads[1]) // 2)
            * math.factorial(t - sum(triads[2]) // 2)
            * math.factorial(t - sum(triads[3]) // 2)
            * math.factorial((j1_ + j2_ + j4_ + j5_) // 2 - t)
            * math.factorial((j2_ + j3_ + j5_ + j6_) // 2 - t)
            * math.factorial((j1_ + j3_ + j4_ + j6_) // 2 - t)
        )
        total += (-1.0) ** t * math.factorial(t + 1) / denom

    return prefactor * total

def _doubled_m(m: float, name: str) -> int:
    two_m = round(2 * m)
    if abs(2 * m - two_m) > 1e-9:
        raise ValueError(f"{name} must be integer or half-integer, got {m}")
    return two_m

@lru_cache(maxsize=_CACHE_SIZE)
def wigner_3j(
    j1: float, j2: float, j3: float, m1: float, m2: float, m3: float
) -> float:
    j1_, j2_, j3_ = (_doubled(j, n) for j, n in ((j1, "j1"), (j2, "j2"), (j3, "j3")))
    m1_, m2_, m3_ = (_doubled_m(m, n) for m, n in ((m1, "m1"), (m2, "m2"), (m3, "m3")))

    if m1_ + m2_ + m3_ != 0:
        return 0.0
    for j, m in ((j1_, m1_), (j2_, m2_), (j3_, m3_)):
        if abs(m) > j or (j - m) % 2 != 0:
            return 0.0
    if not _triangular_doubled(j1_, j2_, j3_):
        return 0.0

    prefactor = _delta(j1_, j2_, j3_)
    for j, m in ((j1_, m1_), (j2_, m2_), (j3_, m3_)):
        prefactor *= math.sqrt(
            math.factorial((j + m) // 2) * math.factorial((j - m) // 2)
        )

    lower = max(0, -((j3_ - j2_ + m1_) // 2), -((j3_ - j1_ - m2_) // 2))
    upper = min(
        (j1_ + j2_ - j3_) // 2,
        (j1_ - m1_) // 2,
        (j2_ + m2_) // 2,
    )
    total = 0.0
    for t in range(lower, upper + 1):
        denom = (
            math.factorial(t)
            * math.factorial((j3_ - j2_ + m1_) // 2 + t)
            * math.factorial((j3_ - j1_ - m2_) // 2 + t)
            * math.factorial((j1_ + j2_ - j3_) // 2 - t)
            * math.factorial((j1_ - m1_) // 2 - t)
            * math.factorial((j2_ + m2_) // 2 - t)
        )
        total += (-1.0) ** t / denom

    sign = (-1.0) ** ((j1_ - j2_ - m3_) // 2)
    return sign * prefactor * total
