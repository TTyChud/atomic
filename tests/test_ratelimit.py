
import pytest

from atomic.server.ratelimit import DEFAULT_CAPACITY, DEFAULT_PERIOD, TokenBucket


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds

def test_a_full_burst_is_allowed_then_the_next_request_is_not():
    clock = FakeClock()
    bucket = TokenBucket(capacity=5, period=10.0, clock=clock)
    assert [bucket.check("a") for _ in range(5)] == [None] * 5
    wait = bucket.check("a")
    assert wait is not None
    assert wait == pytest.approx(2.0)

def test_refill_is_continuous_and_proportional():
    clock = FakeClock()
    bucket = TokenBucket(capacity=4, period=8.0, clock=clock)
    for _ in range(4):
        bucket.check("a")
    clock.advance(2.0)
    assert bucket.check("a") is None
    assert bucket.check("a") is not None

def test_clients_do_not_share_a_bucket():
    clock = FakeClock()
    bucket = TokenBucket(capacity=2, period=10.0, clock=clock)
    assert bucket.check("a") is None
    assert bucket.check("a") is None
    assert bucket.check("a") is not None
    assert bucket.check("b") is None

def test_capacity_is_a_ceiling_so_idle_time_does_not_bank_credit():
    clock = FakeClock()
    bucket = TokenBucket(capacity=3, period=3.0, clock=clock)
    clock.advance(10_000.0)
    assert [bucket.check("a") for _ in range(3)] == [None] * 3
    assert bucket.check("a") is not None

def test_hammering_a_refused_endpoint_does_not_extend_the_wait():
    clock = FakeClock()
    bucket = TokenBucket(capacity=1, period=4.0, clock=clock)
    assert bucket.check("a") is None
    clock.advance(2.0)
    first = bucket.check("a")
    second = bucket.check("a")
    assert first == pytest.approx(2.0)
    assert second == pytest.approx(2.0)
    clock.advance(2.0)
    assert bucket.check("a") is None

def test_the_client_table_is_bounded():
    clock = FakeClock()
    bucket = TokenBucket(capacity=1, period=1.0, max_clients=10, clock=clock)
    for i in range(50):
        bucket.check(f"client-{i}")
    clock.advance(5.0)
    bucket.check("trigger-the-prune")
    assert len(bucket) <= 10

def test_a_still_limited_client_survives_pruning():
    clock = FakeClock()
    bucket = TokenBucket(capacity=2, period=100.0, max_clients=1, clock=clock)
    bucket.check("heavy")
    bucket.check("heavy")
    for i in range(20):
        bucket.check(f"passer-by-{i}")
    assert bucket.check("heavy") is not None

@pytest.mark.parametrize("capacity,period", [(0, 60.0), (-1, 60.0), (5, 0.0), (5, -1.0)])
def test_nonsense_configuration_is_refused_at_construction(capacity, period):
    with pytest.raises(ValueError):
        TokenBucket(capacity=capacity, period=period)

WIDEST_ROW = 36
JOBS_PER_TILE = 2
HF_SOLVE = 1

def test_default_burst_covers_widest_click_through():
    clock = FakeClock()
    bucket = TokenBucket(clock=clock)

    storm = WIDEST_ROW * JOBS_PER_TILE + HF_SOLVE
    refused = [i for i in range(storm) if bucket.check("reader") is not None]
    assert refused == [], f"refused {len(refused)} of {storm} honest jobs, first at {refused[:1]}"

def test_widening_the_burst_did_not_raise_the_sustained_rate():
    assert DEFAULT_CAPACITY / DEFAULT_PERIOD == pytest.approx(1 / 3)

    clock = FakeClock()
    bucket = TokenBucket(clock=clock)
    for _ in range(DEFAULT_CAPACITY):
        bucket.check("drained")
    assert bucket.check("drained") == pytest.approx(3.0)
