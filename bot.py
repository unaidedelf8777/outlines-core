import argparse
import collections
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import requests

BASE = "https://berghain.challenges.listenlabs.ai"

# ----------------------- Data containers -----------------------

class Constraints:
    def __init__(self, items: List[Dict]):
        # items: [{"attribute": "attrId", "minCount": int}, ...]
        self.min_by_attr: Dict[str, int] = {it["attribute"]: int(it["minCount"]) for it in items}
        self.progress: Dict[str, int] = {attr: 0 for attr in self.min_by_attr}

    def remaining_deficit(self, attr: str) -> int:
        return max(self.min_by_attr.get(attr, 0) - self.progress.get(attr, 0), 0)

    def total_remaining_deficit(self) -> int:
        return sum(self.remaining_deficit(a) for a in self.min_by_attr)

    def record_accept(self, attributes_present: Dict[str, bool]) -> None:
        for a, need in self.min_by_attr.items():
            if attributes_present.get(a, False):
                self.progress[a] = self.progress.get(a, 0) + 1


class OnlineFrequencies:
    """Blend server-provided frequencies with online EMA from observed candidates."""
    def __init__(self, server_freqs: Dict[str, float], alpha: float = 0.05):
        self.alpha = alpha
        self.freq = dict(server_freqs)  # attributeId -> prob (0..1)
        self.counts = collections.Counter()  # raw observed counts
        self.total_seen = 0

    def observe(self, attrs: Dict[str, bool]) -> None:
        self.total_seen += 1
        for a, v in attrs.items():
            if v:
                self.counts[a] += 1

        # EMA update for attributes we know about
        for a in set(list(self.freq.keys()) + list(attrs.keys())):
            p_hat = (self.counts[a] / self.total_seen) if self.total_seen > 0 else 0.0
            old = self.freq.get(a, 0.0)
            self.freq[a] = (1 - self.alpha) * old + self.alpha * p_hat

    def p(self, a: str) -> float:
        return max(min(self.freq.get(a, 0.0), 1.0), 1e-6)  # clamp to avoid div-by-zero


# ----------------------- API helpers -----------------------

def new_game(scenario: int, player_id: str):
    r = requests.get(f"{BASE}/new-game", params={"scenario": scenario, "playerId": player_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def decide_and_next(game_id: str, person_index: int, accept: Optional[bool] = None):
    params = {"gameId": game_id, "personIndex": person_index}
    if person_index > 0 or accept is not None:
        params["accept"] = "true" if accept else "false"
    r = requests.get(f"{BASE}/decide-and-next", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


# ----------------------- Policies -----------------------

def compute_weights(constraints: Constraints, freqs: OnlineFrequencies, slots_remaining: int) -> Dict[str, float]:
    """
    Dualish weights per attribute: w_a = (deficit_a / slots_remaining) * (1 / p_a)
    Meaning: prioritize large deficits; pay a rarity premium for low-frequency attributes.
    """
    slots_remaining = max(slots_remaining, 1)
    w = {}
    for a in constraints.min_by_attr:
        deficit = constraints.remaining_deficit(a)
        if deficit <= 0:
            w[a] = 0.0
            continue
        pa = freqs.p(a)
        w[a] = (deficit / slots_remaining) * (1.0 / pa)
    return w

def must_take_mask(constraints: Constraints, slots_remaining: int) -> Dict[str, bool]:
    """
    Hard feasibility guard:
      - If deficit_a == slots_remaining, then EVERY remaining admission must have 'a'.
        Enforce a must_have[a] = True mask; if a candidate lacks such 'a', reject.
    """
    mask = {}
    for a in constraints.min_by_attr:
        if constraints.remaining_deficit(a) >= slots_remaining:
            mask[a] = True
    return mask

def accept_rule(attributes_present: Dict[str, bool],
                constraints: Constraints,
                freqs: OnlineFrequencies,
                admitted_count: int,
                capacity: int) -> bool:
    slots_remaining = max(capacity - admitted_count, 0)

    # Hard feasibility: if we're in a must-take regime for any attribute, enforce it.
    mt = must_take_mask(constraints, slots_remaining)
    if any(mt.values()):
        for a, must in mt.items():
            if must and not attributes_present.get(a, False):
                return False  # cannot afford to admit lacking 'a'
        # all required must-take attributes present → accept
        return True

    # Compute soft weights
    w = compute_weights(constraints, freqs, slots_remaining)

    # Score candidate by sum of contributing weights (diminishing returns via sqrt on sum)
    raw = 0.0
    for a, weight in w.items():
        if weight <= 0:
            continue
        if attributes_present.get(a, False):
            raw += weight

    # Dynamic threshold: start stricter, relax as we fill
    fill = admitted_count / max(capacity, 1)
    # Base threshold tuned empirically; you can tweak these
    base = 0.25
    min_th = 0.03
    threshold = max(min_th, base * (1.0 - fill))

    return raw >= threshold


# ----------------------- Runners -----------------------

def run_game(scenario: int,
             player_id: str,
             policy_mode: str = "calibrate",
             ema_alpha: float = 0.05,
             capacity_hint: int = 1000,
             max_steps: int = 25000) -> Dict:
    """
    Runs one game. For 'calibrate', be stingy to sample many; for 'play', use accept_rule.
    Returns a dict with final stats and learned frequencies.
    """
    game = new_game(scenario, player_id)
    game_id = game["gameId"]
    constraints = Constraints(game["constraints"])
    freqs = OnlineFrequencies(game.get("attributeStatistics", {}).get("relativeFrequencies", {}),
                              alpha=ema_alpha)

    # Policy knobs
    if policy_mode == "calibrate":
        # Reject-by-default; accept rare/very helpful sometimes to keep progress realistic
        acceptor = lambda attrs, adm: _calibration_accept(attrs, constraints, freqs, adm, capacity_hint)
    else:
        acceptor = lambda attrs, adm: accept_rule(attrs, constraints, freqs, adm, capacity_hint)

    person_index = 0
    last_status = None
    start = time.time()
    while person_index < max_steps:
        # For the first person, accept is optional. For others, we pass decision.
        if person_index == 0:
            resp = decide_and_next(game_id, person_index)
        else:
            # Decide based on current state
            attrs = next_attrs  # from previous loop
            freqs.observe(attrs)
            take = acceptor(attrs, admitted_count)
            resp = decide_and_next(game_id, person_index, accept=take)
            if take:
                constraints.record_accept(attrs)

        status = resp["status"]
        last_status = status
        admitted_count = resp.get("admittedCount", 0)
        rejected_count = resp.get("rejectedCount", 0)

        if status == "completed":
            break
        if status == "failed":
            break

        nextp = resp.get("nextPerson")
        if not nextp:
            break

        person_index = nextp["personIndex"]
        next_attrs = nextp["attributes"]

        # Stopping guard: if we've basically filled to capacity_hint, keep going but protect from runaway loops
        if admitted_count >= capacity_hint and policy_mode == "calibrate":
            # In calibration we prefer more observations; keep going unless the API completes.
            pass

    elapsed = time.time() - start
    return {
        "mode": policy_mode,
        "scenario": scenario,
        "status": last_status,
        "admitted": admitted_count,
        "rejected": rejected_count,
        "constraints_progress": dict(constraints.progress),
        "server_freqs": game.get("attributeStatistics", {}).get("relativeFrequencies", {}),
        "learned_freqs": freqs.freq,
        "elapsed_sec": elapsed,
    }

def _calibration_accept(attrs: Dict[str, bool],
                        constraints: Constraints,
                        freqs: OnlineFrequencies,
                        admitted_count: int,
                        capacity: int) -> bool:
    """Very stingy: accept only if rare AND helpful, to maximize observations early."""
    slots_remaining = max(capacity - admitted_count, 1)
    # Take only if reduces >=2 active deficits or if some present attribute has p < 0.02
    active_help = sum(1 for a in constraints.min_by_attr if constraints.remaining_deficit(a) > 0 and attrs.get(a, False))
    rare_present = any(freqs.p(a) < 0.02 and attrs.get(a, False) for a in constraints.min_by_attr)
    must_take = must_take_mask(constraints, slots_remaining)
    if any(must_take.values()):
        for a, must in must_take.items():
            if must and not attrs.get(a, False):
                return False
        return True  # satisfies all must-take attributes
    return active_help >= 2 or rare_present


# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser(description="Berghain API Bot: calibrate once, then play each scenario to win.")
    ap.add_argument("--player-id", required=True, help="UUID shown in /new-game link (playerId=...)")
    ap.add_argument("--scenarios", default="1,2,3", help="Comma list of scenarios to run")
    ap.add_argument("--ema-alpha", type=float, default=0.05, help="EMA blending rate for online frequency estimates")
    ap.add_argument("--capacity", type=int, default=1000, help="Capacity hint (default 1000)")
    ap.add_argument("--max-steps", type=int, default=30000, help="Max steps per game as a safety valve")
    args = ap.parse_args()

    scenarios = [int(s.strip()) for s in args.scenarios.split(",") if s.strip()]

    results = []
    for s in scenarios:
        print(f"\n=== Scenario {s}: CALIBRATION RUN ===")
        cal = run_game(s, args.player_id, policy_mode="calibrate",
                       ema_alpha=args.ema_alpha, capacity_hint=args.capacity, max_steps=args.max_steps)
        print(f"Calib: status={cal['status']} admitted={cal['admitted']} rejected={cal['rejected']} observed_p~{len(cal['learned_freqs'])} attrs")

        print(f"\n=== Scenario {s}: PLAY TO WIN ===")
        win = run_game(s, args.player_id, policy_mode="play",
                       ema_alpha=args.ema_alpha, capacity_hint=args.capacity, max_steps=args.max_steps)
        print(f"Win:   status={win['status']} admitted={win['admitted']} rejected={win['rejected']}")

        results.append((cal, win))

    # Compact summary
    print("\n=== Summary ===")
    for (cal, win) in results:
        print(f"Scenario {cal['scenario']}: rejected calibration={cal['rejected']}, rejected win={win['rejected']} (status={win['status']})")


if __name__ == "__main__":
    main()
