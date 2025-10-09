from typing import Dict, Tuple

# Simple proportional allocator for green light durations based on observed counts.
# Returns a mapping of class/category to suggested green time in seconds.
def compute_signal_plan(vehicle_counts: Dict[str, int], cycle_seconds: int = 90,
                        min_green: int = 10) -> Dict[str, int]:
    total = sum(max(0, c) for c in vehicle_counts.values())
    if total == 0:
        # Default equal split if no counts yet
        n = max(1, len(vehicle_counts) or 1)
        equal = max(min_green, cycle_seconds // n)
        return {k: equal for k in (vehicle_counts.keys() or ["all"])}

    raw_allocs: Dict[str, float] = {
        k: (max(0, c) / total) * cycle_seconds for k, c in vehicle_counts.items()
    }
    # Enforce minimums and round
    plan_int: Dict[str, int] = {k: max(min_green, int(v)) for k, v in raw_allocs.items()}

    # Normalize to exactly cycle_seconds by adjusting the largest buckets
    diff = cycle_seconds - sum(plan_int.values())
    if diff != 0 and len(plan_int) > 0:
        # sort by count desc to adjust largest categories first if we need to add/remove seconds
        for k, _ in sorted(vehicle_counts.items(), key=lambda x: x[1], reverse=True):
            if diff == 0:
                break
            if diff > 0:
                plan_int[k] += 1
                diff -= 1
            else:
                # avoid dropping below min_green
                if plan_int[k] > min_green:
                    plan_int[k] -= 1
                    diff += 1
    return plan_int


