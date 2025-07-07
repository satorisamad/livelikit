import csv
from statistics import mean
from pathlib import Path

METRICS_PATH = Path(__file__).with_suffix("").parent / "metrics.csv"

if not METRICS_PATH.exists():
    print("metrics.csv not found. Interact with the agent first to generate it.")
    exit(1)

latencies = []
rows = []
with METRICS_PATH.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)
        try:
            latencies.append(float(row.get("latency_ms", 0)))
        except ValueError:
            pass

if not rows:
    print("No data in metrics.csv yet.")
    exit(0)

# Print per-turn summary
title = f"{'Timestamp':<26} | {'Latency (ms)':>12} | User Transcript => Agent Response"
print(title)
print("-" * len(title))
for r in rows:
    ts = r.get('timestamp', '')[:26]
    lat = r.get('latency_ms', 'N/A').rjust(12)
    print(f"{ts} | {lat} | {r.get('user_transcript', '').strip()} => {r.get('agent_response', '').strip()}")

# Stats
if latencies:
    print("\nLatency stats (ms):")
    print(f"  Count : {len(latencies)}")
    print(f"  Avg   : {mean(latencies):.2f}")
    print(f"  Min   : {min(latencies):.2f}")
    print(f"  Max   : {max(latencies):.2f}")
