"""Generate fictitious daily OHLCV data for execution checks; never vendor data."""

import argparse
import csv
import math
from datetime import date, timedelta
from pathlib import Path
import random


def generate(rows=1209, seed=42):
    rng = random.Random(seed)
    current_date = date(2020, 1, 2)
    previous_close = 100.0
    records = []
    while len(records) < rows:
        if current_date.weekday() < 5:
            # Weekday dates are fictitious sessions; this is not an exchange calendar.
            open_price = previous_close * math.exp(rng.gauss(0.0, 0.004))
            close_price = open_price * math.exp(rng.gauss(0.0001, 0.012))
            high = max(open_price, close_price) * math.exp(abs(rng.gauss(0.0, 0.004)))
            low = min(open_price, close_price) * math.exp(-abs(rng.gauss(0.0, 0.004)))
            volume = rng.lognormvariate(math.log(1_000_000), 0.35)
            records.append({
                'time': current_date.isoformat(), 'open_price': open_price,
                'pre_close_price': previous_close, 'high_price': high,
                'low_price': low, 'close_price_nvnf': close_price, 'volume': volume,
            })
            previous_close = close_price
        current_date += timedelta(days=1)
    return records


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=project / 'data/private/synthetic_daily.csv')
    parser.add_argument('--rows', type=int, default=1209)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.rows < 750:
        parser.error('--rows must be at least 750 for the notebook demonstration')
    records = generate(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    except FileExistsError:
        parser.error('Output already exists; choose a different --output. Existing data was preserved.')
    print(f'Created {len(records)} SYNTHETIC rows. These are not market observations.')


if __name__ == '__main__':
    main()
