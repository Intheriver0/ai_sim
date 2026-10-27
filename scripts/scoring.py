"""README-formula settlement, not the official portal scoring program."""
import argparse
import json
from pathlib import Path

from pipeline_common import (check_finite, check_keys, check_sites, check_time,
                             connect, prepare_prices, read_view, reject)


def validate_composition(con):
    check_sites(con)
    check_keys(con, 'composition', ['site_id'])
    reject(con, '''SELECT count(*) FROM (
        (SELECT site_id FROM sites EXCEPT SELECT site_id FROM composition)
        UNION ALL
        (SELECT site_id FROM composition EXCEPT SELECT site_id FROM sites))''',
        'composition: missing or unknown site')
    reject(con, '''SELECT count(*) FROM composition
        WHERE "group" IS NULL OR "group" NOT IN ('G1', 'G2')''',
        'composition: groups must be G1 and G2')
    capacities = dict(con.execute('''SELECT "group", sum(capacity_kw)
        FROM composition JOIN sites USING (site_id) GROUP BY "group"''').fetchall())
    if set(capacities) != {'G1', 'G2'} or min(capacities.values()) < 20000 - 1e-6:
        raise ValueError('composition: each group must have at least 20000 kW')
    return capacities


def validate_forecast(con):
    check_keys(con, 'forecast', ['group', 'ts'])
    check_time(con, 'forecast')
    check_finite(con, 'forecast', 'forecast_kwh', nonnegative=True)
    con.execute('''CREATE OR REPLACE TEMP VIEW expected_forecast AS
        SELECT "group", ts FROM (VALUES ('G1'), ('G2')) g("group") CROSS JOIN prices''')
    reject(con, '''SELECT count(*) FROM (
        (SELECT "group", ts FROM expected_forecast EXCEPT SELECT "group", ts FROM forecast)
        UNION ALL
        (SELECT "group", ts FROM forecast EXCEPT SELECT "group", ts FROM expected_forecast))''',
        'forecast: missing or extra group/timestamp')


def score(con):
    capacities = validate_composition(con)
    validate_forecast(con)
    check_keys(con, 'actual', ['site_id', 'ts'])
    check_time(con, 'actual')
    # Measured energy is preserved, including the delivered negative readings.
    check_finite(con, 'actual', 'energy')
    reject(con, '''SELECT count(*) FROM actual ANTI JOIN sites USING (site_id)''',
           'actual: unknown site')
    reject(con, '''SELECT count(*) FROM
        (SELECT site_id, ts FROM sites CROSS JOIN prices) expected
        ANTI JOIN actual USING (site_id, ts)''',
        'actual: incomplete site coverage; missing generation must not become zero')
    rows = con.execute('''WITH grouped AS (
        SELECT c."group", a.ts, sum(a.energy) AS actual_kwh
        FROM actual a JOIN prices p USING (ts) JOIN composition c USING (site_id)
        GROUP BY c."group", a.ts
    ) SELECT g."group", sum(p.smp * g.actual_kwh) AS revenue,
        sum(0.10 * p.smp * p.k * abs(f.forecast_kwh - g.actual_kwh)) AS penalty
        FROM grouped g JOIN forecast f USING ("group", ts) JOIN prices p USING (ts)
        GROUP BY g."group" ORDER BY g."group"''').fetchall()
    groups = {g: {'capacity_kw': capacities[g], 'revenue_won': revenue,
                  'W_won': penalty, 'S_won': revenue - penalty}
              for g, revenue, penalty in rows}
    return {'basis': 'provided README formula; local validation only', 'groups': groups,
            **{key: sum(v[key] for v in groups.values())
               for key in ('revenue_won', 'W_won', 'S_won')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sites', 'composition', 'forecast', 'prices'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--actual', type=Path, help='Omit to validate submission structure only')
    args = parser.parse_args()
    with connect() as con:
        for name in ('sites', 'composition', 'forecast'):
            read_view(con, name, getattr(args, name))
        read_view(con, 'raw_prices', args.prices)
        prepare_prices(con)
        if args.actual:
            read_view(con, 'actual', args.actual)
            result = score(con)
        else:
            capacities = validate_composition(con)
            validate_forecast(con)
            result = {'valid_structure': True, 'capacity_kw': capacities,
                      'note': 'Local checks only; no score computed'}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
