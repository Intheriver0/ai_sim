import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pipeline_common import connect, export, prepare_prices, read_view
from preprocess import run
from clean_observed import clean
from scoring import score, validate_composition, validate_forecast


def create_sites(con):
    con.execute('''CREATE TABLE sites AS SELECT 'PV' || lpad(i::VARCHAR, 3, '0') AS site_id,
        CASE WHEN i<=126 THEN 'A' ELSE 'B' END AS current_group, 200.0 AS capacity_kw,
        37.0 AS lat, 127.0 AS lon, 'WA001' AS wa_station, 'WB001' AS wb_station,
        100.0 AS wa_dist, 200.0 AS wb_dist FROM range(1, 254) t(i)''')


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.con = connect()
        c = self.con
        create_sites(c)
        c.execute('''CREATE TABLE composition AS SELECT site_id,
            CASE current_group WHEN 'A' THEN 'G1' ELSE 'G2' END AS "group" FROM sites''')
        c.execute('''CREATE TABLE raw_prices AS SELECT * FROM (VALUES
            (TIMESTAMPTZ '2025-01-01 10:00:00+09', 100.0),
            (TIMESTAMPTZ '2025-01-01 11:00:00+09', 200.0)) t(ts, smp)''')
        prepare_prices(c)
        c.execute('CREATE TABLE actual AS SELECT site_id, ts, 1.0::DOUBLE AS energy FROM sites CROSS JOIN prices')
        c.execute('''CREATE TABLE forecast AS SELECT "group", ts,
            CASE "group" WHEN 'G1' THEN 136.0 ELSE 122.0 END::DOUBLE AS forecast_kwh
            FROM (VALUES ('G1'), ('G2')) g("group") CROSS JOIN prices''')

    def tearDown(self):
        self.con.close()

    def test_hand_calculated_settlement_and_cancellation(self):
        result = score(self.con)
        self.assertAlmostEqual(result['revenue_won'], 75900)
        self.assertAlmostEqual(result['W_won'], 1050)
        self.assertAlmostEqual(result['S_won'], 74850)
        self.con.execute("UPDATE actual SET energy=CASE site_id WHEN 'PV001' THEN -4 ELSE 6 END WHERE site_id IN ('PV001', 'PV002')")
        self.assertAlmostEqual(score(self.con)['W_won'], 1050)

    def test_zero_price(self):
        self.con.execute('UPDATE raw_prices SET smp=0 WHERE smp=100')
        self.assertAlmostEqual(score(self.con)['W_won'], 900)

    def test_invalid_inputs(self):
        changes = [
            "DELETE FROM composition WHERE site_id='PV001'",
            'INSERT INTO composition SELECT * FROM composition LIMIT 1',
            "UPDATE composition SET site_id='UNKNOWN' WHERE site_id='PV001'",
            '''UPDATE composition SET "group"='G3' WHERE site_id='PV001' ''',
            '''UPDATE composition SET "group"='G1' ''',
            "UPDATE sites SET capacity_kw=1 WHERE current_group='A'",
            'DELETE FROM forecast WHERE forecast_kwh=136',
            'INSERT INTO forecast SELECT * FROM forecast LIMIT 1',
            'UPDATE forecast SET forecast_kwh=-1',
            "UPDATE forecast SET forecast_kwh='NaN'::DOUBLE",
            "UPDATE forecast SET forecast_kwh='Infinity'::DOUBLE",
            "UPDATE forecast SET ts=ts+INTERVAL '1 day'",
            "DELETE FROM actual WHERE site_id='PV001'",
            'INSERT INTO actual SELECT * FROM actual LIMIT 1',
            'UPDATE actual SET energy=NULL',
        ]
        for sql in changes:
            with self.subTest(sql=sql):
                self.con.execute('BEGIN')
                try:
                    self.con.execute(sql)
                    with self.assertRaises(ValueError):
                        score(self.con)
                finally:
                    self.con.execute('ROLLBACK')

    def test_naive_time_rejected(self):
        self.con.execute('ALTER TABLE forecast ALTER ts TYPE TIMESTAMP')
        with self.assertRaisesRegex(ValueError, 'timezone'):
            validate_forecast(self.con)

    def test_k_boundaries_and_wrong_k(self):
        self.con.execute('''CREATE TABLE boundary AS SELECT ts, 100 AS smp FROM
            unnest([TIMESTAMPTZ '2025-01-01 01:00:00+00', TIMESTAMPTZ '2025-01-01 02:00:00+00',
                    TIMESTAMPTZ '2025-01-01 06:00:00+00', TIMESTAMPTZ '2025-01-01 07:00:00+00']) t(ts)''')
        prepare_prices(self.con, 'boundary', 'boundary_prices')
        self.assertEqual([r[0] for r in self.con.execute('SELECT k FROM boundary_prices ORDER BY ts').fetchall()], [1, 3, 3, 1])
        self.con.execute('ALTER TABLE raw_prices ADD COLUMN k INTEGER DEFAULT 1')
        with self.assertRaisesRegex(ValueError, 'KST hour'):
            prepare_prices(self.con)


class PreprocessTests(unittest.TestCase):
    def test_observed_anomaly_boundaries(self):
        with connect() as c:
            for provider, fields, valid, invalid in (
                ('wa', ['wa_wind_speed'], [50], [9999]),
                ('wb', ['wb_cloud', 'wb_uv_idx', 'wb_vis', 'wb_wind_speed', 'wb_pressure'],
                 [100, 15, 50, 200, 900], [9144, 68, 51, 353, 0]),
            ):
                c.execute(f'''CREATE OR REPLACE TABLE raw AS SELECT '{provider.upper()}001' AS {provider}_station,
                    TIMESTAMPTZ '2025-01-01 01:00:00+09' AS ts,
                    {', '.join(f'{v}::DOUBLE AS {col}' for col, v in zip(fields, valid))}''')
                c.execute(f"INSERT INTO raw VALUES ('{provider.upper()}001', TIMESTAMPTZ '2025-01-01 02:00:00+09', {', '.join(map(str, invalid))})")
                clean(c, 'raw', 'cleaned', provider)
                selected = ', '.join(fields)
                self.assertEqual(c.execute(f'SELECT {selected} FROM cleaned ORDER BY ts').fetchall(),
                                 [tuple(valid), tuple(None for _ in fields)])
                for col in fields:
                    self.assertEqual(c.execute(f'SELECT {col}_invalid FROM cleaned ORDER BY ts').fetchall(), [(False,), (True,)])
                c.execute('INSERT INTO raw SELECT * FROM raw LIMIT 1')
                self.assertEqual(c.execute('SELECT count(*) FROM cleaned WHERE duplicate_key').fetchone()[0], 2)

    def test_fixture_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / 'source', root / 'output'
            (source / 'train').mkdir(parents=True)
            (source / 'test').mkdir()
            with connect() as c:
                create_sites(c)
                c.execute(f"COPY sites TO '{(source / 'sites.csv').as_posix()}' (HEADER)")
                c.execute('''CREATE TABLE times AS SELECT ts, 100 AS smp FROM unnest([
                    TIMESTAMPTZ '2025-01-02 00:00:00+09',
                    TIMESTAMPTZ '2025-01-02 01:00:00+09']) t(ts)''')
                for relative in ('smp.csv', 'train/smp.csv'):
                    c.execute(f"COPY times TO '{(source / relative).as_posix()}' (HEADER)")
                c.execute('''CREATE TABLE generation AS SELECT site_id, ts,
                    CASE WHEN site_id='PV001' AND hour(ts)=0 THEN -1 ELSE 2 END::DOUBLE AS energy
                    FROM sites CROSS JOIN times WHERE NOT (site_id='PV002' AND hour(ts)=1)''')
                export(c, 'SELECT * FROM generation', source / 'train/generation.parquet')
                for split in ('train', 'test'):
                    export(c, '''SELECT 'WA001' AS wa_station, round, ts, 3.6 AS wa_wind_speed,
                        2.0 AS wa_snow, 105.0 AS wa_humidity FROM times CROSS JOIN (VALUES (10),(17)) r(round)
                        WHERE NOT (hour(ts)=0 AND round=10)''', source / split / 'forecast_wa.parquet')
                    export(c, '''SELECT 'WB001' AS wb_station, round, ts, 36.0 AS wb_wind_speed,
                        72.0 AS wb_wind_gust_spd, 2.0 AS wb_snow, NULL::DOUBLE AS wb_rain
                        FROM times CROSS JOIN (VALUES (10),(17)) r(round)
                        UNION ALL SELECT 'WB001', 10, TIMESTAMPTZ '2025-01-02 00:55:00+09', 36,72,2,NULL''',
                           source / split / 'forecast_wb.parquet')
                for provider, col in (('wa', 'wa_wind_speed'), ('wb', 'wb_cloud')):
                    extra = ', 15.0 AS wb_uv_idx, 50.0 AS wb_vis, 200.0 AS wb_wind_speed, 900.0 AS wb_pressure' if provider == 'wb' else ''
                    export(c, f"SELECT '{provider.upper()}001' AS {provider}_station, ts, 9999.0 AS {col}{extra} FROM times",
                           source / 'train' / f'observed_{provider}.parquet')
                report = run(source, output)
                self.assertEqual(report['observed_weather']['outputs']['observed_wa.parquet']['masked_cells']['wa_wind_speed'], 2)
                self.assertEqual(report['generation_rows'], 505)
                self.assertEqual(report['negative_energy_rows'], 1)
                self.assertEqual(report['complete_actual_hours'], 1)
                self.assertEqual(report['outputs']['test_round10.parquet']['rows'], 506)
                self.assertEqual(report['outputs']['test_round10.parquet']['missing_wa_rows'], 253)
                read_view(c, 'features', output / 'test_round17.parquet')
                row = c.execute('''SELECT wb_wind_speed_ms, wb_wind_gust_spd_ms, wb_snow_mm,
                    wa_wind_speed_ms, wa_snow_mm, wa_humidity, wb_rain,
                    target_day::VARCHAR, issued_at::VARCHAR FROM features WHERE hour(ts)=0 LIMIT 1''').fetchone()
                self.assertEqual(row[:7], (10, 20, 20, 3.6, 2, 105, None))
                self.assertEqual(row[7:], ('2025-01-01', '2024-12-31 17:00:00+09'))
                columns = [r[0] for r in c.execute('DESCRIBE features').fetchall()]
                self.assertNotIn('smp', columns)
                self.assertNotIn('energy', columns)
                read_view(c, 'targets', output / 'train_targets.parquet')
                self.assertEqual(c.execute('''SELECT energy_raw_kwh, target_kwh, training_eligible
                    FROM targets WHERE negative_energy''').fetchone(), (-1, None, False))
                self.assertEqual(json.loads((output / 'report.json').read_text())['status'], 'complete')
                with self.assertRaises(FileExistsError):
                    run(source, output)


if __name__ == '__main__':
    unittest.main()
