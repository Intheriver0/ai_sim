"""Audit source coverage and the union of v2 training/evaluation roles."""
import argparse
import json
from pathlib import Path

from pipeline_common import connect, read_view
from seasonal_splits import load_plan
from time_splits import digest, interval, stamp


def records(con, sql):
    result = con.execute(sql)
    names = [r[0] for r in result.description]
    return [dict(zip(names, row)) for row in result.fetchall()]


def audit(source, config, output):
    cfg, outers = load_plan(config)
    source, output = Path(source), Path(output)
    with connect() as c:
        files = []
        for path in sorted(source.rglob('*')):
            if path.suffix not in ('.csv','.parquet'):
                continue
            read_view(c,'source_file',path)
            columns = [r[0] for r in c.execute('DESCRIBE source_file').fetchall()]
            timing = ',min(ts) AS start,max(ts) AS end' if 'ts' in columns else ''
            files.append({'file':path.relative_to(source).as_posix(),
                          **records(c,f'SELECT count(*) AS rows {timing} FROM source_file')[0]})
        read_view(c,'generation',source/'train/generation.parquet')
        read_view(c,'prices',source/'train/smp.csv')
        first, last = outers[0], outers[-1]
        inner = ' OR '.join('('+interval(w['start'],w['end'])+')' for o in outers for w in o['inner'])
        c.execute(f'''CREATE TEMP TABLE calendar AS SELECT p.ts,
            CAST(date_trunc('month',p.ts-INTERVAL '1 hour') AS DATE) AS month,
            coalesce(g.n,0) AS sites,coalesce(g.negative,0) AS negative,
            ({interval(first['start'],last['end'])}) AS outer_eval,
            (ts >= {stamp(first['training_start'])} AND ts <= {stamp(last['training_cutoff'])}) AS outer_train,
            ({inner}) AS inner_eval
            FROM prices p LEFT JOIN (SELECT ts,count(*) AS n,count(*) FILTER(WHERE energy<0) AS negative
                FROM generation GROUP BY ts) g USING(ts)''')
        monthly = records(c,'''SELECT month,count(*) AS hours,min(sites) AS min_sites,max(sites) AS max_sites,
            sum(sites) AS observed_site_hours, sum(negative) AS negative_rows,
            round(100.0*sum(sites)/(count(*)*253),2) AS observed_pct,
            count(*) FILTER(WHERE sites=253) AS complete_hours,
            sum(CASE WHEN outer_train THEN sites-negative ELSE 0 END) AS train_rows_union,
            sum(CASE WHEN inner_eval THEN sites ELSE 0 END) AS inner_rows_union,
            sum(CASE WHEN outer_eval THEN sites ELSE 0 END) AS outer_rows,
            round(100.0*sum(CASE WHEN outer_train THEN sites-negative ELSE 0 END)/nullif(sum(sites),0),2) AS train_pct,
            round(100.0*sum(CASE WHEN inner_eval THEN sites ELSE 0 END)/nullif(sum(sites),0),2) AS inner_pct,
            round(100.0*sum(CASE WHEN outer_eval THEN sites ELSE 0 END)/nullif(sum(sites),0),2) AS outer_pct
            FROM calendar GROUP BY month ORDER BY month''')
        distinct = dict(c.execute("SELECT CAST(date_trunc('month',ts-INTERVAL '1 hour') AS DATE),count(DISTINCT site_id) FROM generation GROUP BY 1").fetchall())
        for m in monthly:
            m['distinct_sites'] = distinct.get(m['month'],0)
        yearly = records(c,'''SELECT year(month) AS year,count(*) AS hours,min(sites) AS min_sites,max(sites) AS max_sites,
            sum(sites) AS observed_site_hours,round(100.0*sum(sites)/(count(*)*253),2) AS observed_pct
            FROM calendar GROUP BY 1 ORDER BY 1''')
        folds = []
        for o in outers:
            n = c.execute(f'''SELECT count(*) FROM generation WHERE energy>=0
                AND ts>={stamp(o['training_start'])} AND ts<={stamp(o['training_cutoff'])}''').fetchone()[0]
            evaluation = c.execute(f'SELECT count(*) FROM generation WHERE {interval(o["start"],o["end"])}').fetchone()[0]
            folds.append({'month': o['start'].strftime('%Y-%m'), 'cutoff':o['training_cutoff'],
                          'training_rows':n,'evaluation_rows':evaluation,
                          'training_pct':round(100*n/(n+evaluation),2),
                          'evaluation_pct':round(100*evaluation/(n+evaluation),2)})
        totals = records(c,'''SELECT count(*) AS hours,sum(sites) AS observed_rows,sum(negative) AS negative_rows,
            sum(CASE WHEN outer_eval THEN sites ELSE 0 END) AS outer_rows,
            count(*) FILTER(WHERE outer_eval) AS outer_hours,
            count(*) FILTER(WHERE sites=253) AS complete_hours FROM calendar''')[0]
        assert totals['observed_rows']==c.execute('SELECT count(*) FROM generation').fetchone()[0]
        assert totals['outer_hours']==8760
        assert sum(f['evaluation_rows'] for f in folds)==totals['outer_rows']
    report = {'config':cfg['version'],'config_sha256':digest(config),'sources':files,'totals':totals,
              'monthly':monthly,'yearly':yearly,'outer_folds':folds}
    output.mkdir(parents=True,exist_ok=True)
    (output/'data_usage_audit.json').write_text(json.dumps(report,default=str,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


def markdown(report):
    lines = ['# 전체 데이터 기간·발전소 실적·학습/평가 사용 현황', '',
        '2026-10-27 원본 전체 재집계. 현재 승인된 `seasonal_splits_v2` 기준이며 실제 모델 학습은 아직 실행하지 않았다.', '',
        '## 제공 파일 기간', '', '| 파일 | 행 수 | 최초 시각 KST | 마지막 시각 KST |', '|---|---:|---|---|']
    for f in report['sources']:
        lines.append(f"| {f['file']} | {f['rows']:,} | {f.get('start','—')} | {f.get('end','—')} |")
    lines += ['', '## 월별 발전소 실적과 사용 비율', '',
        '- 월은 대상일 기준이다. 예: 2025년 5월은 5월 1일 01시~6월 1일 00시다.',
        '- `실적 개소`는 그 달 한 번 이상 실적이 있는 발전소 수, `시각별 개소`는 시간당 최소~최대다. 참여 계약 상태가 아니라 데이터 보유 현황이다.',
        '- `관측률` = 실제 발전소·시간 행 / (해당 월 가격 시간 수 × 253). 최초 2021년 1월도 가격 시간축 744시간을 분모로 쓴다.',
        '- `학습 사용`, `내부 평가`, `외부 평가`는 해당 월 실제 발전량 행을 분모로 한 비율이다. 각 용도로 한 번 이상 쓰이는 행의 합집합이며 반복 사용 횟수는 세지 않는다.',
        '- 서로 다른 실행에서 과거 평가 자료가 이후 학습 자료가 될 수 있어 세 열의 합은 100%가 아니다. 동일 실행 안의 학습·외부 평가는 겹치지 않는다.',
        '- `학습 사용`은 12개 외부 모델의 재학습 범위 합집합이다. 음수 40행은 학습에서 제외한다. `내부 평가`는 전체 외부 실행의 내부 선택 구간 합집합이다.', '',
        '| 대상월 | 실적 개소 /253 | 시각별 개소 | 관측률 | 전체 실적 시간 | 학습 사용 | 내부 평가 | 외부 평가 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in report['monthly']:
        lines.append(f"| {str(m['month'])[:7]} | {m['distinct_sites']} | {m['min_sites']}~{m['max_sites']} | {m['observed_pct']:.2f}% | {m['complete_hours']}/{m['hours']} | {m['train_pct']:.2f}% | {m['inner_pct']:.2f}% | {m['outer_pct']:.2f}% |")
    lines += ['', '## 외부 평가 실행별 학습/평가 비율', '',
        '각 실행에서 누적 학습 행과 해당 월 외부 평가 행만 합친 분모다. 내부 선택은 이 누적 과거 자료 안에서 수행하므로 별도로 더하지 않는다. 학습은 비음수 원실적, 평가는 원실적 기준이며 라운드별 반복 행은 세지 않는다.', '',
        '| 외부 평가월 | 학습 마감 KST | 학습 행 | 평가 행 | 학습 : 평가 |', '|---|---|---:|---:|---:|']
    for f in report['outer_folds']:
        lines.append(f"| {f['month']} | {f['cutoff']} | {f['training_rows']:,} | {f['evaluation_rows']:,} | {f['training_pct']:.2f}% : {f['evaluation_pct']:.2f}% |")
    lines += ['', '## 해석', '',
        '- 내부 선택은 각 외부 평가월 이전 12개월에 분산한다. 매월 1~10일 모델 선택, 11~20일 혼합, 21일~월말 배분 후보 구간이며 외부 마감 이후 정답은 제외한다.',
        '- 내부 구간의 시간상 분할은 약 1/3씩이지만 월 길이·편입·결측·마감에 따라 실제 행 비율은 달라진다. 외부 실행마다 사용하는 12개월도 이동한다.',
        '- 연중 발전소별 진단은 월별로 동일 가중 평가한다. 표의 데이터 사용 비율과 최종 지표의 월별 가중치는 다른 개념이다.',
        '- 253개소가 모두 있는 3,802시간에서만 전체 집합 정산이 가능하다. 부분 실적 월의 전체 W를 0으로 기록하지 않는다.',
        '- 원본 test 기간(2025-06-01~2026-05-31 대상일)은 예보·SMP만 제공되므로 학습 정답이나 로컬 실측 평가에 사용할 수 없다. 253개소 모두 최종 예측 대상이다.',
        '- 원자료 수치는 [집계 JSON](data/data_usage_audit.json), 분할 규칙은 [v2 안내](seasonal_validation.md), 재집계 코드는 [summarize_data_usage.py](scripts/summarize_data_usage.py)에 있다.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    root = Path(__file__).resolve().parents[1]
    args = parser.parse_args()
    report = audit(args.source,root/'configs/seasonal_splits_v2.json',root/'data')
    (root/'data_usage_summary.md').write_text(markdown(report),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('totals','yearly','outer_folds')},default=str,indent=2))


if __name__ == '__main__':
    main()
