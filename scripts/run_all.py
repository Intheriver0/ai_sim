"""Run the current methodology runners sequentially with one log and result bundle."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('mlp', 'tabm', 'hyper_head', 'maml', 'mlp_sgd',
          'mlp_hypergradient', 'mlp_learned_optimizer')
METHODS = ('fixed', 'random', 'tpe', 'gp', 'asha', 'hyperband',
           'tpe_hyperband', 'smac', 'smac_mf', 'pbt')
EXCLUDED = {
    'dehb': 'Adapter exists but the production runner blocks it.',
    'bohb': 'No runnable adapter; installation was blocked by build prerequisites.',
    'p-diff / GHN / NNiT': 'Not implemented.',
    'full-trajectory hypergradient': 'Only the one-step variant is implemented.',
    'mlp_hypergradient + pbt': 'PBT conflicts with the learned learning rates.',
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/all_methods_v1')
    p.add_argument('--data-dir', type=Path, default=ROOT/'data/preprocessed_v1')
    p.add_argument('--regions-dir', type=Path, default=ROOT/'data/ridge_regions_v1')
    p.add_argument('--terrain-dir', type=Path, default=ROOT/'data/terrain_features_v2')
    p.add_argument('--config', type=Path, default=ROOT/'configs/seasonal_splits_v2.json')
    p.add_argument('--results-root', type=Path, default=ROOT/'results')
    p.add_argument('--error-run', type=Path)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def jobs(args):
    common = ['--data-dir', str(args.data_dir), '--config', str(args.config),
              '--threads', str(args.threads)]
    plans = []

    def add(name, script, options):
        output = args.output_dir/name
        command = [sys.executable, '-u', str(ROOT/'scripts'/script),
                   '--output-dir', str(output), *common, *options]
        plans.append(dict(name=name, output=str(output), command=command))

    add('literature', 'run_literature.py', ['--regions-dir', str(args.regions_dir)])
    for model in MODELS:
        methods = [m for m in METHODS if not (model == 'mlp_hypergradient' and m == 'pbt')]
        add('parameter_'+model, 'run_parameter_methods.py',
            ['--regions-dir', str(args.regions_dir), '--terrain-dir', str(args.terrain_dir),
             '--model', model, '--methods', *methods])
    options = ['--terrain-dir', str(args.terrain_dir), '--results-root', str(args.results_root)]
    if args.error_run:
        options += ['--error-run', str(args.error_run)]
    add('pipeline', 'run_pipeline.py', options)
    return plans


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def collect(job):
    artifacts = {}
    for name in ('preflight.json', 'comparison.json', 'fold_metrics.json'):
        path = Path(job['output'])/name
        if path.exists():
            try:
                artifacts[name] = json.loads(path.read_text(encoding='utf-8'))
            except (ValueError, OSError) as error:
                artifacts[name] = {'read_error': str(error)}
    return artifacts


def save_report(output, report):
    write_json(output/'results.json', report)
    lines = ['# 전체 방법 실행 결과', '',
             '평가 절차별 결과를 합쳤으며 절차 간 통합 순위나 제출 모델을 선택하지 않습니다.', '',
             f"상태: {report['status']}", '', '| 실행 | 상태 | 초 |', '|---|---|---:|']
    for job in report['jobs']:
        lines.append(f"| {job['name']} | {job['status']} | {job.get('seconds', 0):.1f} |")
    lines += ['', '## 제외 항목', '']
    lines += [f'- {name}: {reason}' for name, reason in EXCLUDED.items()]
    for job in report['jobs']:
        lines += ['', f"## {job['name']}", '',
                  '실패·중단된 실행의 기존 결과는 부분 결과일 수 있습니다.', '',
                  '```json', json.dumps(job.get('artifacts', {}), ensure_ascii=False, indent=2), '```']
    (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def launch(command, emit):
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as child:
        try:
            for line in child.stdout:
                emit(line.rstrip('\r\n'))
            return child.wait()
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise


def run(args, execute=launch):
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.threads < 1 or ROOT/'results' not in args.output_dir.parents:
        raise ValueError('Use positive threads and a dedicated directory under ai_sim/results')
    plans = jobs(args)
    if args.dry_run:
        print(json.dumps({'jobs': plans, 'excluded': EXCLUDED}, ensure_ascii=False, indent=2))
        return 0
    manifest = {'jobs': plans, 'sources': {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'scripts').glob('*.py')}}
    output = args.output_dir
    if output.exists():
        if not args.resume:
            raise FileExistsError('Use --resume with unchanged settings')
        if json.loads((output/'manifest.json').read_text(encoding='utf-8')) != manifest:
            raise ValueError('Commands or code changed; use a new output directory')
    else:
        output.mkdir(parents=True)
        write_json(output/'manifest.json', manifest)
    report = {'status': 'running', 'preflight_only': args.preflight_only,
              'excluded': EXCLUDED, 'jobs': [dict(j, status='pending') for j in plans]}
    save_report(output, report)
    with (output/'run.log').open('a', encoding='utf-8', buffering=1) as log:
        def emit(name, message):
            line = f'{datetime.now(timezone.utc).isoformat()} [{name}] {message}'
            print(line, flush=True)
            log.write(line+'\n')

        for job in report['jobs']:
            command = list(job['command'])
            # Re-enter completed runners so their input/version/cache checks still run.
            if args.resume and (Path(job['output'])/'manifest.json').exists():
                command.append('--resume')
            if job['name'] == 'pipeline':
                command += ['--stage', 'preflight' if args.preflight_only else 'compare']
            elif args.preflight_only:
                command.append('--preflight-only')
            job.update(status='running', executed_command=command)
            save_report(output, report)
            emit(job['name'], 'START '+json.dumps(command, ensure_ascii=False))
            started = time.monotonic()
            interrupted = False
            try:
                job['returncode'] = execute(command, lambda line: emit(job['name'], line))
                job['artifacts'] = collect(job)
                required = 'preflight.json' if args.preflight_only else 'comparison.json'
                complete = job['artifacts'].get(required, {}).get('status') == 'complete'
                job['status'] = ('preflight_complete' if args.preflight_only else 'complete') if (
                    job['returncode'] == 0 and complete) else 'failed'
            except KeyboardInterrupt:
                job['status'] = 'interrupted'
                interrupted = True
            except Exception as error:
                job.update(status='failed', error=str(error))
                emit(job['name'], str(error))
            job['seconds'] = time.monotonic()-started
            job['artifacts'] = collect(job)
            emit(job['name'], 'END '+job['status'])
            if interrupted:
                report['status'] = 'interrupted'
                save_report(output, report)
                return 130
            save_report(output, report)
        success = all(j['status'] in ('complete', 'preflight_complete') for j in report['jobs'])
        report['status'] = ('preflight_complete' if args.preflight_only else 'complete') if success else 'failed'
        save_report(output, report)
        emit('suite', report['status'])
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(run(parser().parse_args()))
