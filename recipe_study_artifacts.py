"""Publish each model with its checkpoint under an aggregation-safe master path."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from artifacts import atomic_write_json


def publish_run(pair, variant, run_id, protocol):
    from kaggle_session import git_environment, remote_url
    state=json.loads((pair/'pair_status.json').read_text())
    relative=Path('calibrated_recipe_study')/protocol/state['family']/f'seed_{state["seed"]}'/state['initialization']/run_id/variant
    branch=f'research/calibrated-recipe/{protocol}/{state["family"]}/{state["seed"]}/{state["initialization"]}/{run_id}/{variant}'
    subprocess.run(['git','check-ref-format','refs/heads/'+branch],check=True)
    with tempfile.TemporaryDirectory(prefix='recipe-run-publish-') as temporary, git_environment() as env:
        root=Path(temporary)
        destination=root/'results'/relative
        destination.mkdir(parents=True)
        included=[]
        for source in sorted((pair/variant).rglob('*')):
            if not source.is_file() or source.is_symlink() or source.suffix=='.tmp':continue
            if source.stat().st_size>=90*1024*1024:
                raise ValueError(f'Run artifact is too large for direct GitHub publishing: {source.name}')
            target=destination/source.relative_to(pair/variant)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            included.append(dict(path=source.relative_to(pair/variant).as_posix(),bytes=source.stat().st_size,
                                 sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        for name in ['pair_status.json','study_protocol.json','numerical_qualification.json',variant+'.log']:
            if (pair/name).exists():shutil.copy2(pair/name,destination/name)
        atomic_write_json(destination/'export_manifest.json',dict(protocol=protocol,branch=branch,
            master_path=relative.as_posix(),checkpoints_included=True,files=included))
        for arguments in [('init','-b',branch),('config','user.name','Research results'),
                          ('config','user.email','research-results@users.noreply.github.com'),
                          ('add','--','results'),('commit','-m','Record study run with checkpoint'),
                          ('push',remote_url('origin'),f'HEAD:refs/heads/{branch}')]:
            subprocess.run(['git',*arguments],cwd=root,env=env,check=True,timeout=300)
    return branch


def aggregate_remote(output, protocol):
    """Fetch independent model branches without merging or overwriting results."""
    prefix=f'refs/heads/research/calibrated-recipe/{protocol}/'
    lines=subprocess.check_output(['git','ls-remote','--heads','origin',prefix+'*'],text=True).splitlines()
    output.mkdir(parents=True,exist_ok=True)
    for line in lines:
        sha,branch=line.split()
        subprocess.run(['git','fetch','origin',branch],check=True)
        names=subprocess.check_output(['git','ls-tree','-r','--name-only',sha,'results/calibrated_recipe_study'],text=True).splitlines()
        for name in names:
            relative=Path(name).relative_to('results')
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid artifact path')
            content=subprocess.check_output(['git','show',sha+':'+name])
            target=output/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.read_bytes()!=content:raise ValueError(f'Conflicting aggregated result: {target}')
            target.write_bytes(content)
    rows=[]
    for path in sorted(output.rglob('summary.json')):
        s=json.loads(path.read_text());cfg=s['settings']
        rows.append(dict(path=str(path),family=cfg['family'],seed=cfg['seed'],initialization=cfg['initialization'],
                         variant=cfg['variant'],stage=cfg['stage'],learning=s['learning'],
                         final_development=s['final_development']['answer_free'],wall_seconds=s['wall_seconds']))
    atomic_write_json(output/'aggregate_index.json',dict(branches=len(lines),runs=rows))
    return rows


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('research_results/aggregated_recipe_study'))
    p.add_argument('--protocol',default='matched-decay-v1')
    a=p.parse_args();aggregate_remote(a.output,a.protocol)
