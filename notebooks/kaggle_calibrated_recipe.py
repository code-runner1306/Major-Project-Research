# Paste this into a Kaggle Python cell. GPU T4 x2, Internet, GH_TOKEN secret.
# Default: minimum four-model exploratory comparison on overwrite, seed42.
STAGE = 'explore'  # pilot (16 epochs), explore (16), study (64)
FAMILIES = ['overwrite']  # ['mqar'], ['overwrite'], or both
SEEDS = [42]  # explicit list, e.g. [42,43]; None uses SEED_COUNT/START_SEED
SEED_COUNT = 1
START_SEED = 42
INITIALIZATIONS = ['default','calibrated']
QUALIFICATION_BRANCHES = []  # successful pilot model branches, needed for study
PUBLISH = True  # includes each model's latest checkpoint

import os
import pathlib
import subprocess
import sys
import time
import uuid
from kaggle_secrets import UserSecretsClient

started = time.monotonic()
def left():
    seconds = int(41400 - (time.monotonic()-started) - 30)
    if seconds <= 900:
        raise RuntimeError('Notebook setup exhausted the time reserve')
    return seconds
def command(args, capture=False):
    if capture:
        return subprocess.check_output(args,text=True,timeout=left())
    return subprocess.run(args,check=True,timeout=left())

repo = pathlib.Path('/kaggle/working/Major-Project-Research')
if not repo.exists():
    command(['git','clone','--branch','maink',
             'https://github.com/code-runner1306/Major-Project-Research.git',str(repo)])
os.chdir(repo)
command(['git','fetch','origin','maink'])
command(['git','checkout','--detach','origin/maink'])
print('Source:',command(['git','rev-parse','HEAD'],capture=True).strip())
command([sys.executable,'-m','pip','install','-r','requirements.txt'])
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHON_BIN=sys.executable)
if PUBLISH:
    os.environ['GH_TOKEN'] = UserSecretsClient().get_secret('GH_TOKEN')

qualification = pathlib.Path('/kaggle/working/calibrated_qualification')
for branch in QUALIFICATION_BRANCHES:
    command(['git','check-ref-format','refs/heads/'+branch])
    ref = 'refs/remotes/origin/'+branch
    command(['git','fetch','origin','refs/heads/'+branch+':'+ref])
    names = command(['git','ls-tree','-r','--name-only',ref,'results/calibrated_recipe_study'],capture=True).splitlines()
    for name in names:
        if not name.endswith('.json'):
            continue
        relative = pathlib.PurePosixPath(name).relative_to('results')
        if '..' in relative.parts:
            raise RuntimeError('Invalid qualification path')
        target = qualification.joinpath(*relative.parts)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(command(['git','show',ref+':'+name],capture=True),encoding='utf-8')

run_id = time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
output = repo/'output'/'calibrated_recipe_study'/'matched-decay-v1'/run_id
args = ['bash','run_kaggle_calibrated_recipe.sh','--stage',STAGE,'--devices','0,1',
        '--output',str(output),'--max-seconds',str(min(39600,left()-600)),
        '--family',*FAMILIES,'--initialization',*INITIALIZATIONS]
if SEEDS is not None:
    args += ['--seeds',*map(str,SEEDS)]
else:
    args += ['--seed-count',str(SEED_COUNT),'--start-seed',str(START_SEED)]
if STAGE == 'study':
    assert QUALIFICATION_BRANCHES, 'Supply both scalar/channel pilot branches for each requested task'
    args += ['--qualification-root',str(qualification)]
if PUBLISH:
    args += ['--publish']
else:
    args += ['--no-publish']
command(args)
print('Session report:',output/'session_status.json')
