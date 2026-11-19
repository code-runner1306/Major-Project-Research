import re
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
root=Path('indicon_gate_paper')
build=Path('.scratch/indicon_build')
tex=(root/'main.tex').read_text(encoding='utf-8')
assert 'PLACEHOLDER' not in tex
figures=re.findall(r'\\includegraphics\[[^\]]+\]\{(figures/[^}]+\.pdf)\}',tex)
assert len(figures)==4 and len(set(figures))==4
entries=['main.tex','references.bib','README.md','IEEEtran.cls',*figures]
with ZipFile(build/'Overleaf_IEEE_Conference.zip','w',ZIP_DEFLATED) as archive:
    for entry in entries:
        archive.write(root/entry,entry)
with ZipFile(build/'Overleaf_IEEE_Conference.zip') as archive:
    assert archive.testzip() is None
    archive.extractall(build/'final_source')
