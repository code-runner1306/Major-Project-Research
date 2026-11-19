from pathlib import Path
from zipfile import ZipFile
import re
import unicodedata
import pymupdf
from PIL import Image

build=Path('.scratch/indicon_build')
paper=Path('indicon_gate_paper')
doc=pymupdf.open(build/'main.pdf')
bbl=(build/'main.bbl').read_text(encoding='utf-8')
refs=len(re.findall(r'\\bibitem\{[^}]+\}',bbl))
pages=[unicodedata.normalize('NFKC',page.get_text()) for page in doc]
text='\n'.join(pages)
assert len(doc)==6 and refs==18, (len(doc),refs)
assert '55M' not in text and 'PLACEHOLDER' not in text
assert all(x in text for x in ('1,024','0.9998','0.9830','Latest value','85.4%','Value-marginal floor'))
log=(build/'main.log').read_text(encoding='utf-8')
assert not re.search(r'(?im)undefined (citations|references)|Overfull',log)
assert not re.search(r'(?im)(?:Citation|Reference).*undefined',log)
figures={}
for i,page in enumerate(pages):
    for number in re.findall(r'^Fig\.\s*(\d)\.',page,re.M):
        figures[int(number)]=i+1
assert set(figures)=={1,2,3,4},figures
print(f'{len(doc)} pages, {refs} references, figure pages {figures}')
out=Image.new('RGB',(360*3,452*((len(doc)+2)//3)),'#ddd')
for i,p in enumerate(doc):
    pix=p.get_pixmap(matrix=pymupdf.Matrix(.55,.55),alpha=False)
    out.paste(Image.frombytes('RGB',(pix.width,pix.height),pix.samples),((i%3)*360,(i//3)*452))
out.save(build/'followup_final_contact.png')
with ZipFile(build/'Overleaf_IEEE_Conference.zip') as archive:
    assert all(archive.read(name)==(paper/name).read_bytes() for name in archive.namelist())
