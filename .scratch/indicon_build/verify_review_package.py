from pathlib import Path
from zipfile import ZipFile
import re
import pymupdf
from PIL import Image

build=Path('.scratch/indicon_build')
paper=Path('indicon_gate_paper')
doc=pymupdf.open(build/'main.pdf')
bbl=(build/'main.bbl').read_text(encoding='utf-8')
refs=len(re.findall(r'\\bibitem\{[^}]+\}',bbl))
pages=[page.get_text() for page in doc]
text='\n'.join(pages)
assert len(doc)>=6 and refs==18 and '55M' not in text
assert '1,024' in text and '24.90' in text and '31.45' in text
assert re.search(r'Value.marginal (?:floor|.oor)',text)
log=(build/'main.log').read_text(encoding='utf-8')
assert not re.search(r'(?im)undefined (citations|references)|Overfull',log)
assert not re.search(r'(?im)(?:Citation|Reference).*undefined',log)
figures={}
expected_text=['Value-by-key recurrence','Filler extension','Anatomy harness paired differences']
for figure, expected in enumerate(expected_text,1):
    index=next(i for i,page in enumerate(pages) if re.search(rf'^Fig\.\s*{figure}\.',page,re.M))
    assert expected in pages[index], (figure,index)
    figures[figure]=index+1
print(f'{len(doc)} pages, {refs} references, figures on pages {figures}')
out=Image.new('RGB',(320*4,430*((len(doc)+3)//4)),'#ddd')
for i,p in enumerate(doc):
    pix=p.get_pixmap(matrix=pymupdf.Matrix(.49,.49),alpha=False)
    out.paste(Image.frombytes('RGB',(pix.width,pix.height),pix.samples),((i%4)*320,(i//4)*430))
out.save(build/'review_final_contact.png')
for number,page_no in figures.items():
    p=doc[page_no-1]
    pix=p.get_pixmap(matrix=pymupdf.Matrix(1.25,1.25),clip=pymupdf.Rect(35,30,575,465),alpha=False)
    pix.save(build/f'review_final_fig{number}.png')
with ZipFile(build/'Overleaf_IEEE_Conference.zip') as archive:
    assert all(archive.read(name)==(paper/name).read_bytes() for name in archive.namelist())
