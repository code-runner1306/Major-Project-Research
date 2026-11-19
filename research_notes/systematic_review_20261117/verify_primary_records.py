"""Check primary arXiv title/ID pairs discovered in the review notes."""
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import urllib.request

ROOT=Path(__file__).resolve().parent
class Metadata(HTMLParser):
    def __init__(self):super().__init__();self.values={}
    def handle_starttag(self,tag,attrs):
        values=dict(attrs)
        if tag=='meta' and values.get('name','').startswith('citation_'):
            self.values.setdefault(values['name'],[]).append(values.get('content',''))

def verify(identifier):
    url='https://arxiv.org/abs/'+identifier
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'ResearchLiteratureReview/1.0'})
        with urllib.request.urlopen(request,timeout=25) as response:page=response.read().decode('utf-8')
        parser=Metadata();parser.feed(page)
        if not parser.values.get('citation_title'):raise ValueError('No primary citation-title metadata')
        return dict(arxiv_id=identifier,url=url,status='verified metadata',metadata=parser.values,
                    evidence='metadata only; see thematic notes for abstract/full-text assessment')
    except Exception as error:return dict(arxiv_id=identifier,url=url,status='access error',error=str(error))

def main():
    identifiers=set()
    for name in ('history.md','ownhistory.md','gates_initialization.md','tasks_memory.md'):
        text=(ROOT/name).read_text(encoding='utf-8')
        identifiers.update(re.findall(r'arxiv\.org/(?:abs|html)/(\d{4}\.\d{4,5})',text))
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(verify,sorted(identifiers)))
    (ROOT/'primary_metadata_verification.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    for row in rows:print(row['arxiv_id'],row['status'],row.get('metadata',{}).get('citation_title',row.get('error')),flush=True)

if __name__=='__main__':main()
