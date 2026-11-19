"""Verify citation seeds by DOI; paginate bounded forward citation sets."""
import csv
import datetime
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parent
def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'ResearchLiteratureReview/1.0'})
    with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)

def main():
    log=[];discovered=[];seeds=[]
    for title,doi,cap in [('Gated Delta Networks','10.48550/arxiv.2412.06464',400),
                          ('Kimi Linear','10.48550/arxiv.2510.26692',400),
                          ('Zoology','10.48550/arxiv.2312.04927',200)]:
        url='https://api.openalex.org/works/https://doi.org/'+doi
        entry=dict(query='DOI seed '+doi,url=url,retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            seed=fetch(url);seeds.append(seed);entry.update(status='ok',title=seed['title'],id=seed['id'])
        except Exception as e:
            entry.update(status='error',error=str(e));log.append(entry);continue
        log.append(entry)
        for page in range(1,cap//100+1):
            identifier=seed['id'].rsplit('/',1)[-1]
            url='https://api.openalex.org/works?'+urllib.parse.urlencode(dict(filter='cites:'+identifier+',to_publication_date:2026-11-17',per_page=100,page=page))
            entry=dict(query='cites '+seed['title'],page=page,url=url,retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            try:
                data=fetch(url);rows=data['results'];discovered.extend(rows)
                entry.update(status='ok',returned=len(rows),reported_total=data['meta']['count'])
                if page*100>=data['meta']['count']:break
            except Exception as e:
                entry.update(status='error',error=str(e));break
            finally:log.append(entry)
        print(title,entry,flush=True)
    (ROOT/'verified_citation_seeds.json').write_text(json.dumps(seeds,indent=2),encoding='utf-8')
    (ROOT/'citation_extension_log.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
    (ROOT/'citation_extension_raw.json').write_text(json.dumps(discovered,indent=2),encoding='utf-8')
    path=ROOT/'catalog_candidates.csv'
    rows=list(csv.DictReader(path.open(encoding='utf-8')))
    import re
    key=lambda title:re.sub(r'[^a-z0-9]','',title.lower())
    known={key(row['title']) for row in rows}
    for row in discovered:
        if key(row['title']) in known:continue
        known.add(key(row['title']))
        rows.append(dict(title=row['title'],year=row.get('publication_year'),doi=row.get('doi') or '',
            catalog_id=row['id'],url=row.get('primary_location',{}).get('landing_page_url') or row.get('doi') or row['id'],
            sources='OpenAlex verified-seed citation extension',queries='verified DOI citation search',
            screening='catalog discovery only; not full-text screened'))
    with (ROOT/'extended_catalog_candidates.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    print('TOTAL_UNIQUE_TITLE_KEYS',len(rows),flush=True)

if __name__=='__main__':main()
