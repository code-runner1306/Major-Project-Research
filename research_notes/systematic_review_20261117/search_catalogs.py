"""Read-only catalog discovery; save query evidence, never infer novelty from absence."""
import csv
import datetime
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parent
CUTOFF='2026-11-17'
QUERIES=[
    'Gated Delta Networks', 'Kimi Linear', 'DAMI Dual Axis Memory Initialization',
    'Delta Rule Writes Linear Attention Controlled Factorial Decay Design',
    'Kernelized Linear Attention Symmetric Cones', 'How Linear Attention Remembers',
    'linear attention gate initialization calibration', 'multi query associative recall overwrite',
    'fast weight memory delta rule', 'chrono initialization recurrent networks',
    'linear attention static channel decay', 'associative memory repeated overwrite retention',
]


def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'ResearchLiteratureReview/1.0'})
    with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    logs=[];records={};responses=[]
    def query(provider,term,url):
        entry=dict(provider=provider,query=term,url=url,cutoff=CUTOFF,
                   retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            data=fetch(url)
            rows=data.get('results',[]) if provider.startswith('OpenAlex') else data['message']['items']
            entry.update(status='ok',returned=len(rows),
                reported_total=data.get('meta',{}).get('count') if provider.startswith('OpenAlex') else data['message'].get('total-results'))
            responses.append(dict(query=entry,response=data))
            for row in rows:
                title=row.get('title','')
                if isinstance(title,list):title=' '.join(title)
                if not title:continue
                doi=row.get('doi') or row.get('DOI') or ''
                key=re.sub(r'[^a-z0-9]','',title.lower())
                year=row.get('publication_year')
                if year is None:
                    parts=row.get('issued',{}).get('date-parts',[[None]])
                    year=parts[0][0]
                candidate=records.setdefault(key,dict(title=title,year=year,doi=doi,
                    catalog_id=row.get('id',''),url=row.get('primary_location',{}).get('landing_page_url') or row.get('URL') or doi,
                    sources=[],queries=[],screening='catalog discovery only; not full-text screened'))
                if provider not in candidate['sources']:candidate['sources'].append(provider)
                candidate['queries'].append(term)
            return rows
        except Exception as error:
            entry.update(status='error',error=str(error));return []
        finally:
            logs.append(entry)
            (ROOT/'catalog_query_log.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
            print(provider,term,entry['status'],entry.get('returned',entry.get('error')),flush=True)
    seeds={}
    for term in QUERIES:
        url='https://api.openalex.org/works?'+urllib.parse.urlencode(dict(search=term,per_page=20,filter='to_publication_date:'+CUTOFF))
        rows=query('OpenAlex search',term,url)
        # A top-ranked fuzzy hit is not necessarily the intended citation seed.
        expected={
            QUERIES[0]:'Gated Delta Networks: Improving Mamba2 with Delta Rule',
            QUERIES[1]:'Kimi Linear: An Expressive, Efficient Attention Architecture',
            QUERIES[2]:'DAMI: Dual-Axis Memory Initialization for Enhanced Long-Context Retrieval in Gated Linear Attention',
            QUERIES[3]:'Delta-Rule Writes in Linear Attention: A Controlled Factorial Study of Decay Design',
            QUERIES[4]:'Kernelized Linear Attention: Breaking the Capacity Wall with Symmetric Cones',
            QUERIES[5]:'How Linear Attention Remembers',
        }
        match=next((row for row in rows if row['title'].casefold()==expected.get(term,'').casefold()),None)
        if match:seeds[term]=match
    for term in ('Linear Transformers Are Secretly Fast Weight Programmers',
                 'Can recurrent neural networks warp time',
                 'Using Fast Weights to Attend to the Recent Past',
                 'Zoology Measuring and Improving Recall in Efficient Language Models'):
        url='https://api.crossref.org/works?'+urllib.parse.urlencode({'query.bibliographic':term,'rows':20,'filter':'until-pub-date:'+CUTOFF})
        query('Crossref search',term,url)
    # These are catalog-derived links, not claims of having read all citing papers.
    for term,row in seeds.items():
        if not row.get('id'):continue
        identifier=row['id'].rsplit('/',1)[-1]
        url='https://api.openalex.org/works?'+urllib.parse.urlencode(dict(filter='cites:'+identifier+',to_publication_date:'+CUTOFF,per_page=100))
        query('OpenAlex forward citations','cites '+row['title'],url)
        # Preserve the seed's backward list verbatim for the citation trace.
        (ROOT/('references_'+identifier+'.json')).write_text(json.dumps(dict(seed_title=row['title'],
            seed_id=row['id'],referenced_works=row.get('referenced_works',[])),indent=2),encoding='utf-8')
    (ROOT/'catalog_raw_responses.json').write_text(json.dumps(responses,indent=2),encoding='utf-8')
    with (ROOT/'catalog_candidates.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['title','year','doi','catalog_id','url','sources','queries','screening'])
        writer.writeheader()
        for row in records.values():writer.writerow(dict(row,sources='; '.join(row['sources']),queries='; '.join(row['queries'])))
    print('UNIQUE_DISCOVERY_RECORDS',len(records),flush=True)


if __name__=='__main__':main()
