"""Build auditable discovery/evidence registers without claiming all titles were read in full."""
import csv
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent

def write_csv(name,rows):
    with (ROOT/name).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    metadata=json.loads((ROOT/'primary_metadata_verification.json').read_text())
    full={'2312.04927','2610.41994','2607.17419','2611.08376','2609.33093',
          '2609.16183','2605.11196','2610.00232','2609.16540'}
    evidence=[]
    for row in metadata:
        if row['status']!='verified metadata':continue
        m=row['metadata'];identifier=row['arxiv_id']
        evidence.append(dict(id='arxiv:'+identifier,title=m['citation_title'][0],
            date='; '.join(m.get('citation_date',m.get('citation_publication_date',[]))),url=row['url'],
            evidence='A: relevant full-text sections' if identifier in full else 'B: primary record/abstract',
            scope='included as direct or contextual evidence; see thematic notes'))
    extra=[
        ('hopfield1982','Neural networks and physical systems with emergent collective computational abilities','1982','https://www.pnas.org/doi/10.1073/pnas.79.8.2554','B: historical primary record'),
        ('schmidhuber1992','Learning to Control Fast-Weight Memories','1992','https://gwern.net/doc/ai/nn/rnn/1992-schmidhuber-2.pdf','B: historical paper/author reference'),
        ('ba2016','Using Fast Weights to Attend to the Recent Past','2016','https://proceedings.neurips.cc/paper/2016/hash/9f44e956e3a2b7b5598c625fcc802c36-Abstract.html','A: relevant paper sections'),
        ('schlag2021','Linear Transformers Are Secretly Fast Weight Programmers','2021','https://proceedings.mlr.press/v139/schlag21a.html','A: relevant paper sections'),
        ('das2026','Delta-Rule Writes in Linear Attention: A Controlled Factorial Study of Decay Design','2026','https://tmlr.infinite-conf.org/paper_pages/FdXB4u27LD.html','A: relevant paper/results sections'),
        ('mao2022','Fine-Tuning Pre-trained Transformers into Decaying Fast Weights','2022','https://aclanthology.org/2022.emnlp-main.697/','A: relevant paper sections'),
    ]
    for identifier,title,date,url,level in extra:evidence.append(dict(id=identifier,title=title,date=date,url=url,evidence=level,scope='included as direct or contextual evidence; see thematic notes'))
    write_csv('evidence_register.csv',evidence)
    def key(title):return re.sub(r'[^a-z0-9]','',title.lower())
    by_title={key(row['title']):row for row in evidence}
    by_id={row['id']:row for row in evidence}
    candidates=list(csv.DictReader((ROOT/'extended_catalog_candidates.csv').open(encoding='utf-8')))
    screened=[]
    # Unread plausible records are retained, never silently ruled out by a keyword.
    false_positive=re.compile(r'Paronychiurus|CMOS|Nile Delta|olive oil|polyphenols|tauopathy|BepiColombo|cadmium|heat shock protein|Alzheimer|solar photovoltaic|quantum.gate|two.qubit|chiral symmetry|magnon decays|hard.disk fluid|conform|collembola',re.I)
    for row in candidates:
        match=re.search(r'arxiv[./:](\d{4}\.\d{4,5})',row['doi']+' '+row['url'],re.I)
        selected=by_id.get('arxiv:'+match.group(1)) if match else None
        selected=selected or by_title.get(key(row['title']))
        if selected:status,reason='primary_followed_up',selected['id']+'; '+selected['evidence']
        elif false_positive.search(row['title']):status,reason='exclude_title_scope','Clear unrelated lexical/domain match; title-level exclusion only'
        else:status,reason='catalog_only_not_selected','Title triaged; not selected for primary reading in this bounded review; not evidence of irrelevance'
        screened.append(dict(row,screening=status,screening_reason=reason))
    write_csv('screening_ledger.csv',screened)
    # Export backward/forward catalog links between discovered records.
    seeds=json.loads((ROOT/'verified_citation_seeds.json').read_text())
    raw=json.loads((ROOT/'citation_extension_raw.json').read_text())
    by_catalog={row['catalog_id']:row for row in candidates}
    edges=[]
    for seed in seeds:
        for target in seed.get('referenced_works',[]):
            edges.append(dict(source=seed['id'],source_title=seed['title'],target=target,
                target_title=by_catalog.get(target,{}).get('title','unresolved catalog ID'),
                relation='seed references target; catalog link, not full-text reading'))
        for source in raw:
            if seed['id'] in source.get('referenced_works',[]):
                edges.append(dict(source=source['id'],source_title=source['title'],target=seed['id'],target_title=seed['title'],
                    relation='source cites seed; catalog link, not full-text reading'))
    unique={(r['source'],r['target']):r for r in edges}
    write_csv('citation_edges.csv',list(unique.values()))
    counts={status:sum(r['screening']==status for r in screened) for status in sorted({r['screening'] for r in screened})}
    summary=dict(cutoff='2026-11-17',discovery_title_keys=len(candidates),selected_primary_records=len(evidence),
        evidence_A=sum(r['evidence'].startswith('A') for r in evidence),evidence_B=sum(r['evidence'].startswith('B') for r in evidence),
        title_screening=counts,catalog_citation_edges=len(unique),
        caveat='Discovery title keys are not full papers read; conference/preprint aliases may remain. No completeness guarantee.')
    (ROOT/'review_counts.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
