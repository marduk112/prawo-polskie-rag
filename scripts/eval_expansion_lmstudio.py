"""Bateria 129 przez LM Studio. Fazy: `expand A` | `expand BC` | `rank`. Jedna dziedzina w pamięci naraz."""
import sys, json, os, threading, gc
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,'scripts')
SP=os.environ['SP']; OUT=SP+'/exp_battery_expansions.json'
HINT="\n(W przeformułowaniu wymień konkretne numery przepisów, które według Ciebie regulują tę sprawę, w postaci „art. N ustawy ...”.)"
cases=json.load(open('data/expand_query_reference_cases.json'))
data=json.load(open(OUT)) if os.path.exists(OUT) else {}
phase=sys.argv[1]
if phase=='expand':
    import chat_lmstudio as c
    variants=sys.argv[2]
    titles={}
    for d in {x['domain'] for x in cases}:
        meta=json.load(open(f'data/processed/rag_index_{d}_meta.json'))
        titles[d]=sorted({m['act_title'] for m in meta})
    lock=threading.Lock()
    def work(case):
        cid=case['id']; r=data.get(cid,{}); d=case['domain']; q=case['question']; at=titles[d]
        try:
            if 'A' in variants and 'A' not in r: r['A']=c.expand_query(c.DEFAULT_URL,'bielik-11b-v3.0-instruct',q,act_titles=at,domain=d)
            if 'B' in variants and 'B' not in r: r['B']=c.expand_query(c.DEFAULT_URL,'qwen/qwen3.8-27b',q,act_titles=at,domain=d,reasoning_effort='low')
            if 'C' in variants and 'C' not in r: r['C']=c.expand_query(c.DEFAULT_URL,'qwen/qwen3.8-27b',q+HINT,act_titles=at,domain=d,reasoning_effort='low')
        except Exception as e:
            r['err']=str(e)[:200]
        with lock:
            data[cid]=r; json.dump(data,open(OUT,'w'),ensure_ascii=False); print(cid,'ok',flush=True)
    with ThreadPoolExecutor(3) as ex: list(ex.map(work,cases))
else:
    from rag_search import RagIndex, RERANK_CANDIDATE_POOL
    shared=None; summary={v:0 for v in 'ABC'}; empty={v:0 for v in 'ABC'}; rows=[]
    for d in sorted({x['domain'] for x in cases}):
        ix=RagIndex(domain=d,rerank=True,fuse=True,model=shared.model if shared else None,cross_encoder=shared.cross_encoder if shared else None)
        shared=shared or ix
        for case in [x for x in cases if x['domain']==d]:
            r=data[case['id']]; row={'id':case['id'],'max':case['max_acceptable_rank']}
            for v in 'ABC':
                e=r.get(v,'')
                if not e.strip(): empty[v]+=1
                res=ix.search(case['question'],top_k=2*RERANK_CANDIDATE_POOL,extra_queries=[e] if e.strip() else None,candidate_pool=RERANK_CANDIDATE_POOL)
                rk=next((i+1 for i,x in enumerate(res) if x['act_short']==case['act_short'] and x['article']==case['article']),None)
                row[v]=rk
                if rk is not None and rk<=case['max_acceptable_rank']: summary[v]+=1
            rows.append(row)
        del ix; gc.collect()
        print(d,'done',flush=True)
    json.dump({'summary':summary,'empty':empty,'rows':rows},open(SP+'/exp_battery_result.json','w'),ensure_ascii=False,indent=1)
    print('WYNIK OK z',len(cases),summary,'puste ekspansje:',empty)
    for row in rows:
        if len({row[v] is not None and row[v]<=row['max'] for v in 'ABC'})>1: print('ROZNICA',row)
print('DONE',phase,flush=True)
