from __future__ import annotations
import argparse, asyncio, csv, json, time, uuid
from datetime import datetime, timezone
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]
BANK = Path(__file__).with_name('edge_case_questions.json')
BASE_COLUMNS = ['request_id','scenario','source','start_time','first_token_ms','total_ms','status','retries','tokens_in','tokens_out','retrieval_setting','answer_grade']
EXTRA_COLUMNS = ['test_id','category','paper_scope','question','expected_behavior','evaluation_notes','answer','top_score','citation_valid','citation_validation_reason','citation_found','citation_invalid','citation_uncited_blocks','error_kind','http_status','tokens_source','draft_first_token_ms','queue_wait_ms','retrieval_ms','generation_ms','citation_check_ms','grader_notes']
FIELDS = BASE_COLUMNS + EXTRA_COLUMNS

def cases_for(profile, bank):
    if profile == 'smoke':
        cats = ['normal_answerable','partial_support','not_in_library','equation_and_citation','robustness_prompt_injection']
        out=[]
        for cat in cats:
            item=next((x for x in bank if x['category']==cat),None)
            if item: out.append(item)
        return out
    return bank

async def main(args):
    bank=json.loads(BANK.read_text(encoding='utf-8'))
    cases=cases_for(args.profile,bank)
    if not args.include_boundary:
        cases=[x for x in cases if x['category']!='boundary_and_length']
    if args.limit: cases=cases[:args.limit]
    runid=uuid.uuid4().hex[:8]; base=args.api_url.rstrip('/'); rows=[]
    timeout=httpx.Timeout(args.timeout,connect=10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        health=await client.get(base+'/health'); health.raise_for_status(); info=health.json()
        actual=info.get('backend','').lower(); expected='standin' if args.source=='stand-in' else 'groq'
        if actual != expected: raise SystemExit(f"Backend mismatch: API={actual}, --source={args.source}. Restart API with matching LLM_BACKEND.")
        print(f"Backend={actual}; model={info.get('model')}; tests={len(cases)}; serial execution")
        for i,c in enumerate(cases,1):
            q=c['question']
            if q=='OVERSIZE_SENTINEL': q='X'*4101
            start=datetime.now(timezone.utc).isoformat(); t=time.perf_counter(); http_status=''
            try:
                r=await client.post(base+'/ask',json={'question':q,'retrieval_setting':args.retrieval_setting,'scenario':'edge_'+c['category']})
                http_status=r.status_code; elapsed=(time.perf_counter()-t)*1000
                if r.status_code>=400: result={'status':'error','answer':r.text,'error_kind':'http_error','total_ms':round(elapsed,2)}
                else: result=r.json()
            except Exception as e:
                elapsed=(time.perf_counter()-t)*1000; result={'status':'error','answer':str(e),'error_kind':type(e).__name__,'total_ms':round(elapsed,2)}
            row={
              'request_id':result.get('request_id',str(uuid.uuid4())),'scenario':'edge_'+c['category'],'source':args.source,
              'start_time':start,'first_token_ms':result.get('first_token_ms',''),'total_ms':result.get('total_ms',round(elapsed,2)),
              'status':result.get('status','error'),'retries':result.get('retries',0),'tokens_in':result.get('tokens_in',0),'tokens_out':result.get('tokens_out',0),
              'retrieval_setting':args.retrieval_setting,'answer_grade':'','test_id':c['id'],'category':c['category'],'paper_scope':c['paper'],
              'question':q,'expected_behavior':c['expected_behavior'],'evaluation_notes':c['evaluation_notes'],'answer':result.get('answer',''),
              'top_score':result.get('top_score',''),'citation_valid':result.get('citation_valid',''),
              'citation_validation_reason':result.get('citation_validation_reason',''),
              'citation_found':json.dumps(result.get('citation_found',[]),ensure_ascii=False),
              'citation_invalid':json.dumps(result.get('citation_invalid',[]),ensure_ascii=False),
              'citation_uncited_blocks':json.dumps(result.get('citation_uncited_blocks',[]),ensure_ascii=False),
              'error_kind':result.get('error_kind',''),'http_status':http_status,'tokens_source':result.get('tokens_source',''),
              'draft_first_token_ms':result.get('draft_first_token_ms',''),'queue_wait_ms':result.get('queue_wait_ms',''),
              'retrieval_ms':result.get('retrieval_ms',''),'generation_ms':result.get('generation_ms',''),
              'citation_check_ms':result.get('citation_check_ms',''),'grader_notes':''}
            rows.append(row)
            print(f"[{i}/{len(cases)}] {c['id']} {c['category']} status={row['status']} score={row['top_score']} citation={row['citation_valid']} time={row['total_ms']}ms")
            if args.pause: await asyncio.sleep(args.pause)
    outdir=ROOT/'results'; outdir.mkdir(exist_ok=True); out=outdir/f'edge_case_evaluation_{runid}.csv'
    with out.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    meta={'run_id':runid,'backend':actual,'model':info.get('model'),'profile':args.profile,'retrieval_setting':args.retrieval_setting,'requests':len(rows),'csv':str(out)}
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('\nSaved:',out); print('Grade answer_grade manually: correct, wrong, refused_correctly, refused_wrongly.')
    print('Pipeline recorder separately appends processed requests to results/results.csv.')
    print('Do not use stand-in outputs as answer-quality evidence.')

def cli():
    p=argparse.ArgumentParser(description='Run Relay RAG edge-case evaluation.')
    p.add_argument('--api-url',default='http://127.0.0.1:8000'); p.add_argument('--source',choices=['real','stand-in'],required=True)
    p.add_argument('--profile',choices=['smoke','comprehensive'],default='comprehensive')
    p.add_argument('--retrieval-setting',choices=['small','balanced','large'],default='balanced')
    p.add_argument('--limit',type=int,default=0); p.add_argument('--pause',type=float,default=1.5); p.add_argument('--timeout',type=float,default=180)
    p.add_argument('--include-boundary',action='store_true'); args=p.parse_args()
    if args.source=='real' and args.profile=='comprehensive' and not args.limit:
        print('WARNING: comprehensive run sends 32+ real model requests and can consume substantial daily quota.')
    asyncio.run(main(args))
if __name__=='__main__': cli()
