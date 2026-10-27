"""Batch scoring entry point; the same API serves whole charts and segments."""
import argparse,csv,json,sys
from pathlib import Path
from . import load_model,rate_file

def main():
    p=argparse.ArgumentParser(description='SPM Rating v1.2.0 chart and segment difficulty')
    p.add_argument('paths',nargs='+',help='.osu file or directory')
    p.add_argument('--model',choices=['detail','compact'],default='detail',help='detail=detailed model; compact=simpler model')
    p.add_argument('--rate',type=float,default=1.)
    p.add_argument('--scale',choices=['reference','raw'],default='reference')
    p.add_argument('--start-ms',type=float);p.add_argument('--end-ms',type=float)
    p.add_argument('--scope',choices=['context','isolated'],default='context')
    p.add_argument('--csv',type=Path);args=p.parse_args()
    model=load_model(args.model);files=[]
    for value in args.paths:
        path=Path(value)
        files.extend(sorted(path.rglob('*.osu')) if path.is_dir() else [path])
    files=list(dict.fromkeys(path.resolve() for path in files));rows=[];failed=False
    for path in files:
        try:row=rate_file(path,model,rate=args.rate,scale=args.scale,start_ms=args.start_ms,end_ms=args.end_ms,scope=args.scope)
        except Exception as error:row=dict(path=str(path),error=str(error));failed=True
        rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    if args.csv:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        with args.csv.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    if failed:raise SystemExit(1)

if __name__=='__main__':main()
