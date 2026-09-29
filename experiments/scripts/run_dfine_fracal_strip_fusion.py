"""FRACAL calibration evaluated with the established cyclic strip-fusion protocol."""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dfine_fracal_weld as fw
import fuse_dfine_strip_instances as fuse
import evaluate_dfine_strict_detection_errors as ev

MAP = {0:0,2:0,3:0,12:12,14:12,15:12,18:12}
EXCLUDE = {11,17,19,21,22}
NAMES = {0:'烟类合并',12:'缺焊长塌合并'}

def canonical(cid:int)->int:
    return -1 if cid in EXCLUDE else MAP.get(cid,cid)

def args():
 p=argparse.ArgumentParser();
 for n in ['dfine-root','config','checkpoint','image-root','train-annotations','native-annotations','source-annotations','tiles-metadata','strip-boxes','output-dir']:
  p.add_argument('--'+n, type=Path, required=True)
 p.add_argument('--raw-threshold',type=float,default=.25); p.add_argument('--fracal-thresholds',default='.01,.02,.03,.04,.05')
 p.add_argument('--top-k',type=int,default=300); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--device',default='cuda')
 p.add_argument('--merge-iou',type=float,default=.50); p.add_argument('--match-iou',type=float,default=.50); p.add_argument('--y-pad',type=float,default=5.0)
 return p.parse_args()

def canon_strip(rows):
 out=[]
 for item in rows:
  x=dict(item); bs=[]
  for box in item['boxes']:
   c=canonical(int(box['category_id']))
   if c>=0: bs.append(dict(box,category_id=c))
  x['boxes']=bs; out.append(x)
 return out

def make_gt(strip):
 return [{'source_annotation_id':int(b['source_annotation_id']),'source_image_id':int(s['source_image_id']),'category_id':int(b['category_id']),'box':[float(b['x']),float(b['y']),float(b['width']),float(b['height'])]} for s in strip for b in s['boxes']]

def infer(a):
 native=fw.load_json(a.native_annotations); train=fw.load_json(a.train_annotations)
 tile={str(x['tile_file']):x for x in fw.load_json(a.tiles_metadata)}
 width={int(x['source_image_id']):float(x['strip_width']) for x in tile.values()}
 model,transform,_=fw.build_raw_model(a,torch.device(a.device))
 adjustment,_=fw.compute_fracal_adjustments(train,[int(x['id']) for x in train['categories']],2.,32)
 raw,cal=[],[]
 records=sorted(native['images'],key=lambda x:int(x['id']))
 with torch.inference_mode():
  for start in range(0,len(records),a.batch_size):
   batch=records[start:start+a.batch_size]; tensors=[]; sizes=[]
   for r in batch:
    from PIL import Image
    with Image.open(a.image_root/str(r['file_name'])) as im:
     im=im.convert('RGB'); tensors.append(transform(im)); sizes.append([float(im.width),float(im.height)])
   st=torch.tensor(sizes,dtype=torch.float32,device=a.device); out=model(torch.stack(tensors).to(a.device))
   for target,adj in [(raw,None),(cal,adjustment)]:
    labs,boxes,scores=fw.topk_predictions(out['pred_logits'],out['pred_boxes'],st,a.top_k,adj)
    for i,r in enumerate(batch):
     meta=tile[str(r['file_name'])]; target.append((r,meta,labs[i].cpu(),boxes[i].cpu(),scores[i].cpu()))
   if (start+len(batch))%200==0 or start+len(batch)==len(records): print(f'inferred {start+len(batch)}/{len(records)}',flush=True)
 return raw,cal,width

def candidates(batches, threshold, ypad):
 groups=defaultdict(list)
 for r,m,labs,boxes,scores in batches:
  for cid,box,score in zip(labs.tolist(),boxes.tolist(),scores.tolist()):
   cid=canonical(int(cid))
   if cid<0 or float(score)<threshold: continue
   x0,y0,x1,y1=(float(x) for x in box); x0,x1=sorted((max(0.,x0),min(float(r['width'])-1.,x1))); y0,y1=sorted((max(0.,y0),min(float(r['height'])-1.,y1)))
   if x1<=x0 or y1<=y0: continue
   groups[(int(m['source_image_id']),cid)].append({'source_image_id':int(m['source_image_id']),'category_id':cid,'score':float(score),'strip_box':np.asarray([x0+float(m['tile_start']),y0-ypad,x1+float(m['tile_start']),y1-ypad],dtype=np.float32),'tile_file':str(r['file_name'])})
 return groups

def evaluate(name,batches,threshold,strip,width,a,outdir):
 outdir.mkdir(parents=True, exist_ok=True)
 fused=fuse.fuse_predictions(candidates(batches,threshold,a.y_pad),width,a.merge_iou)
 gtl,predl=fuse.match_strip_instances(make_gt(strip),fused,width,a.match_iou,.20,False)
 c=Counter(x['outcome'] for x in gtl.values()); correct=c['correct']; miss=c['miss']; wrong=len(gtl)-correct-miss
 summary={'method':name,'threshold':threshold,'instances':len(gtl),'correct':correct,'wrong':wrong,'miss':miss,'correct_rate':round(correct/len(gtl),6),'tile_candidates':sum(len(v) for v in candidates(batches,threshold,a.y_pad).values()),'fused_candidates':len(fused)}
 classrows=[]
 for cid in sorted({x['category_id'] for x in gtl.values()}):
  rows=[x for x in gtl.values() if x['category_id']==cid]; cc=Counter(x['outcome'] for x in rows); co=cc['correct']; mi=cc['miss']; wr=len(rows)-co-mi
  classrows.append({'method':name,'class_id':cid,'class_name':NAMES.get(cid,str(cid)),'gt_instances':len(rows),'correct':co,'wrong':wr,'miss':mi,'correct_rate':round(co/len(rows),6)})
 fuse.write_report(outdir,NAMES|{i:str(i) for i in range(1,23) if i not in NAMES},gtl,predl,fused,None)
 ev.write_csv(outdir/'summary.csv',[summary]); ev.write_csv(outdir/'per_class_cwm.csv',classrows)
 return summary

def main():
 a=args(); a.output_dir.mkdir(parents=True,exist_ok=True)
 strip=canon_strip(fw.load_json(a.strip_boxes)); raw,cal,width=infer(a)
 results=[]
 results.append(evaluate('D-FINE-S raw',raw,a.raw_threshold,strip,width,a,a.output_dir/'raw'))
 for t in [float(x) for x in a.fracal_thresholds.split(',')]: results.append(evaluate(f'FRACAL t={t:.3f}',cal,t,strip,width,a,a.output_dir/f'fracal_t{t:.3f}'))
 ev.write_csv(a.output_dir/'threshold_summary.csv',results)
 (a.output_dir/'metadata.json').write_text(json.dumps({'protocol':'cyclic strip fusion, same 13 grouped classes and exclusions as manuscript A0','note':'FRACAL frozen post-calibration; no detector training','args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}},ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__': main()