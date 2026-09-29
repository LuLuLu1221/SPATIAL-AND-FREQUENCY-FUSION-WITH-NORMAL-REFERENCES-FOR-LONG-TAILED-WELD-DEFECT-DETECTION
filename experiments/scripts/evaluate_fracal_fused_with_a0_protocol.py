"""Re-evaluate fused FRACAL candidates with the manuscript A0 union-coverage protocol."""
from __future__ import annotations
import ast,csv,sys
from pathlib import Path
ROOT=Path(r'D:\1\项目论文'); sys.path.insert(0,str(ROOT/'zhwk_project'/'scripts'))
import train_ibo_group_evidence_fusion_v2 as ev

def read(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
 base=ROOT/'zhwk_runs'/'fracal_dfine_s_cyclic_fusion_1122_20260923_seed0'; out=base/'a0_union_reevaluation'; out.mkdir(exist_ok=True)
 source=ev.load_json(ROOT/'zhwk_dfine_v1'/'annotations'/'instances_val.json')
 strip=ev.load_json(ROOT/'zhwk_unwrapped_150px_cyclic_tiles_v1'/'metadata'/'strip_boxes_val.json')
 rows=[]
 for folder in sorted(x for x in base.iterdir() if x.is_dir() and (x/'fused_strip_instances.csv').exists()):
  selected=[]
  for r in read(folder/'fused_strip_instances.csv'):
   x,y,w,h=(float(v) for v in r['strip_box_xywh'].split(';'))
   selected.append({'source_image_id':int(r['source_image_id']),'class_id':int(r['category_id']),'class_name':r['category_name'],'global_box':[x,y,x+w,y+h]})
  summary,classes=ev.evaluate_instance_from_selected(selected,source,strip,.20,.50,{11,17,19,21,22},ev.parse_match_families(['0,2,3','12,14,15,18']),folder.name)
  rows.append(summary)
  ev.write_csv(out/f'{folder.name}_per_class.csv',classes)
 ev.write_csv(out/'summary.csv',rows)
 print(rows)
if __name__=='__main__':main()