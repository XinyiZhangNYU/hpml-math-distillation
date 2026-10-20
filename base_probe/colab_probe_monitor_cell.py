# Monitor HPML base probe (read-only).
import subprocess,json
from pathlib import Path
R=Path('/content/hpml_base_probe')
print('PIPELINE_EXIT',probe_proc.poll())
for name in ['install.log','controller.log']:
    p=R/name
    if p.exists():print(name.upper(),p.read_text()[-800:])
for name in ['9b-base_status.json','9b-base_summary.json','9b-base_error.json']:
    p=R/name
    if p.exists():print(name,p.read_text()[-4000:])
p=R/'9b-base.log'
if p.exists():print('PROBE_LOG_TAIL',p.read_text()[-2000:])
print('LIVE_GPU',subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip())
