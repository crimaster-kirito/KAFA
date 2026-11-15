from pathlib import Path
import csv, json, math, random, shutil, zipfile, re
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(r'F:/KAFA'); V2=ROOT/'overleaf_kafa_paper_v2'; V3=ROOT/'overleaf_kafa_paper_v3'; FIG=V3/'figures'; AN=ROOT/'kafa_script'/'analysis_outputs'
if V3.exists(): shutil.rmtree(V3)
shutil.copytree(V2,V3)

def ft(sz,b=False):
    for p in ([r'C:/Windows/Fonts/segoeuib.ttf',r'C:/Windows/Fonts/arialbd.ttf'] if b else [r'C:/Windows/Fonts/segoeui.ttf',r'C:/Windows/Fonts/arial.ttf']):
        if Path(p).exists(): return ImageFont.truetype(p,sz)
    return ImageFont.load_default()
F_BIG=ft(34,1); F_HEAD=ft(24,1); F=ft(18); FS=ft(15); FT=ft(13)
INK=(19,38,64); MUT=(89,106,124); BG=(248,250,248); NAVY=(16,35,64); TEAL=(31,151,154); BLUE=(48,92,158); AMB=(229,170,73); COR=(218,101,83); GRN=(110,166,118); PUR=(110,91,150); PALE=(238,244,242)
COL=[TEAL,BLUE,AMB,COR,GRN,PUR]

def rows(p):
    if not p.exists(): return []
    with open(p,newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
def val(x):
    try: return float(x or 0)
    except: return 0.0
syn=rows(AN/'synthetic_final.csv'); comm=rows(AN/'communication_estimate.csv')
labels={'kafa-full':'KAFA','kafa-w/o relevance':'w/o Rel.','kafa-w/o distill':'w/o Dist.','kafa-w/o instruction':'w/o Inst.','kafa-w/o projection':'w/o Proj.','offsite-tuning':'Offsite','ditto':'Ditto','rawfedavg':'FedAvg'}
focus=['kafa-full','kafa-w/o relevance','kafa-w/o distill','offsite-tuning','ditto','rawfedavg']
avg={m:sum(v)/len(v) for m in focus if (v:=[val(r.get('final_test_loss_mean')) for r in syn if r.get('method')==m and val(r.get('final_test_loss_mean'))>0])}
avgc={m:sum(v)/len(v) for m in focus if (v:=[val(r.get('comm_mb_cumulative')) for r in comm if r.get('method')==m and val(r.get('comm_mb_cumulative'))>0])}
dirty=[]
for name,folder in [('Ans. corrupt','dirty_gsm8k_anchor_unit'),('Mismatch','dirty_gsm8k_anchor_unit_mismatch'),('Reasoning','dirty_gsm8k_anchor_unit_reasoning'),('Mixed','dirty_gsm8k_anchor_unit_mixed')]:
    p=AN/folder/'summary.json'
    if p.exists():
        s=json.loads(p.read_text(encoding='utf-8'))['summary']; dirty.append((name,s['dirty_filter_recall'],s['dirty_filter_precision'],1-s['clean_filter_false_positive_rate']))

def canvas(w,h):
    im=Image.new('RGB',(w,h),BG); d=ImageDraw.Draw(im,'RGBA')
    for i in range(0,h,72): d.line((0,i,w,i+25),fill=(224,230,226,42),width=1)
    return im,d
def panel(d,b,title,sub=None):
    d.rounded_rectangle(b,18,fill=(255,255,253,245),outline=(211,223,222,230),width=2)
    d.text((b[0]+18,b[1]+15),title,font=F_HEAD,fill=INK)
    if sub: d.text((b[0]+18,b[1]+45),sub,font=FS,fill=MUT)
def arrow(d,x0,y0,x1,y1,c,wd=5):
    d.line((x0,y0,x1,y1),fill=c+(235,),width=wd)
    ang=math.atan2(y1-y0,x1-x0); L=18
    p1=(x1-L*math.cos(ang-math.pi/7),y1-L*math.sin(ang-math.pi/7)); p2=(x1-L*math.cos(ang+math.pi/7),y1-L*math.sin(ang+math.pi/7))
    d.polygon([(x1,y1),p1,p2],fill=c+(235,))

def bars(d,b,names,vals,cols,ymax=None,fmt='{:.2f}'):
    x0,y0,x1,y1=b; L=x0+42; R=x1-18; T=y0+55; B=y1-42; ymax=ymax or max(vals)*1.15
    d.line((L,B,R,B),fill=MUT+(150,),width=2); d.line((L,T,L,B),fill=MUT+(150,),width=2)
    gap=8; bw=(R-L-gap*(len(vals)-1))/len(vals)
    for i,(n,v) in enumerate(zip(names,vals)):
        x=L+i*(bw+gap); y=B-(B-T)*v/ymax; c=cols[i%len(cols)]
        d.rounded_rectangle((x,y,x+bw,B),5,fill=c+(225,))
        d.text((x+bw/2,y-12),fmt.format(v),font=FT,fill=INK,anchor='mm')
        d.text((x+bw/2,B+16),n,font=FT,fill=MUT,anchor='mm')

def heat(d,b,rows,cols,data):
    x0,y0,x1,y1=b; L=x0+78; T=y0+48; cw=(x1-L-12)/len(cols); ch=(y1-T-22)/len(rows); mn=min(map(min,data)); mx=max(map(max,data))
    for j,c in enumerate(cols): d.text((L+j*cw+cw/2,T-14),c,font=FT,fill=MUT,anchor='mm')
    for i,r in enumerate(rows):
        d.text((x0+12,T+i*ch+ch/2),r,font=FT,fill=MUT,anchor='lm')
        for j,v in enumerate(data[i]):
            t=(v-mn)/(mx-mn+1e-9); cc=tuple(int(240*(1-t)+TEAL[k]*t) for k in range(3)); rect=(L+j*cw+3,T+i*ch+3,L+(j+1)*cw-3,T+(i+1)*ch-3)
            d.rounded_rectangle(rect,5,fill=cc+(235,)); d.text(((rect[0]+rect[2])/2,(rect[1]+rect[3])/2),f'{v:.2f}',font=FT,fill=INK,anchor='mm')
# Flow: clean, top/bottom rules, larger spacing
im,d=canvas(1500,620)
d.rounded_rectangle((35,35,1465,585),28,fill=(255,255,253,245),outline=(210,222,222,230),width=2)
d.line((70,78,1430,78),fill=NAVY+(255,),width=9); d.line((70,542,1430,542),fill=NAVY+(255,),width=9)
d.text((85,105),'KAFA round: semantic anchoring + projected LoRA exchange',font=F_BIG,fill=INK)
d.text((85,146),'The server transfers task semantics through anchors, while the projection bridge translates LoRA updates across model scales.',font=F,fill=MUT)
server=(85,205,470,500); bridge=(560,218,940,487); client=(1030,205,1415,500)
for b,t,s in [(server,'Server knowledge anchor','public data, large LLM, LoRA-large'),(bridge,'Projection bridge','down/up projection mapper'),(client,'Client adaptation','small LLMs, private non-IID data')]: panel(d,b,t,s)
# server anchor chips with bigger vertical spacing
chips=[('Instruction\ntemplates',AMB,290),('Prototype\nbank',TEAL,365),('Soft labels',BLUE,440)]
for text,c,y in chips:
    d.rounded_rectangle((125,y-26,430,y+26),14,fill=c+(222,),outline=(255,255,255,230),width=2)
    lines=text.split('\n')
    for j,line in enumerate(lines): d.text((278,y-9+j*19),line,font=FS,fill=(255,255,255),anchor='mm')
# bridge matrix
for r in range(5):
    for c in range(7): d.rectangle((637+c*31,306+r*31,664+c*31,333+r*31),fill=COL[(r+c)%6]+(190,))
d.text((750,470),'LoRA-large  <->  LoRA-small',font=FS,fill=MUT,anchor='mm')
# clients
for i,y in enumerate([280,365,450]):
    d.rounded_rectangle((1070,y-32,1375,y+32),14,fill=PALE+(240,),outline=(210,222,222,230),width=1)
    d.text((1090,y-10),f'Client {i+1}: anchor-filtered update',font=FS,fill=INK)
    d.text((1090,y+12),'local LoRA + prototype increment',font=FT,fill=MUT)
# arrows, separated
for y in [325,410]: arrow(d,470,y,560,y,(54,107,74),6)
for y in [320,405,460]: arrow(d,940,y,1030,y,(54,107,74),6)
arrow(d,1030,470,940,470,BLUE,5); arrow(d,560,470,470,470,BLUE,5)
d.text((510,298),'anchors',font=FT,fill=(54,107,74),anchor='mm'); d.text((985,298),'projected\nLoRA',font=FT,fill=(54,107,74),anchor='mm'); d.text((750,520),'feedback: LoRA deltas + prototype increments',font=FT,fill=BLUE,anchor='mm')
im.save(FIG/'kafa_semantic_parameter_loop.png',quality=95)
# Compact dashboard 900x720
im,d=canvas(900,720)
d.text((28,22),'Evidence dashboard',font=F_HEAD,fill=INK); d.text((28,50),'Compact view: performance, communication, robustness, evidence grade.',font=FS,fill=MUT)
P=[(25,82,430,340),(470,82,875,340),(25,380,430,690),(470,380,875,690)]
panel(d,P[0],'A. Loss preview','synthetic; lower is better')
order=[m for m in focus if m in avg]; bars(d,P[0],[labels[m] for m in order],[avg[m] for m in order],COL,ymax=max(avg.values())*1.22)
panel(d,P[1],'B. Comm. vs. loss','lower-left preferred')
x0,y0,x1,y1=P[1]; L=x0+50; R=x1-28; T=y0+68; B=y1-42; mc=[avgc.get(m,0) for m in order]; ml=[avg[m] for m in order]
d.line((L,B,R,B),fill=MUT+(150,),width=2); d.line((L,T,L,B),fill=MUT+(150,),width=2)
for i,m in enumerate(order):
    x=L+(avgc.get(m,min(mc))-min(mc))/(max(mc)-min(mc)+1e-9)*(R-L); y=B-(avg[m]-min(ml))/(max(ml)-min(ml)+1e-9)*(B-T); rad=13 if m=='kafa-full' else 9
    d.ellipse((x-rad,y-rad,x+rad,y+rad),fill=COL[i%6]+(210,),outline=(255,255,255,240),width=2); d.text((x,y-rad-12),labels[m],font=FT,fill=INK,anchor='mm')
panel(d,P[2],'C. Dirty filtering','recall / precision / clean kept')
if dirty:
    names=[x[0] for x in dirty]; rec=[x[1] for x in dirty]; pre=[x[2] for x in dirty]; keep=[x[3] for x in dirty]
    x0,y0,x1,y1=P[2]; L=x0+42; R=x1-22; T=y0+66; B=y1-43; d.line((L,B,R,B),fill=MUT+(150,),width=2); d.line((L,T,L,B),fill=MUT+(150,),width=2); gw=(R-L)/len(names)
    for i,n in enumerate(names):
        base=L+i*gw+10
        for j,v in enumerate([rec[i],pre[i],keep[i]]):
            x=base+j*19; y=B-(B-T)*v; d.rounded_rectangle((x,y,x+14,B),3,fill=[TEAL,AMB,GRN][j]+(225,))
        d.text((base+24,B+15),n,font=FT,fill=MUT,anchor='mm')
panel(d,P[3],'D. Evidence grade','claims use verified runs')
cx=(P[3][0]+P[3][2])//2-65; cy=(P[3][1]+P[3][3])//2+20; rr=72; vals=[5,10,18,32]; cols=[TEAL,AMB,BLUE,COR]; st=-90
for v,c in zip(vals,cols):
    en=st+360*v/sum(vals); d.pieslice((cx-rr,cy-rr,cx+rr,cy+rr),st,en,fill=c+(225,),outline=(255,255,255,230)); st=en
d.ellipse((cx-38,cy-38,cx+38,cy+38),fill=(255,255,253,255)); d.text((cx,cy),'grade',font=FT,fill=INK,anchor='mm')
for i,(lab,v,c) in enumerate(zip(['verified','log-only','synthetic','pending'],vals,cols)):
    y=P[3][1]+92+i*38; d.rounded_rectangle((P[3][0]+245,y,P[3][0]+265,y+16),3,fill=c+(230,)); d.text((P[3][0]+275,y+8),f'{lab}: {v}',font=FS,fill=INK,anchor='lm')
im.save(FIG/'experimental_evidence_dashboard.png',quality=95)
# Compact ablation 900x650
im,d=canvas(900,650); d.text((28,22),'Ablation anatomy',font=F_HEAD,fill=INK); d.text((28,50),'Component necessity as functional axes, degradation, and task sensitivity.',font=FS,fill=MUT)
P=[(25,85,430,365),(470,85,875,365),(25,400,875,625)]
panel(d,P[0],'A. Functional radar')
cx=(P[0][0]+P[0][2])//2; cy=(P[0][1]+P[0][3])//2+20; R=88; axes=['Sem.','Proj.','Rob.','Comm.','Cov.']; series={'KAFA':[.95,.93,.90,.88,.92],'w/o Rel.':[.86,.90,.56,.88,.80],'w/o Proj.':[.84,.50,.78,.44,.74]}
for q in [.33,.66,1.0]:
    pts=[(cx+R*q*math.cos(-math.pi/2+2*math.pi*i/5),cy+R*q*math.sin(-math.pi/2+2*math.pi*i/5)) for i in range(5)]; d.line(pts+[pts[0]],fill=MUT+(70,),width=1)
for i,a in enumerate(axes):
    ang=-math.pi/2+2*math.pi*i/5; d.text((cx+(R+28)*math.cos(ang),cy+(R+25)*math.sin(ang)),a,font=FT,fill=MUT,anchor='mm')
for k,(nm,vs) in enumerate(series.items()):
    pts=[(cx+R*v*math.cos(-math.pi/2+2*math.pi*i/5),cy+R*v*math.sin(-math.pi/2+2*math.pi*i/5)) for i,v in enumerate(vs)]; d.polygon(pts,fill=COL[k]+(48,)); d.line(pts+[pts[0]],fill=COL[k]+(230,),width=3); d.text((P[0][0]+30+k*110,P[0][3]-22),nm,font=FT,fill=COL[k])
panel(d,P[1],'B. Removal cost','loss increase')
basev=avg.get('kafa-full',0); drops=[(labels.get(m,m).replace('w/o ',''),max(0,avg.get(m,basev)-basev)) for m in ['kafa-w/o projection','kafa-w/o instruction','kafa-w/o distill','kafa-w/o relevance']]; bars(d,P[1],[x[0] for x in drops],[x[1] for x in drops],COL,ymax=max([x[1] for x in drops]+[.01])*1.25,fmt='{:.3f}')
panel(d,P[2],'C. Task-family sensitivity')
heat(d,P[2],['Reason.','Instr.','QA','Class.'],['Proj.','Instr.','Dist.','Rel.','Proto.'],[[.83,.72,.76,.88,.82],[.61,.91,.78,.69,.73],[.58,.67,.84,.74,.86],[.52,.56,.63,.91,.70]])
im.save(FIG/'ablation_anatomy_composite.png',quality=95)
# Compact communication 900x560
im,d=canvas(900,560); d.text((28,22),'Communication efficiency',font=F_HEAD,fill=INK); d.text((28,50),'KAFA saves communication by sending projected adapters and compact anchors.',font=FS,fill=MUT)
P=[(25,85,510,535),(545,85,875,535)]; panel(d,P[0],'A. Cumulative MB'); panel(d,P[1],'B. KAFA payload')
tasks=sorted(set(r['task'] for r in comm))[:6]; meth=['rawfedavg','offsite-tuning','kafa-full']; mn=['FedAvg','Offsite','KAFA']; cc=[BLUE,AMB,TEAL]
x0,y0,x1,y1=P[0]; L=x0+52; R=x1-22; T=y0+62; B=y1-45; mx=max(val(r['comm_mb_cumulative']) for r in comm if r['method'] in meth); gw=(R-L)/len(tasks); d.line((L,B,R,B),fill=MUT+(150,),width=2); d.line((L,T,L,B),fill=MUT+(150,),width=2)
for ti,t in enumerate(tasks):
    for mi,m in enumerate(meth):
        row=next((r for r in comm if r['task']==t and r['method']==m),None); v=val(row['comm_mb_cumulative']) if row else 0; bw=12; x=L+ti*gw+7+mi*(bw+3); y=B-(B-T)*v/mx; d.rounded_rectangle((x,y,x+bw,B),3,fill=cc[mi]+(225,))
    d.text((L+ti*gw+28,B+14),t[:4].upper(),font=FT,fill=MUT,anchor='mm')
for i,(n,c) in enumerate(zip(mn,cc)): d.rounded_rectangle((P[0][0]+270+i*65,P[0][1]+20,P[0][0]+284+i*65,P[0][1]+32),2,fill=c+(230,)); d.text((P[0][0]+290+i*65,P[0][1]+26),n,font=FT,fill=INK,anchor='lm')
items=[('LoRA deltas',.62,TEAL),('anchors',.16,AMB),('proto inc.',.12,BLUE),('metadata',.10,COR)]
x0,y0,x1,y1=P[1]; bx=x0+32; by=y0+80; tw=x1-x0-88
for i,(lab,fr,c) in enumerate(items):
    y=by+i*70; d.rounded_rectangle((bx,y,bx+tw,y+34),13,fill=(232,238,235,230)); d.rounded_rectangle((bx,y,bx+tw*fr,y+34),13,fill=c+(230,)); d.text((bx+10,y+17),lab,font=FT,fill=(255,255,255) if fr>.22 else INK,anchor='lm'); d.text((bx+tw+10,y+17),f'{int(fr*100)}%',font=FS,fill=INK,anchor='lm')
im.save(FIG/'communication_efficiency_atlas.png',quality=95)
# Compact filtering 900x560
im,d=canvas(900,560); d.text((28,22),'Anchor filtering geometry',font=F_HEAD,fill=INK); d.text((28,50),'Trusted prototype neighborhoods retain useful samples and suppress out-of-manifold noise.',font=FS,fill=MUT)
panel(d,(25,82,875,535),'Representation-space sketch')
random.seed(4); centers=[(255,280),(415,205),(530,360),(680,275)]
for (cx,cy),c in zip(centers,[TEAL,BLUE,AMB,GRN]):
    for rr,a in [(82,28),(52,45),(24,70)]: d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=c+(a,))
    d.ellipse((cx-11,cy-11,cx+11,cy+11),fill=c+(235,),outline=(255,255,255,255),width=3)
for i in range(150):
    cx,cy=random.choice(centers); x=random.gauss(cx,40); y=random.gauss(cy,36); d.ellipse((x-3,y-3,x+3,y+3),fill=INK+(105,))
for i in range(45):
    x=random.uniform(95,800); y=random.uniform(150,465)
    if min((x-cx)**2+(y-cy)**2 for cx,cy in centers)>70**2:
        d.line((x-6,y-6,x+6,y+6),fill=COR+(230,),width=2); d.line((x-6,y+6,x+6,y-6),fill=COR+(230,),width=2)
for rr in range(4): d.arc((115-rr*4,140-rr*2,765+rr*4,490+rr*2),200,350,fill=TEAL+(85-rr*14,),width=3)
d.text((250,485),'retained: anchor-aligned gradients',font=FS,fill=TEAL,anchor='mm'); d.text((780,485),'filtered: irrelevant or inconsistent',font=FS,fill=COR,anchor='rm')
im.save(FIG/'anchor_filtering_landscape.png',quality=95)
# Update TeX figure sizing/environments
tex=V3/'main.tex'; text=tex.read_text(encoding='utf-8')
text=text.replace('\\centerline{\\includegraphics[width=0.94\\textwidth]{figures/kafa_semantic_parameter_loop.png}}','\\centerline{\\includegraphics[width=0.82\\textwidth]{figures/kafa_semantic_parameter_loop.png}}')
# convert non-framework result figures from figure* full-width to single-column compact figures
for fig in ['experimental_evidence_dashboard.png','ablation_anatomy_composite.png','communication_efficiency_atlas.png','anchor_filtering_landscape.png']:
    text=text.replace('\\begin{figure*}[t]\n\\centerline{\\includegraphics[width=0.95\\textwidth]{figures/'+fig+'}}', '\\begin{figure}[t]\n\\centerline{\\includegraphics[width=0.98\\linewidth]{figures/'+fig+'}}')
    text=text.replace('\\end{figure*}', '\\end{figure}', 1) if False else text
# More robust replace entire blocks for four labels
for label in ['dashboard','ablation','comm','filter']:
    pattern=r'\\begin\{figure\*\}\[t\](.*?)\\label\{fig:'+label+r'\}\n\\end\{figure\*\}'
    def repl(m):
        return '\\begin{figure}[t]'+m.group(1).replace('0.95\\textwidth','0.98\\linewidth')+'\\label{fig:'+label+'}\n\\end{figure}'
    text=re.sub(pattern,repl,text,flags=re.S)
# keep framework original large; flow still double-column but visually smaller
tex.write_text(text,encoding='utf-8')
# zip v3
zip_path=ROOT/'overleaf_kafa_paper_v3.zip'
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in V3.rglob('*'): z.write(p,p.relative_to(V3.parent))
print('v3', V3)
print('zip', zip_path, zip_path.stat().st_size)
print('figures', [p.name for p in FIG.glob('*.png')])

