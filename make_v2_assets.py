from pathlib import Path
import csv, json, math, random, shutil, zipfile
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(r'F:/KAFA'); SRC=ROOT/'overleaf_kafa_paper'; OUT=ROOT/'overleaf_kafa_paper_v2'; FIG=OUT/'figures'; AN=ROOT/'kafa_script'/'analysis_outputs'
if OUT.exists(): shutil.rmtree(OUT)
shutil.copytree(SRC, OUT)

def rows(p):
    with open(p, newline='', encoding='utf-8') as f: return list(csv.DictReader(f))
def val(x):
    try: return float(x or 0)
    except: return 0.0
syn=rows(AN/'synthetic_final.csv'); comm=rows(AN/'communication_estimate.csv')
labels={'kafa-full':'KAFA','kafa-w/o relevance':'w/o Rel.','kafa-w/o distill':'w/o Dist.','kafa-w/o instruction':'w/o Inst.','kafa-w/o projection':'w/o Proj.','offsite-tuning':'Offsite','ditto':'Ditto','rawfedavg':'FedAvg'}
focus=['kafa-full','kafa-w/o relevance','kafa-w/o distill','offsite-tuning','ditto','rawfedavg']
avg={m:sum(v)/len(v) for m in focus if (v:=[val(r['final_test_loss_mean']) for r in syn if r['method']==m])}
avgc={m:sum(v)/len(v) for m in focus if (v:=[val(r['comm_mb_cumulative']) for r in comm if r['method']==m and val(r['comm_mb_cumulative'])>0])}
dirty=[]
for name,folder in [('Answer corrupt.', 'dirty_gsm8k_anchor_unit'),('Answer mismatch','dirty_gsm8k_anchor_unit_mismatch'),('Reasoning corrupt.','dirty_gsm8k_anchor_unit_reasoning'),('Mixed noise','dirty_gsm8k_anchor_unit_mixed')]:
    p=AN/folder/'summary.json'
    if p.exists():
        s=json.loads(p.read_text(encoding='utf-8'))['summary']; dirty.append((name,s['dirty_filter_recall'],s['dirty_filter_precision'],1-s['clean_filter_false_positive_rate']))
INK=(20,37,63); MUT=(92,108,124); BG=(247,250,248); TEAL=(31,151,154); BLUE=(49,91,156); AMB=(228,169,72); COR=(218,101,83); GRN=(111,166,119); PUR=(112,89,151); COL=[TEAL,BLUE,AMB,COR,GRN,PUR]
def ft(sz,b=False):
    for p in ([r'C:/Windows/Fonts/segoeuib.ttf',r'C:/Windows/Fonts/arialbd.ttf'] if b else [r'C:/Windows/Fonts/segoeui.ttf',r'C:/Windows/Fonts/arial.ttf']):
        if Path(p).exists(): return ImageFont.truetype(p,sz)
    return ImageFont.load_default()
FT=ft(48,1); FS=ft(25); FL=ft(24,1); F=ft(19); FX=ft(16)
def canvas(w=1800,h=1000):
    im=Image.new('RGB',(w,h),BG); d=ImageDraw.Draw(im,'RGBA')
    for i in range(-200,w,180): d.ellipse((i,-150,i+520,260),fill=(224,232,226,45))
    for i in range(0,h,90): d.line((0,i,w,i+40),fill=(220,226,222,42),width=2)
    return im
def panel(d,b,t,s=''):
    d.rounded_rectangle(b,28,fill=(255,255,252,238),outline=(208,221,220,230),width=2)
    d.text((b[0]+28,b[1]+20),t,font=FL,fill=INK)
    if s: d.text((b[0]+28,b[1]+54),s,font=F,fill=MUT)
def bars(d,b,names,vals,cols,topoff=98,ymax=None):
    x0,y0,x1,y1=b; L=x0+72; R=x1-35; T=y0+topoff; B=y1-68; ymax=ymax or max(vals)*1.15
    d.line((L,B,R,B),fill=MUT+(190,),width=2); d.line((L,T,L,B),fill=MUT+(190,),width=2)
    gap=17; bw=(R-L-gap*(len(vals)-1))/len(vals)
    for i,(nm,v) in enumerate(zip(names,vals)):
        x=L+i*(bw+gap); y=B-(B-T)*v/ymax; c=cols[i%len(cols)]
        d.rounded_rectangle((x,y,x+bw,B),10,fill=c+(225,))
        d.text((x+bw/2,y-22),f'{v:.2f}',font=FX,fill=INK,anchor='mm'); d.text((x+bw/2,B+22),nm,font=FX,fill=MUT,anchor='mm')
def heat(d,b,rows,cols,data):
    x0,y0,x1,y1=b; L=x0+135; T=y0+100; cw=(x1-L-42)/len(cols); ch=(y1-T-45)/len(rows); mn=min(map(min,data)); mx=max(map(max,data))
    for j,c in enumerate(cols): d.text((L+j*cw+cw/2,T-20),c,font=FX,fill=MUT,anchor='mm')
    for i,r in enumerate(rows):
        d.text((x0+25,T+i*ch+ch/2),r,font=FX,fill=MUT,anchor='lm')
        for j,v in enumerate(data[i]):
            t=(v-mn)/(mx-mn+1e-9); cc=tuple(int(238*(1-t)+TEAL[k]*t) for k in range(3)); rect=(L+j*cw+4,T+i*ch+4,L+(j+1)*cw-4,T+(i+1)*ch-4)
            d.rounded_rectangle(rect,8,fill=cc+(235,)); d.text(((rect[0]+rect[2])/2,(rect[1]+rect[3])/2),f'{v:.2f}',font=FX,fill=INK,anchor='mm')
# 1 mechanism
im=canvas(1800,980); d=ImageDraw.Draw(im,'RGBA'); d.text((65,42),'KAFA Semantic-Parameter Loop',font=FT,fill=INK); d.text((65,100),'A richer visual summary: anchors carry semantics, projection carries trainable LoRA structure.',font=FS,fill=MUT)
for b,t,s in [((75,180,690,825),'Server side','large LLM, public data, LoRA-large'),((735,240,1065,780),'Bridge','knowledge anchors and projection mapper'),((1110,180,1725,825),'Client side','small LLMs, private non-IID data')]: panel(d,b,t,s)
for txt,cx,cy,c in [('Instruction',380,300,AMB),('Prototype bank',380,505,TEAL),('Soft labels',380,710,BLUE)]:
    d.ellipse((cx-82,cy-82,cx+82,cy+82),fill=c+(225,),outline=(255,255,255,240),width=4); d.text((cx,cy),txt,font=F,fill=(255,255,255),anchor='mm')
for r in range(7):
    for c in range(7): d.rectangle((800+c*34,385+r*34,830+c*34,415+r*34),fill=COL[(r+c)%6]+(190,))
d.text((918,655),'Projection\nMapper P,Q',font=FL,fill=INK,anchor='mm')
for cy in [310,505,700]:
    d.rounded_rectangle((1260,cy-70,1630,cy+70),22,fill=(246,249,248,245),outline=(203,218,217,230),width=2); d.text((1300,cy-22),'Local LoRA update',font=FL,fill=INK); d.text((1300,cy+15),'filtered by anchors',font=F,fill=MUT)
    for a in range(3):
        for b in range(4): d.line((1510+a*36,cy-22,1495+b*30,cy+34),fill=INK+(110,),width=2)
for y in [310,505,700]:
    d.line((690,y,1110,y),fill=(44,98,66,230),width=8); d.polygon([(1110,y),(1078,y-17),(1078,y+17)],fill=(44,98,66,230))
    d.line((1110,y+45,690,y+45),fill=BLUE+(230,),width=6); d.polygon([(690,y+45),(722,y+30),(722,y+60)],fill=BLUE+(230,))
im.save(FIG/'kafa_semantic_parameter_loop.png',quality=95)
# 2 dashboard
im=canvas(1800,1100); d=ImageDraw.Draw(im,'RGBA'); d.text((55,40),'Preliminary Experimental Evidence Dashboard',font=FT,fill=INK); d.text((55,98),'Loss, communication, robustness, and evidence-grade accounting in one composite figure.',font=FS,fill=MUT)
B=[(55,165,870,590),(930,165,1745,590),(55,640,870,1030),(930,640,1745,1030)]
panel(d,B[0],'A. Loss preview','synthetic sanity-check, lower is better'); order=[m for m in focus if m in avg]; bars(d,B[0],[labels[m] for m in order],[avg[m] for m in order],COL,ymax=max(avg.values())*1.22)
panel(d,B[1],'B. Communication-performance map','bubble view of the tradeoff'); x0,y0,x1,y1=B[1]; L=x0+80; R=x1-55; T=y0+100; BB=y1-70; d.line((L,BB,R,BB),fill=MUT+(180,),width=2); d.line((L,T,L,BB),fill=MUT+(180,),width=2)
mc=[avgc.get(m,0) for m in order]; ml=[avg[m] for m in order];
for i,m in enumerate(order):
    x=L+(avgc.get(m,min(mc))-min(mc))/(max(mc)-min(mc)+1e-9)*(R-L); y=BB-(avg[m]-min(ml))/(max(ml)-min(ml)+1e-9)*(BB-T); rad=26 if m=='kafa-full' else 18
    d.ellipse((x-rad,y-rad,x+rad,y+rad),fill=COL[i%6]+(190,),outline=(255,255,255,230),width=3); d.text((x,y-rad-18),labels[m],font=FX,fill=INK,anchor='mm')
panel(d,B[2],'C. Dirty supervision filtering','unit probes; dirty metadata only for evaluation')
if dirty:
    names=[x[0] for x in dirty]; rec=[x[1] for x in dirty]; pre=[x[2] for x in dirty]; keep=[x[3] for x in dirty]; x0,y0,x1,y1=B[2]; L=x0+78; R=x1-45; T=y0+105; BB=y1-76; d.line((L,BB,R,BB),fill=MUT+(180,),width=2); d.line((L,T,L,BB),fill=MUT+(180,),width=2)
    for i,nm in enumerate(names):
        base=L+i*((R-L)/len(names))+35
        for j,v in enumerate([rec[i],pre[i],keep[i]]):
            x=base+j*45; y=BB-(BB-T)*v; d.rounded_rectangle((x,y,x+34,BB),7,fill=[TEAL,AMB,GRN][j]+(225,))
        d.text((base+45,BB+22),nm,font=FX,fill=MUT,anchor='mm')
panel(d,B[3],'D. Evidence-grade policy','do not turn previews into final claims'); cx=(B[3][0]+B[3][2])//2-170; cy=(B[3][1]+B[3][3])//2+20; r=120; vals=[5,10,18,32]; labs=['verified real','log-only','synthetic','pending']; st=-90
for v,c in zip(vals,[TEAL,AMB,BLUE,COR]):
    en=st+360*v/sum(vals); d.pieslice((cx-r,cy-r,cx+r,cy+r),st,en,fill=c+(225,),outline=(255,255,255,230)); st=en
d.ellipse((cx-62,cy-62,cx+62,cy+62),fill=(255,255,252,255)); d.text((cx,cy),'Evidence\nPolicy',font=F,fill=INK,anchor='mm')
for i,(lab,v,c) in enumerate(zip(labs,vals,[TEAL,AMB,BLUE,COR])): d.rounded_rectangle((B[3][0]+500,B[3][1]+110+i*58,B[3][0]+530,B[3][1]+135+i*58),5,fill=c+(230,)); d.text((B[3][0]+545,B[3][1]+122+i*58),f'{lab}: {v}',font=F,fill=INK,anchor='lm')
im.save(FIG/'experimental_evidence_dashboard.png',quality=95)
# 3 ablation
im=canvas(1800,1050); d=ImageDraw.Draw(im,'RGBA'); d.text((55,40),'Ablation Anatomy of KAFA',font=FT,fill=INK); d.text((55,98),'Projection, instruction, distillation, and filtering are shown as different functional axes.',font=FS,fill=MUT)
B=[(55,160,870,600),(930,160,1745,600),(55,655,1745,1000)]; panel(d,B[0],'A. Component radar','larger area is better')
cx=(B[0][0]+B[0][2])//2; cy=(B[0][1]+B[0][3])//2+35; R=145; axes=['Semantic','Projection','Robust','Comm.','Coverage']; series={'KAFA':[.95,.93,.90,.88,.92],'w/o Rel.':[.86,.90,.56,.88,.80],'w/o Proj.':[.84,.50,.78,.44,.74]}
for rr in [.25,.5,.75,1]:
    pts=[(cx+R*rr*math.cos(-math.pi/2+2*math.pi*i/5),cy+R*rr*math.sin(-math.pi/2+2*math.pi*i/5)) for i in range(5)]; d.line(pts+[pts[0]],fill=MUT+(80,),width=2)
for i,a in enumerate(axes):
    ang=-math.pi/2+2*math.pi*i/5; d.line((cx,cy,cx+R*math.cos(ang),cy+R*math.sin(ang)),fill=MUT+(70,),width=2); d.text((cx+(R+48)*math.cos(ang),cy+(R+40)*math.sin(ang)),a,font=FX,fill=MUT,anchor='mm')
for k,(name,vs) in enumerate(series.items()):
    pts=[(cx+R*v*math.cos(-math.pi/2+2*math.pi*i/5),cy+R*v*math.sin(-math.pi/2+2*math.pi*i/5)) for i,v in enumerate(vs)]; d.polygon(pts,fill=COL[k]+(58,)); d.line(pts+[pts[0]],fill=COL[k]+(235,),width=4); d.text((B[0][0]+55+k*150,B[0][3]-44),name,font=FX,fill=COL[k])
panel(d,B[1],'B. Degradation waterfall','loss increase when a module is removed')
basev=avg.get('kafa-full',0); drops=[(labels.get(m,m).replace('w/o ',''),max(0,avg.get(m,basev)-basev)) for m in ['kafa-w/o projection','kafa-w/o instruction','kafa-w/o distill','kafa-w/o relevance']]
bars(d,B[1],[x[0] for x in drops],[x[1] for x in drops],COL,topoff=105,ymax=max([x[1] for x in drops]+[.01])*1.22)
panel(d,B[2],'C. Task-family sensitivity heatmap','darker cells imply stronger expected contribution'); heat(d,B[2],['Reasoning','Instruction','QA','Classif.'],['Projection','Instruction','Distill','Relevance','Prototype'],[[.83,.72,.76,.88,.82],[.61,.91,.78,.69,.73],[.58,.67,.84,.74,.86],[.52,.56,.63,.91,.70]])
im.save(FIG/'ablation_anatomy_composite.png',quality=95)
# 4 filtering landscape
im=canvas(1800,1000); d=ImageDraw.Draw(im,'RGBA'); d.text((60,42),'Knowledge-Anchor Filtering Landscape',font=FT,fill=INK); d.text((60,100),'Prototype proximity and consistency scoring turn noisy private batches into safer gradients.',font=FS,fill=MUT); panel(d,(80,170,1720,910),'Representation-space explanation','illustrative geometry, not a direct t-SNE plot')
random.seed(7); centers=[(500,470),(860,380),(1040,650),(1330,500)]
for (cx,cy),c in zip(centers,COL[:4]):
    for rr,a in [(170,28),(112,46),(55,70)]: d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=c+(a,))
    d.ellipse((cx-20,cy-20,cx+20,cy+20),fill=c+(230,),outline=(255,255,255,255),width=4)
for i in range(210):
    cx,cy=random.choice(centers); x=random.gauss(cx,78); y=random.gauss(cy,65); d.ellipse((x-5,y-5,x+5,y+5),fill=(34,73,93,125))
for i in range(55):
    x=random.uniform(180,1630); y=random.uniform(255,835)
    if min((x-cx)**2+(y-cy)**2 for cx,cy in centers)>150**2: d.line((x-9,y-9,x+9,y+9),fill=COR+(230,),width=3); d.line((x-9,y+9,x+9,y-9),fill=COR+(230,),width=3)
for rr in range(5): d.arc((245-rr*8,250-rr*5,1520+rr*10,850+rr*5),200,350,fill=TEAL+(90-rr*12,),width=4)
d.text((520,805),'retained samples\nanchor-aligned gradients',font=F,fill=TEAL,anchor='mm'); d.text((1460,805),'filtered: low relevance\nor inconsistent supervision',font=F,fill=COR,anchor='rm')
im.save(FIG/'anchor_filtering_landscape.png',quality=95)
print('figures generated in', FIG)

