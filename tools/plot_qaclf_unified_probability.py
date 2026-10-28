"""Export the common-window probability figure and the updated ranking table."""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"result/qaclf-nested-v1-unified-evaluation-20261027"
OUT=ROOT/"figures/qaclf_nested_v1_20261027"
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(Path.home()/".codex/skills/nature-figure/scripts"))
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None

ORDER=["BLCA","BRCA","GBMLGG","LUAD","UCEC"]
METHODS=["ABMIL","TransMIL","ILRA","DSMIL","VLSA","MCAT","CAT-with-TPL","QACLF"]
COLORS={"VLSA":"#8A8F98","QACLF":"#D55E00","VLSA same projection":"#007D7A"}
fold=pd.read_csv(DATA/"all_fold_metrics.csv")
cohort=pd.read_csv(DATA/"dataset_metrics.csv")
macro=pd.read_csv(DATA/"macro_metrics.csv").set_index("method")
old=pd.read_csv(ROOT/"result/qaclf-eight-method-comparison/eight_method_dataset_summary.csv")
for method,key in [("MCAT","MCAT-CONCH"),("CAT-with-TPL","CAT-TPL-CONCH")]:
    f=fold[fold.method.eq(method)]
    for d in ORDER:
        values=f.loc[f.dataset.eq(d),"c_index"]
        old.loc[old.method.eq(key),d]=values.mean()
        old.loc[old.method.eq(key),d+"_std"]=values.std(ddof=0)
    old.loc[old.method.eq(key),"MACRO"]=macro.loc[method,"c_index"]
old.to_csv(DATA/"updated_eight_method_dataset_summary.csv",index=False)

contract={
    "backend":"Python/matplotlib", "archetype":"quantitative grid",
    "claim":"QACLF has the lowest observed macro IBS over 12-60 months; the pathology-only mapping control explains much of the improvement over native VLSA.",
    "panels":{"A":"Cohort heterogeneity of integrated probability error", "B":"Error at a fixed clinical horizon", "C":"All eight models and mapping control", "D":"Descriptive calibration non-rejection, not proof of calibration"},
    "data":"All 225 fold rows retained; equal-fold then equal-cohort aggregation; no exclusions.",
    "inference":"Descriptive point estimates; no new significance tests or confidence intervals.",
    "size_inches":[7.2,5.2],"font_floor_pt":7,"exports":["png600dpi","pdf","svg"],
    "collision_qa":"Matplotlib display bounding boxes plus PDF font audit; PyMuPDF collision tool unavailable (module fitz missing)."
}
(OUT/"figure_contract.json").write_text(json.dumps(contract,indent=2),encoding="utf-8")
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":8,"axes.linewidth":.8,"svg.fonttype":"none","pdf.fonttype":42})
fig=plt.figure(figsize=(7.2,5.2),facecolor="white")
axes=[fig.add_axes(b) for b in [[.15,.59,.33,.28],[.64,.59,.33,.28],[.15,.16,.33,.28],[.64,.16,.33,.28]]]
labels={"VLSA":"VLSA","QACLF":"QACLF","VLSA same projection":"VLSA + same projection"}
for method,color in COLORS.items():
    f=cohort[cohort.method.eq(method)].set_index("dataset").loc[ORDER]
    for ax,metric in zip(axes[:2],["ibs_12_60","brier_36"]):
        ax.plot(range(5),f[metric],"o-",color=color,lw=1,ms=3,label=labels[method])
    counts=fold[fold.method.eq(method)].assign(ok=lambda x:x.d_calibration_p_admin60.ge(.05)).groupby("dataset").ok.sum().reindex(ORDER)
    offset={"VLSA":-.16,"QACLF":0,"VLSA same projection":.16}[method]
    axes[3].scatter(np.arange(5)+offset,counts,color=color,s=22,marker={"VLSA":"o","QACLF":"D","VLSA same projection":"s"}[method])
for ax in [axes[0],axes[1],axes[3]]:
    ax.set_xticks(range(5));ax.set_xticklabels(ORDER,fontsize=7)
    ax.set_xlim(-.4,4.4)
axes[0].set_ylabel("IBS [12, 60] months")
axes[1].set_ylabel("Brier score at 36 months")
axes[0].set_ylim(0,.32);axes[1].set_ylim(0,.35)
axes[3].set_ylabel("D-cal non-rejected folds / 5")
axes[3].set_ylim(-.2,5.5);axes[3].set_yticks(range(6))
all_methods=METHODS+["VLSA same projection"]
for i,method in enumerate(all_methods):
    axes[2].scatter(macro.loc[method,"ibs_12_60"],i,color=COLORS.get(method,"#52656D"),s=25,marker="D" if method=="QACLF" else "o")
axes[2].set_yticks(range(9));axes[2].set_yticklabels(["VLSA + projection" if m=="VLSA same projection" else m for m in all_methods],fontsize=7)
axes[2].invert_yaxis();axes[2].set_xlim(.145,.215);axes[2].set_xticks([.15,.17,.19,.21]);axes[2].set_xlabel("Macro IBS [12, 60] months")
for i,(ax,title) in enumerate(zip(axes,["Integrated error by cohort","Fixed-horizon error","Eight models and mapping control","Calibration diagnostic at 60 months"])):
    ax.set_title(title,fontsize=8.5,pad=8)
    ax.spines[["top","right"]].set_visible(False);ax.tick_params(labelsize=7,length=2.5)
    ax.grid(axis="x" if i==2 else "y",color="#DADCE0",lw=.5);ax.set_axisbelow(True)
    fig.text(ax.get_position().x0-.058,ax.get_position().y1+.025,"ABCD"[i],fontsize=10,weight="bold")
handles,labs=axes[0].get_legend_handles_labels()
fig.legend(handles,labs,loc="upper center",bbox_to_anchor=(.53,.985),ncol=3,frameon=False,fontsize=8,columnspacing=1.6)
fig.text(.085,.035,"Outer-test folds; censoring weights from outer training only. Points are descriptive summaries.",fontsize=7,color="#5F6B76")
stem=OUT/"QACLF_unified_probability"
if require_matplotlib_panel_alignment is not None:
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=list("ABCD"),row_groups=[{"id":"upper","panels":["A","B"]},{"id":"lower","panels":["C","D"]}],column_groups=[{"id":"left","panels":["A","C"]},{"id":"right","panels":["B","D"]}],json_out=str(stem)+".alignment.json",overlay_svg=str(stem)+".alignment.svg",strict=True)
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
outside=[]
for text in fig.findobj(matplotlib.text.Text):
    if not text.get_visible() or not text.get_text():continue
    box=text.get_window_extent(renderer)
    if box.x0<0 or box.y0<0 or box.x1>fig.bbox.width or box.y1>fig.bbox.height:outside.append(text.get_text())
assert not outside,outside
for ext in ["svg","pdf","png"]:fig.savefig(str(stem)+"."+ext,dpi=600,facecolor="white")
plt.close(fig)
print(stem)
