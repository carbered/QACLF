"""Plot retained six-method results without conflating training protocols."""

from pathlib import Path
import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path.home()/".codex/skills/nature-figure/scripts"))
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--include-ilra",action="store_true")
parser.add_argument("--include-dsmil",action="store_true",help="Include both audited ILRA and DSMIL results")
parser.add_argument("--canonical-names",action="store_true",help="Remove input-encoder suffixes from display labels only")
parser.add_argument("--output-stem",type=Path)
parser.add_argument("--model-label",default="FICPF",help="Display name only; retain original result keys")
parser.add_argument("--data-summary",type=Path)
parser.add_argument("--nested-rerun",action="store_true")
args=parser.parse_args()
if args.include_dsmil:
    OUT=ROOT/"figures/ficpf_20260821/FICPF_eight_method_comparison"
    DATA=ROOT/"result/ficpf-eight-method-comparison/eight_method_dataset_summary.csv"
else:
    OUT=ROOT/("figures/ficpf_20260821/FICPF_seven_method_comparison" if args.include_ilra else "figures/ficpf_20260821/FICPF_six_method_comparison")
    DATA=ROOT/("result/ficpf-seven-method-comparison/seven_method_dataset_summary.csv" if args.include_ilra else "result/ficpf-six-method-comparison/six_method_dataset_summary.csv")
if args.output_stem:
    OUT=args.output_stem if args.output_stem.is_absolute() else ROOT/args.output_stem
if args.data_summary:
    DATA=args.data_summary
OUT.parent.mkdir(parents=True,exist_ok=True)
BOOT=ROOT/"result/external_multimodal_comparison_20260821/unified/paired_bootstrap_unified.csv"
ORDER=["BLCA","BRCA","GBMLGG","LUAD","UCEC","MACRO"]
METHODS=[("ABMIL-CONCH","ABMIL-CONCH","#8C98A6","v"),
         ("TransMIL-CONCH","TransMIL-CONCH","#52656D","P"),
         ("Matched VLSA","Matched VLSA","#8A8F98","o"),
         ("MCAT-CONCH","MCAT-CONCH*","#0072B2","s"),
         ("CAT-TPL-CONCH","CAT-TPL-CONCH*","#009E73","^"),
         ("FICPF","FICPF","#D55E00","D")]
if args.include_ilra or args.include_dsmil:
    METHODS.insert(2,("ILRA-CONCH","ILRA-CONCH","#7A7F2B","X"))
if args.include_dsmil:
    METHODS.insert(3,("DSMIL-CONCH","DSMIL-CONCH","#6E5A83","h"))
if args.canonical_names:
    display={"ABMIL-CONCH":"ABMIL","TransMIL-CONCH":"TransMIL","ILRA-CONCH":"ILRA",
             "DSMIL-CONCH":"DSMIL","Matched VLSA":"VLSA","MCAT-CONCH":"MCAT*",
             "CAT-TPL-CONCH":"CAT-with-TPL*","FICPF":"FICPF"}
    METHODS=[(key,display[key],color,marker) for key,_,color,marker in METHODS]
METHODS=[(key,args.model_label if key=="FICPF" else label,color,marker)
         for key,label,color,marker in METHODS]
if args.nested_rerun:
    METHODS=[(key,label.rstrip("*"),color,marker) for key,label,color,marker in METHODS]

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
                     "font.size":8,"axes.linewidth":.8,"svg.fonttype":"none","pdf.fonttype":42})
frame=pd.read_csv(DATA).set_index("method")
fig=plt.figure(figsize=(7.2,3.7),facecolor="white")
left=fig.add_axes([.078,.225,.412,.545])
right=fig.add_axes([.580,.225,.390,.545])
for shift,(method,label,color,marker) in zip(np.linspace(-.27,.27,len(METHODS)),METHODS):
    left.scatter(np.arange(6)+shift,frame.loc[method,ORDER].to_numpy(dtype=float),
                 color=color,marker=marker,s=22 if method!="FICPF" else 31,
                 label=label,zorder=4)
left.axvline(4.5,color="#B8BABC",ls="--",lw=.7)
left.set_xticks(range(6));left.set_xticklabels(ORDER[:-1]+["Macro"],fontsize=7.2)
left.set_ylim(.46,.85);left.set_yticks(np.arange(.50,.851,.05))
left.set_ylabel("C-index",fontsize=8.4,labelpad=4)
left.set_title("Five-cohort comparison",fontsize=9,pad=8)
left.legend(frameon=False,ncol=2,loc="lower left",bbox_to_anchor=(-.03,1.12 if args.include_ilra or args.include_dsmil else 1.20),
            fontsize=7.2,columnspacing=.6,handletextpad=.3,borderaxespad=0,
            labelspacing=.45,markerscale=.8)
boot=pd.read_csv(BOOT)
boot=boot[boot.comparison=="ficpf_vs_matched"].set_index("dataset").loc[ORDER]
delta=boot.observed_delta.to_numpy();lower=boot.delta_ci025.to_numpy();upper=boot.delta_ci975.to_numpy()
y=np.arange(6)
for i,(d,lo,hi) in enumerate(zip(delta,lower,upper)):
    color="#202124" if i==5 else "#D55E00"
    right.plot([lo,hi],[i,i],color=color,lw=1.25)
    right.plot([lo,lo],[i-.09,i+.09],color=color,lw=1)
    right.plot([hi,hi],[i-.09,i+.09],color=color,lw=1)
    right.scatter(d,i,s=25,facecolor=color if lo>0 or i==5 else "white",edgecolor=color,zorder=4)
right.axvline(0,color="#60666B",lw=.8)
right.set_yticks(y);right.set_yticklabels(ORDER[:-1]+["Macro"],fontsize=7.3)
right.invert_yaxis();right.set_xlim(-.025,.034)
right.set_xticks([-.02,-.01,0,.01,.02,.03])
right.set_xlabel("Gain vs VLSA (95% CI)" if args.canonical_names else "Gain vs matched VLSA (95% CI)",fontsize=8.0,labelpad=4)
right.set_title("Paired bootstrap effect",fontsize=9,pad=8)
for ax in (left,right):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y" if ax is left else "x",color="#DADCE0",lw=.5)
    ax.set_axisbelow(True);ax.tick_params(labelsize=7.2,length=2.5)
fig.text(.042,.805,"A",fontsize=10,weight="bold")
fig.text(.535,.805,"B",fontsize=10,weight="bold")
footnote="* Retained MCAT/CAT-with-TPL adaptations use cohort-wide time bins; see Methods." if args.canonical_names else "* Retained MCAT/CAT adaptations use cohort-wide time bins; see Methods."
if args.nested_rerun:
    footnote="MCAT/CAT-with-TPL: inner-validation epoch selection, then outer-training refit; see Methods."
fig.text(.078,.065,footnote,fontsize=7.1,color="#5F6B76")
if require_matplotlib_panel_alignment is not None:
    require_matplotlib_panel_alignment(
        fig,axes=[left,right],panel_ids=["comparison","paired_effect"],
        row_groups=[{"id":"main_row","panels":["comparison","paired_effect"]}],
        exemptions=[{"panels":["paired_effect"],"checks":["panel-width"],
                     "reason":"Paired-effect plot has a narrower intentional span than the cohort comparison plot."}],
        json_out=str(OUT)+".alignment.json",overlay_svg=str(OUT)+".alignment.svg",strict=True)
else:
    print("Optional layout-audit tool unavailable; exporting without a new alignment report.")
fig.savefig(str(OUT)+".svg",facecolor="white")
fig.savefig(str(OUT)+".pdf",facecolor="white")
fig.savefig(str(OUT)+".png",dpi=600,facecolor="white")
plt.close(fig)
OUT.with_suffix(".provenance.json").write_text(json.dumps({
    "source_data":str(DATA),"paired_intervals":str(BOOT),
    "claim":"Retained comparisons describe current adaptations; paired primary inference concerns FICPF versus matched VLSA.",
    "archetype":"quantitative grid",
    "training_protocol_note":"Post-hoc nested MCAT/CAT rerun; other frozen baselines retained." if args.nested_rerun else "MCAT/CAT use cohort-wide discretization; not fully train-only controls.",
    "statistics":"Five-fold population SD remains in manuscript table. Plot A shows means, plot B paired patient bootstrap 95% intervals.",
    "data_integrity":"All "+str(len(METHODS))+" methods and all five cohorts plus macro means retained; no generated observations."},indent=2))
print(OUT)
