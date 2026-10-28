"""Refine the existing FICPF schematic without changing its scientific scope."""

from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Polygon, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "figures/ficpf_20260821/real_tcga_inputs/TCGA-3C-AALI"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tinted", action="store_true", help="Export a separate soft-color preview.")
parser.add_argument("--canonical-names", action="store_true", help="Use the original model names in display labels.")
parser.add_argument("--output-stem", type=Path, help="Write separate assets without replacing previous figures.")
parser.add_argument("--model-label", default="FICPF", help="Display name only; experiment IDs remain unchanged.")
parser.add_argument("--source-wsi", type=Path,
                    help="Optional original SVS for a fresh pixel/crop audit; otherwise verify bundled source-image hashes.")
args = parser.parse_args()
suffix = "_soft_color_preview" if args.tinted else ""
OUT = ROOT / ("figures/ficpf_20260821/FICPF_method_overview_refined" + suffix)
if args.output_stem:
    OUT = args.output_stem if args.output_stem.is_absolute() else ROOT / args.output_stem
OUT.parent.mkdir(parents=True, exist_ok=True)
SKILL = Path.home() / ".codex/skills/nature-figure/scripts"
sys.path.insert(0, str(SKILL))
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.6, "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.linewidth": .65,
})
fig = plt.figure(figsize=(7.1, 4.8), facecolor="white")
INK, MUTED, FAINT = "#202A33", "#606E77", "#ADB7BF"
BLUE, GREEN, AMBER = "#286AA0", "#128273", "#B47416"
RED, PURPLE = "#C34C3E", "#775096"
FILLS = {BLUE:"#EDF5FC", GREEN:"#EDF8F2", AMBER:"#FFF7E8",
         INK:"#F7F8FA", PURPLE:"#F6F5FA"}


def text(x, y, s, size=7.6, color=INK, weight="normal", ha="center", **kw):
    return fig.text(x, y, s, fontsize=size, color=color, weight=weight,
                    ha=ha, va="center", zorder=20, **kw)


def line(x1, y1, x2, y2, color=INK, lw=.8, ls="-", z=8):
    fig.add_artist(Line2D([x1, x2], [y1, y2], transform=fig.transFigure,
                         color=color, linewidth=lw, linestyle=ls, zorder=z))


def arrow(x1, y1, x2, y2, color=INK, lw=1.2, scale=7):
    fig.add_artist(FancyArrowPatch((x1, y1), (x2, y2), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=scale,
                                   color=color, linewidth=lw, zorder=12))


def boundary(x, y, w, h, color):
    fig.add_artist(Rectangle((x, y), w, h, transform=fig.transFigure,
                            facecolor=FILLS[color] if args.tinted else "none",
                            edgecolor=color, linewidth=.9,
                            linestyle=(0, (4, 3)), zorder=1))


def label(x, y, letter, title, color, size=8.7):
    text(x, y, letter, 9.8, color, "bold", ha="left")
    text(x+.025, y, title, size, INK, "bold", ha="left")


def image(rect, path, color):
    ax = fig.add_axes(rect, zorder=3)
    ax.imshow(plt.imread(path))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(color); sp.set_linewidth(.75)
    return ax


def curve(rect, kind, color):
    ax = fig.add_axes(rect, zorder=3)
    if args.tinted:
        ax.set_facecolor("none")
    if kind == "survival":
        ax.step([0,.17,.34,.54,.74,.91,1], [1,.87,.70,.51,.32,.17,.17],
                where="post", lw=1.4, color=color)
        ax.set_xticks([0,1]); ax.set_xticklabels(["0","t"])
        ax.set_yticks([0,1])
    elif kind == "ecdf":
        vals = np.array([.05,.12,.20,.27,.34,.43,.51,.61,.68,.78,.86,.95])
        if color == GREEN:
            vals += np.array([0,.01,-.01,.02,0,.01,-.01,0,.01,-.01,0,0])
        ax.step(np.r_[0,vals,1], np.r_[0,np.arange(1,13)/12,1],
                where="post", lw=1.6, color=color)
        ax.axhline(.5, color=FAINT, lw=.6, ls="--")
        ax.set_xticks([]); ax.set_yticks([0,.5,1])
    elif kind == "scale":
        x = np.linspace(-3,3,100)
        ax.plot(x, np.exp(-x*x/2), color=color, lw=1.3)
        ax.axvline(0, color=FAINT, lw=.6, ls="--")
        ax.set_xlim(-3,3); ax.set_xticks([]); ax.set_yticks([])
    elif kind == "pca":
        p = np.array([[.17,.22],[.25,.73],[.41,.40],[.58,.78],[.66,.32],[.83,.63]])
        ax.scatter(p[:,0],p[:,1],s=9,color=color)
        ax.plot([.09,.90],[.10,.86],color=color,lw=.75)
        ax.set_xticks([]); ax.set_yticks([])
    if kind != "scale":
        ax.set_xlim(0,1)
    ax.set_ylim(0,1.04)
    ax.tick_params(labelsize=6.5, length=1.5, pad=1)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(MUTED)
    return ax


def stack(x,y,w=.052):
    for k in range(4):
        yy=y+k*.025
        fig.add_artist(Rectangle((x,yy),w,.018,transform=fig.transFigure,
                                facecolor="#E9F2F8",edgecolor=BLUE,lw=.6))
        fig.add_artist(Polygon([(x,yy+.018),(x+.009,yy+.027),
                                (x+w+.009,yy+.027),(x+w,yy+.018)],
                               transform=fig.transFigure,facecolor="#BAD6E8",
                               edgecolor=BLUE,lw=.5))


def grid(x,y,color):
    for r in range(3):
        for c in range(4):
            fig.add_artist(Rectangle((x+c*.010,y+r*.013),.0085,.010,
                                    transform=fig.transFigure,
                                    facecolor=matplotlib.colors.to_rgba(color,.3+.12*((r+c)%4)),
                                    edgecolor="white",lw=.3))


# The public source record retains the original pixel-validated crop transform.
provenance = json.loads((ASSETS / "provenance.json").read_text(encoding="utf-8"))
overview = plt.imread(ASSETS / "wsi_overview.png")
if args.source_wsi:
    import tifffile
    with tifffile.TiffFile(args.source_wsi) as tiff:
        full = tiff.series[0].levels[-1].asarray()
    full_shape = list(full.shape)
    rgb = full.astype(np.int16)
    mask = (rgb.mean(axis=2)<246) & (rgb.mean(axis=2)>18) & (np.ptp(rgb,axis=2)>8)
    ys,xs = np.where(mask)
    pad_x = max(8,int((xs.max()-xs.min())*.035))
    pad_y = max(8,int((ys.max()-ys.min())*.035))
    crop = [max(0,int(xs.min())-pad_x),max(0,int(ys.min())-pad_y),
            min(full.shape[1],int(xs.max())+pad_x+1),
            min(full.shape[0],int(ys.max())+pad_y+1)]
    assert np.array_equal(np.rint(overview[:,:,:3]*255).astype(np.uint8),
                          full[crop[1]:crop[3],crop[0]:crop[2],:3])
else:
    record_path = ROOT / "figures/ficpf_model_names/FICPF_method_overview.provenance.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    for name, expected in record["input_sha256"].items():
        assert hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() == expected, name
    crop, full_shape = record["overview_crop_box"], record["overview_full_shape"]
assert overview.shape[:2] == (crop[3]-crop[1],crop[2]-crop[0])

text(.024,.961,args.model_label,13,INK,"bold",ha="left")
text(.163,.961,"Pathology–omics survival fusion",10.4,INK,"bold",ha="left")
for x,c,name in [(.62,BLUE,"Pathology"),(.758,GREEN,"Omics"),(.865,RED,"Fusion")]:
    arrow(x,.961,x+.027,.961,c,1.4)
    text(x+.033,.961,name,7.3,c,"bold",ha="left")

# Image-led pathology stream with true region registration.
boundary(.018,.530,.504,.386,BLUE)
label(.031,.892,"a","VLSA pathology branch" if args.canonical_names else "VLSA-CONCH pathology branch",BLUE)
text(.100,.851,"Patient i · H&E",7.6,BLUE,"bold")
wsi = image([.035,.655,.130,.179],ASSETS / "wsi_overview.png",BLUE)
text(.100,.626,"Whole-slide image",7.2,MUTED)
patch_axes=[]
for j in range(3):
    ax=image([.198,.771-j*.084,.046,.068],ASSETS / f"patch_0{j+1}.png",BLUE)
    patch_axes.append(ax)
    text(.185,.829-j*.084,str(j+1),7.1,BLUE,"bold")
text(.222,.851,"Regions",7.5,INK,"bold")
text(.222,.581,"896 × 896 px",7.1,MUTED)
sx,sy=full_shape[1]/provenance["base_dimensions"][0],full_shape[0]/provenance["base_dimensions"][1]
registered=[]
for j,(xx,yy) in enumerate(provenance["selected_coords"]):
    px,py=xx*sx-crop[0],yy*sy-crop[1]
    pw,ph=provenance["patch_size"]*sx,provenance["patch_size"]*sy
    wsi.add_patch(Rectangle((px,py),pw,ph,facecolor="none",edgecolor=BLUE,lw=.9,zorder=6))
    wsi.annotate(str(j+1),(px+pw/2,py+ph/2),xytext=(-4,6),
                 textcoords="offset points",fontsize=7.1,color=BLUE,weight="bold")
    fig.add_artist(ConnectionPatch((px+pw,py+ph/2),(0,.5),
                                  "data","axes fraction",axesA=wsi,axesB=patch_axes[j],
                                  color=BLUE,lw=.65,linestyle=(0,(2,2)),zorder=2))
    registered.append([float(px),float(py),float(pw),float(ph)])
stack(.270,.701)
text(.300,.851,"CONCH",7.5,INK,"bold")
text(.300,.671,"N × 512",7.7,BLUE,"bold")
arrow(.249,.743,.269,.743,BLUE)
grid(.352,.748,BLUE)
text(.372,.851,"VLSA",7.7,INK,"bold")
text(.372,.713,"VLFAN",7.2,BLUE,"bold")
text(.374,.645,"Ordinal prompts",7.1,MUTED)
text(.371,.611,"short · mid · long",7.1,MUTED)
arrow(.331,.743,.350,.743,BLUE)
arrow(.398,.743,.431,.743,BLUE)
curve([.438,.691,.062,.109],"survival",BLUE)
text(.467,.828,r"$S_p(t)$",9.3,BLUE)
text(.468,.637,r"$T_p$",10.4,BLUE)
text(.468,.604,"Mean time",7.2,MUTED)
text(.277,.556,"Pathology parameters frozen during fusion",7.6,BLUE,"bold")

# Molecular data remain a real matched input; other transform graphics are schematic.
boundary(.018,.170,.504,.345,GREEN)
label(.031,.491,"b","RNA-seq / CNV branch",GREEN)
text(.100,.444,"Patient i · omics",7.6,GREEN,"bold")
image([.035,.281,.130,.145],ASSETS / "omics_heatmap.png",GREEN)
text(.100,.254,"Matched measurements",7.1,MUTED)
text(.218,.445,"Signatures",7.4,INK,"bold")
for j,(name,col) in enumerate([("TSG","#3886B8"),("Onco","#D45A5A"),
                             ("Kinase","#DA9A31"),("Diff.","#7660A8"),
                             ("TF","#4C9B75"),("Cyt/GF","#76838B")]):
    yy=.410-j*.026
    fig.add_artist(Rectangle((.185,yy-.006),.012,.012,transform=fig.transFigure,
                            facecolor=col,edgecolor="none"))
    text(.202,yy,name,7.1,ha="left")
arrow(.167,.342,.183,.342,GREEN)
curve([.267,.301,.057,.091],"scale",GREEN)
text(.296,.445,"Train-fold\nscaling",7.4,INK,"bold",linespacing=1.35)
arrow(.244,.342,.265,.342,GREEN)
curve([.349,.301,.057,.091],"pca",GREEN)
text(.377,.445,"PCA",7.4,INK,"bold")
text(.377,.272,"d ≤ 64",7.5,GREEN,"bold")
arrow(.326,.342,.347,.342,GREEN)
grid(.431,.330,GREEN)
text(.450,.445,"L2-Cox",7.4,INK,"bold")
text(.448,.242,r"$\eta=\mathbf{w}^{\mathsf{T}}\mathbf{z}+b$",9,INK)
arrow(.409,.342,.429,.342,GREEN)
arrow(.475,.349,.498,.349,GREEN)
text(.498,.385,r"$T_o$",10.4,GREEN)
text(.498,.311,r"$-\eta$",9.3,GREEN)
text(.274,.211,"Fit scaling, PCA and Cox on training patients",7.5,GREEN,"bold")

# Main inferential step: separate train-fit maps, shared range and fixed weights.
boundary(.541,.170,.259,.746,AMBER)
label(.553,.885,"c","Scale alignment",AMBER,9.0)
text(.574,.853,"& conservative fusion",9.0,INK,"bold",ha="left")
text(.671,.810,"ECDFs fitted on training fold",7.6,AMBER,"bold")
text(.580,.765,r"Pathology score $T_p$",8.8,BLUE,ha="left")
ecdf_p=curve([.585,.638,.145,.098],"ecdf",BLUE)
text(.672,.610,r"$q_p=\widehat F_{p,\mathrm{train}}(T_p)$",10.4,BLUE)
text(.580,.562,r"Omics score $T_o$",8.8,GREEN,ha="left")
ecdf_o=curve([.585,.435,.145,.098],"ecdf",GREEN)
text(.672,.407,r"$q_o=\widehat F_{o,\mathrm{train}}(T_o)$",10.4,GREEN)
arrow(.502,.735,.576,.704,BLUE,1.4)
line(.514,.351,.550,.462,GREEN,1.4)
arrow(.550,.462,.577,.462,GREEN,1.4)
line(.732,.689,.774,.689,BLUE)
arrow(.774,.689,.774,.350,BLUE)
line(.732,.486,.750,.486,GREEN)
arrow(.750,.486,.750,.350,GREEN)
line(.565,.349,.783,.349,AMBER,1.0)
if args.tinted:
    fig.add_artist(Rectangle((.548,.266),.245,.079,transform=fig.transFigure,
                            facecolor="#FBE9E6",edgecolor="none",zorder=2))
text(.671,.317,r"$q=0.80q_p+0.20q_o$",12,RED,"bold")
text(.671,.282,"Fixed conservative weights",7.9,RED,"bold")
text(.671,.242,r"Omics missing: $q=q_p$",9.2,INK)
text(.671,.204,"Higher q → longer survival",7.9,RED,"bold")

# One patient's scores may have different ranks even after scale alignment.
boundary(.817,.566,.168,.350,INK)
label(.829,.884,"d","Why align",INK,8.5)
text(.852,.853,"scales?",8.5,INK,"bold",ha="left")
text(.901,.815,"Patient i · raw scores",7.2,MUTED)
text(.855,.776,r"$T_p$ (time)",8.8,BLUE,ha="left")
text(.855,.718,r"$T_o$ ($-\eta$)",8.8,GREEN,ha="left")
for yy,col,pos in [(.750,BLUE,.936),(.692,GREEN,.873)]:
    line(.844,yy,.964,yy,MUTED,.7)
    fig.add_artist(Line2D([pos],[yy],transform=fig.transFigure,marker="o",
                         markersize=3.5,color=col,zorder=10))
arrow(.971,.748,.971,.650,AMBER,.9,6)
line(.844,.639,.964,.639,MUTED,.8)
for xx,col,lab in [(.897,BLUE,r"$q_p$"),(.927,GREEN,r"$q_o$")]:
    fig.add_artist(Line2D([xx],[.639],transform=fig.transFigure,marker="o",
                         markersize=3.5,color=col,zorder=10))
    text(xx,.664,lab,9.3,col)
text(.840,.655,"0",6.7,MUTED)
text(.955,.655,"1",6.7,MUTED)
text(.902,.598,"Common [0, 1]\nquantile scale",7.4,AMBER,"bold",linespacing=1.15)

boundary(.817,.170,.168,.381,PURPLE)
label(.829,.526,"e","Predictions",PURPLE,8.5)
text(.903,.484,"Ordering by q",7.7,PURPLE,"bold")
rank=fig.add_axes([.845,.365,.119,.105],zorder=3)
if args.tinted:
    rank.set_facecolor("none")
rank.barh(np.arange(5),[.91,.75,.62,.39,.22],color=PURPLE,height=.65)
rank.invert_yaxis();rank.set_xlim(0,1)
rank.set_yticks(np.arange(5));rank.set_yticklabels(["1","2","3","4","5"],fontsize=6.5)
rank.set_xticks([]);rank.tick_params(length=0,pad=1);rank.spines[:].set_visible(False)
arrow(.800,.317,.834,.317,RED,1.5)
text(.904,.335,"Train-fold\nCox–Breslow",7.3,PURPLE,linespacing=1.2)
curve([.853,.220,.105,.081],"survival",PURPLE)
text(.907,.195,r"$\widehat S(t\mid q)$",10.3,PURPLE)

# Keep the distinction between real inputs and method illustrations explicit.
line(.018,.140,.985,.140,FAINT,.65)
text(.020,.106,"Real H&E / omics inputs; model geometry and curves are schematic.",7.3,MUTED,ha="left")
text(.020,.065,"Same patient i across modalities; training-fold maps are fixed for test patients.",7.3,MUTED,ha="left")

if require_matplotlib_panel_alignment is not None:
    require_matplotlib_panel_alignment(
        fig,axes=[ecdf_p,ecdf_o],panel_ids=["pathology_ecdf","omics_ecdf"],
        column_groups=[{"id":"ecdf","panels":["pathology_ecdf","omics_ecdf"]}],
        json_out=str(OUT)+".alignment.json",overlay_svg=str(OUT)+".alignment.svg",strict=True,
    )
else:
    print("Optional layout-audit tool unavailable; exporting without a new alignment report.")
fig.savefig(str(OUT)+".svg",facecolor="white")
fig.savefig(str(OUT)+".pdf",facecolor="white")
fig.savefig(str(OUT)+".png",dpi=400,facecolor="white")
fig.savefig(str(OUT)+".tiff",dpi=600,facecolor="white",
            pil_kwargs={"compression":"tiff_lzw"})
plt.close(fig)
OUT.with_suffix(".provenance.json").write_text(json.dumps({
    "case_id":provenance["case_id"], "overview_crop_box":crop,
    "overview_full_shape":full_shape, "registered_rectangles":registered,
    "original_wsi_rechecked":bool(args.source_wsi),
    "source_check":"Original WSI pixel audit" if args.source_wsi else "Bundled-image SHA256 verification against retained pixel-audit record",
    "selected_level0_coords":provenance["selected_coords"],
    "input_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in ASSETS.glob("*.png")},
    "physical_scale_bar":False,
    "schematic_elements":["feature geometry","PCA","ECDFs","raw score inset","survival curves","rank bars"],
    "inset_scope":"Same-patient scores, separate ECDFs, common range; no equality of ranks implied.",
    "adaptation_level":"Structural adaptation of retained Python schematic; unchanged input assets.",
    "color_variant":"soft module fills" if args.tinted else "white module interiors",
    "module_fills":FILLS if args.tinted else {},
    "claim":"Train-fold ECDFs place matched pathology and omics scores on a common scale for conservative late fusion.",
    "archetype":"schematic-led composite",
    "preview_scope":"Model-name labels only; no changes to inputs, geometry or palette." if args.canonical_names else "Palette only; no changes to data, coordinates, labels, layout or manuscript.",
},indent=2),encoding="utf-8")
print(OUT)
