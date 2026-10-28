"""Validate retained MIL predictions before adding them to the manuscript table."""

from pathlib import Path
import hashlib
import json
import re
import sys
import tarfile
import warnings
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd
import yaml
from scipy.integrate import IntegrationWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dataset.label_converter import MetaSurvData
from eval.SurvivalEVAL.Evaluations.Concordance import concordance
from eval.SurvivalEVAL.Evaluations.util import predict_mean_survival_time

MIL = ROOT / "downloads/matched_mil_backbones_complete_20260903/result/matched-mil-backbones"
OUT = ROOT / "result/qaclf-six-method-comparison"
COHORTS = ["blca","brca","gbmlgg","luad","ucec"]
QACLF = {
    "blca":"blca-frozen-pathomic-fusion-cloud-baseline-outer5-20260820",
    "brca":"brca-frozen-pathomic-fusion-outer5-20260820",
    "gbmlgg":"gbmlgg-frozen-pathomic-fusion-outer5-20260820",
    "luad":"luad-frozen-pathomic-fusion-cloud-baseline-outer5-20260820",
    "ucec":"ucec-frozen-pathomic-fusion-cloud-baseline-outer5-20260820",
}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    audit = {"models":["ABMIL","TransMIL"],"folds":[],
             "evaluator":"Vendored SurvivalEVAL Mean/Linear, continuous outcomes, Comparable/All",
             "all_patient_sets_equal":True,"all_events_equal":True,
             "max_metric_abs_difference":0.0,
             "source_sha256":{},"external_protocol_findings":[]}
    fold_rows, patient_rows = [], []
    reference_all=pd.read_csv(ROOT / "result/external_multimodal_comparison_20260821/unified/paired_patient_predictions.csv",
                              dtype={"patient_id":str})
    for cohort in COHORTS:
        reference = reference_all[reference_all.dataset==cohort.upper()]
        for fold in range(5):
            target = reference[reference.fold==fold].set_index("patient_id")
            split_path = ROOT / f"data_split/5foldcv/tcga_{cohort}/splits_{fold}.csv"
            split = pd.read_csv(split_path,dtype=str)
            split_cols=[c for c in split if not c.startswith("Unnamed")]
            train_ids=split[split_cols[0]].dropna().tolist()
            test_ids=split[split_cols[1]].dropna().tolist()
            assert set(test_ids)==set(target.index), (cohort,fold,"split mismatch")
            assert not set(train_ids)&set(test_ids)
            for model in audit["models"]:
                run = MIL / f"seed_42/{model.lower()}/{cohort}-{model}-CONCH-fold_{fold}"
                config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
                assert config["deepmil_network"]==model
                assert config["seed"]==42 and config["ckpt_for_eval"]=="last"
                assert config["deepmil_use_feat_proj"] is False
                assert "feats-CONCH-vl-proj/pt_files" in config["path_patch"]
                pred_path=run / "bag_clf_train_last_pred_test.csv"
                pred = pd.read_csv(pred_path,dtype={"patient_id":str}).set_index("patient_id")
                assert pred.index.is_unique and set(pred.index)==set(target.index)
                pred=pred.loc[target.index]
                assert np.array_equal(pred.e.to_numpy(),target.event.to_numpy())
                table_path=ROOT / config["path_table"]
                meta=MetaSurvData(str(table_path),data_split={"train":train_ids,"test":test_ids},verbose=False)
                with redirect_stdout(StringIO()):
                    meta.generate_discrete_label(num_bins=config["time_bins"],use_quantiles=False,summary=False)
                labels=meta.pat_data.set_index("patient_id").loc[target.index]
                assert np.allclose(labels.t,target.event_time,atol=1e-4,rtol=0)
                assert np.array_equal(labels.e.to_numpy(),target.event.to_numpy())
                columns=[c for c in pred if c.startswith("surf_")]
                curves=pred[columns].to_numpy()
                assert len(columns)==len(meta.time_coordinates)
                assert np.isfinite(curves).all()
                assert np.max(np.diff(curves,axis=1))<=1e-6
                with warnings.catch_warnings(record=True) as integration_warnings:
                    warnings.simplefilter("always",IntegrationWarning)
                    times=np.asarray([predict_mean_survival_time(c,meta.time_coordinates,interpolation="Linear")
                                      for c in curves])
                train=meta.pat_data.set_index("patient_id").loc[train_ids]
                metric=float(concordance(times,target.event_time.to_numpy(),target.event.to_numpy(),
                                         train.t.to_numpy(),train.e.to_numpy(),
                                         pair_method="Comparable",ties="All")[0])
                logged=float(re.search(r"test/pred_c_index\s+-->\s+([0-9.eE+-]+)",
                                      (run/"train_metrics-last.txt").read_text()).group(1))
                difference=abs(metric-logged)
                assert difference<1e-6,(cohort,fold,model,metric,logged)
                audit["max_metric_abs_difference"]=max(audit["max_metric_abs_difference"],difference)
                record={"model":model,"dataset":cohort.upper(),"fold":fold,
                        "n_test":len(pred),"events":int(target.event.sum()),
                        "c_index":metric,"logged_c_index":logged,
                        "metric_abs_difference":difference,"seed":42,"epochs":config["epochs"],
                        "integration_warnings":sum(issubclass(w.category,IntegrationWarning) for w in integration_warnings)}
                fold_rows.append(record);audit["folds"].append(record)
                for patient,time in zip(target.index,times):
                    patient_rows.append({"model":model,"dataset":cohort.upper(),"fold":fold,
                                         "patient_id":patient,"event_time":target.loc[patient,"event_time"],
                                         "event":target.loc[patient,"event"],"predicted_time":time})
                audit["source_sha256"][str(pred_path.relative_to(ROOT))]=hashlib.sha256(pred_path.read_bytes()).hexdigest()
                print(f"PASS {model} {cohort} fold {fold}: n={len(pred)} C={metric:.6f}",flush=True)

    fold_frame=pd.DataFrame(fold_rows)
    summary=(fold_frame.groupby(["model","dataset"],sort=False).c_index
             .agg(mean_c_index="mean",std_c_index=lambda s:np.std(s.to_numpy(),ddof=0),
                  completed_folds="size").reset_index())
    audit["fold_standard_deviation_ddof"]=0
    old=pd.read_csv(MIL / "summary/dataset_summary.csv").set_index(["model","dataset"])
    for row in summary.itertuples():
        prior=old.loc[(row.model,row.dataset)]
        assert abs(row.mean_c_index-prior.mean_c_index)<1e-6
        assert abs(row.std_c_index-prior.std_c_index)<1e-6
        assert row.completed_folds==5
    summary.to_csv(OUT / "mil_dataset_summary.csv",index=False)
    fold_frame.to_csv(OUT / "mil_fold_metrics.csv",index=False)
    pd.DataFrame(patient_rows).to_csv(OUT / "mil_patient_predictions.csv",index=False)

    existing=pd.read_csv(ROOT / "result/external_multimodal_comparison_20260821/unified/dataset_summary_unified.csv")
    table_rows=[]
    methods=[("ABMIL-CONCH","pathology","ABMIL"),("TransMIL-CONCH","pathology","TransMIL"),
             ("Matched VLSA","pathology","matched_vlsa"),("MCAT-CONCH","multimodal","mcat_conch"),
             ("CAT-TPL-CONCH","multimodal","cat_tpl_conch"),("QACLF","multimodal","qaclf")]
    for name,modality,key in methods:
        row={"method":name,"modality":modality}
        means=[]
        for cohort in COHORTS:
            c=cohort.upper()
            if key in audit["models"]:
                s=summary[(summary.model==key)&(summary.dataset==c)].iloc[0]
                mean,std=s.mean_c_index,s.std_c_index
            else:
                s=existing[existing.dataset==c].iloc[0]
                mean,std=s[key],s[key+"_std"]
            row[c]=float(mean);row[c+"_std"]=float(std);means.append(mean)
        row["MACRO"]=float(np.mean(means));table_rows.append(row)
    pd.DataFrame(table_rows).to_csv(OUT / "six_method_dataset_summary.csv",index=False)

    archive=ROOT / "cat_conch_multimodal_baselines_20260820.tar.gz"
    with tarfile.open(archive,"r:gz") as tar:
        member="external/CAT-with-TPL-master/datasets/dataset_survival.py"
        source=tar.extractfile(member).read().decode("utf-8")
    audit["external_protocol_findings"].append({
        "severity":"requires_train_fold_discretization_recheck",
        "confirmed_from":"retained cloud source archive",
        "source_member":member,"sha256":hashlib.sha256(source.encode()).hexdigest(),
        "global_qcut_before_return_splits":source.index("pd.qcut") < source.index("def return_splits"),
        "effect":"Existing MCAT/CAT numerical outputs are retained adaptations, not fully train-only-discretized controls.",
        "next_action":"Re-run time discretization using training events only. Do not select settings on outer-test performance.",
    })
    assert audit["external_protocol_findings"][0]["global_qcut_before_return_splits"]
    raw=ROOT / "result/external_multimodal_comparison_20260821/raw"
    audit["external_epoch_logs"]={"tensorboard_files":len(list(raw.rglob("events.out.tfevents*"))),
                                   "text_logs":len(list(raw.rglob("*.log"))),
                                   "convergence":"not assessable from retained final summaries alone"}
    audit["complete_mil_folds"]=len(fold_rows)
    (OUT / "audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf-8")
    print("Verified 50 MIL folds; six-method table exported. External discretization issue recorded.")


if __name__=="__main__":
    main()
