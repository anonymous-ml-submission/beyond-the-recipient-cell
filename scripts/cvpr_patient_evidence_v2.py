#!/usr/bin/env python3
"""Honest patient-level evidence, paired matched-decoy comparison, figures.

Primary independent patient units: P2, P3, P4. Equal weighting:
cell pairs -> gene -> section -> patient; each of five frozen markers equal weight.
P1 not silently pooled. Exact sign tests use n patients, NOT cells/genes.
Outputs read-only relative to frozen inputs, in a NEW output directory.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATS={'P2':['TENX195','TENX196'],'P3':['TENX197','TENX198','TENX199'],
      'P4':['TENX200','TENX201','TENX202']}
C=('MMP2','SFRP4','CCDC80','CXCL12','FBLN1')
I=('TRAC','CD3E','IL7R','CD8A','CCL5')
KEY=['gene','recipient_cell_id','source_cell_id','intervention']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def panel(g):
    if g in C:return 'Connective_to_Inflammatory'
    if g in I:return 'Inflammatory_to_Connective'
    raise ValueError('UNEXPECTED_GENE '+str(g))

def get_original(f,sid,pat):
    assert f.is_file(),('MISSING_ORIGINAL',f)
    z=pd.read_csv(f,dtype={c:str for c in KEY+['control_cell_id']})
    need=set(KEY+['control_cell_id','pred_original','delta_source','delta_control','delta_source_minus_control'])
    assert need.issubset(z.columns),('ORIGINAL_SCHEMA_MISMATCH',sorted(need-set(z.columns)))
    assert len(z)>0 and set(z.intervention)=={'blur','inpaint'}
    assert not z.duplicated(KEY+['control_cell_id']).any()
    assert set(z.gene)==set(C+I),('UNEXPECTED_MARKERS',sid,set(z.gene))
    assert np.isfinite(z[['delta_source','delta_control','delta_source_minus_control']].to_numpy(float)).all()
    assert np.allclose(z.delta_source-z.delta_control,z.delta_source_minus_control,atol=1e-5,rtol=1e-5),('DELTA_ARITHMETIC',sid)
    z['sample']=sid;z['patient']=pat;z['panel']=z.gene.map(panel)
    return z

def get_new(f,sid,pat,rank,orig,match):
    assert f.is_file(),('MISSING_DECOY_EVALUATION',f)
    status=f.with_suffix('.json')
    assert status.is_file(),('MISSING_NEW_INTERVENTION_PROVENANCE',f)
    meta=json.loads(status.read_text())
    assert meta['sample']==sid and meta['patient']==pat and meta['rank']==rank
    assert meta['new_predictions_sha256']==sha(f),('DECOY_IMAGE_SHA_CHANGED',sid,rank)
    cp=match/sid/f'decoy_rank{rank}_pairs.csv'
    assert meta['new_pairs_sha256']==sha(cp),('DECOY_PAIRS_SHA_CHANGED',sid,rank)
    assert meta['completed_original_sha256']==sha(orig),('ORIGINAL_HASH_CHANGED',sid)
    z=pd.read_csv(f,dtype={c:str for c in KEY+['old_control_cell_id','control_cell_id']})
    required=set(KEY+['old_control_cell_id','control_cell_id','delta_source',
                      'delta_control','delta_source_minus_control'])
    assert required.issubset(z.columns)
    assert not z.duplicated(KEY+['old_control_cell_id']).any()
    assert set(z.gene).issubset(set(C+I)) and set(z.intervention)=={'blur','inpaint'}
    assert np.isfinite(z[['delta_source','delta_control','delta_source_minus_control']].to_numpy(float)).all()
    assert np.allclose(z.delta_source-z.delta_control,z.delta_source_minus_control,atol=1e-5)
    z['sample']=sid;z['patient']=pat;z['panel']=z.gene.map(panel)
    return z

def nested_means(x,value,group_col='comparison'):
    """Equal gene within section; equal section within patient. No pseudo-n."""
    a=(x.groupby([group_col,'patient','sample','panel','intervention','gene'],as_index=False)
       .agg(gene_mean=(value,'mean'),n_cells=('recipient_cell_id','size')))
    # All five predeclared genes must remain for a panel's section summary.
    b=(a.groupby([group_col,'patient','sample','panel','intervention'],as_index=False)
       .agg(section_mean=('gene_mean','mean'),n_genes=('gene','nunique'),n_pair_rows=('n_cells','sum')))
    assert b.n_genes.eq(5).all(),'PANEL_LOST_A_PRESPECIFIED_GENE; report support, do not silently omit'
    c=(b.groupby([group_col,'patient','panel','intervention'],as_index=False)
       .agg(effect=('section_mean','mean'),n_sections=('sample','nunique')))
    return a,b,c

def test_patients(c):
    rows=[]
    for (comp,pl,mode),g in c.groupby(['comparison','panel','intervention']):
        vals=g.effect.to_numpy(float)
        assert len(vals)==3 and set(g.patient)==set(PATS)
        n=len(vals);mean=float(vals.mean());sd=float(vals.std(ddof=1))
        ci=float(student_t.ppf(0.975,n-1)*sd/math.sqrt(n))
        n_pos=int((vals>0).sum());n_neg=int((vals<0).sum());nnon=n_pos+n_neg
        psign=sum(math.comb(nnon,k) for k in range(n_pos,nnon+1))/2**nnon if nnon else 1.0
        ptwo=min(1.,2*min(psign,1.-psign+math.comb(nnon,n_pos)/2**nnon)) if nnon else 1.0
        loo=[float(np.delete(vals,i).mean()) for i in range(n)]
        rows.append({'comparison':comp,'panel':pl,'intervention':mode,'patients':n,
          'mean_equal_patient':mean,'sd_between_patients':sd,
          't95_lower_exploratory':mean-ci,'t95_upper_exploratory':mean+ci,
          'positive_patients':n_pos,'negative_patients':n_neg,
          'exact_one_sided_sign_p_exploratory':psign,
          'exact_two_sided_sign_p_exploratory':ptwo,
          'leave_one_patient_out_mean_min':min(loo),'leave_one_patient_out_mean_max':max(loo),
          'inference_warning':'n=3; intervals exploratory; do not treat gene/cell pairs as independent donors'})
    return pd.DataFrame(rows)

def plot_patients(c,stats,out):
    comparisons=c.comparison.unique().tolist()
    fig,axes=plt.subplots(2,len(comparisons),figsize=(5.3*len(comparisons),7),squeeze=False)
    styles={'P2':('o',-.13),'P3':('s',0),'P4':('^',.13)}
    for j,comp in enumerate(comparisons):
        for i,mode in enumerate(('blur','inpaint')):
            ax=axes[i,j]
            x=c.loc[(c.comparison==comp)&(c.intervention==mode)]
            for k,pl in enumerate(('Connective_to_Inflammatory','Inflammatory_to_Connective')):
                sub=x.loc[x.panel==pl]
                for z in sub.itertuples(index=False):
                    marker,offset=styles[z.patient]
                    ax.scatter(k+offset,z.effect,s=55,marker=marker,label=z.patient if k==0 else None)
                q=stats.loc[(stats.comparison==comp)&(stats.panel==pl)&(stats.intervention==mode)]
                if len(q)==1: ax.hlines(q.mean_equal_patient.iloc[0],k-.26,k+.26,linewidth=2.5)
            ax.axhline(0,color='gray',linewidth=0.8)
            ax.set_xticks([0,1],['C→I','I→C'])
            ax.set_title(comp.replace('_',' ')+f' | {mode}')
            ax.set_ylabel('Source − control prediction change')
            if i==0 and j==0:ax.legend(frameon=False)
            ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(out/'patient_effects.png',dpi=240)
    fig.savefig(out/'patient_effects.pdf');plt.close(fig)

def plot_coverage(support,out):
    if support.empty:return
    q=support.pivot(index='sample',columns='rank',values='coverage')
    ax=q.plot.bar(figsize=(10,4),ylim=(0,1),rot=45)
    ax.set_ylabel('Original evaluated frozen pairs with support')
    ax.set_title('Outcome-blind, same-morphology decoy common support')
    ax.legend(title='Decoy rank')
    ax.grid(axis='y',alpha=.15)
    plt.tight_layout();plt.savefig(out/'matching_coverage.png',dpi=240)
    plt.savefig(out/'matching_coverage.pdf');plt.close()

def plot_gene_heatmap(allrows,out):
    x=allrows.loc[allrows.comparison=='ORIGINAL_FULL']
    a=(x.groupby(['patient','panel','gene','intervention','sample'],as_index=False)
       .agg(v=('delta_source_minus_control','mean')))
    b=(a.groupby(['patient','gene','intervention'],as_index=False).agg(v=('v','mean')))
    for mode in ('blur','inpaint'):
        q=b.loc[b.intervention==mode].pivot(index='gene',columns='patient',values='v')
        q=q.reindex(list(C+I),columns=list(PATS))
        assert np.isfinite(q.to_numpy()).all()
        fig,ax=plt.subplots(figsize=(5.8,5.2))
        vmax=max(1e-5,float(np.max(np.abs(q.to_numpy()))))
        im=ax.imshow(q.to_numpy(),cmap='coolwarm',vmin=-vmax,vmax=vmax,aspect='auto')
        ax.set_xticks(range(3),list(PATS));ax.set_yticks(range(10),list(C+I))
        ax.set_title('Original source-control contrast: '+mode)
        fig.colorbar(im,ax=ax,label='Prediction change')
        fig.tight_layout();fig.savefig(out/f'original_gene_heatmap_{mode}.png',dpi=240)
        fig.savefig(out/f'original_gene_heatmap_{mode}.pdf');plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--original-root',type=Path,required=True)
    p.add_argument('--decoy-result-root',type=Path)
    p.add_argument('--match-root',type=Path)
    p.add_argument('--ranks',nargs='+',type=int,default=[1])
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    out=a.out.resolve();assert not out.exists(),f'REFUSE_OVERWRITE {out}'
    assert a.ranks and set(a.ranks)<=set((1,2,3))
    assert (a.decoy_result_root is None)==(a.match_root is None),'MATCH_ROOT_AND_IMAGE_RESULTS_MUST_BOTH_BE_SUPPLIED'
    has_decoy=a.decoy_result_root is not None
    origs=[];comp=[];support=[];hashes={}
    for pat,sids in PATS.items():
        for sid in sids:
            f=a.original_root/pat/f'{sid}_ctx128_intervention.csv'
            old=get_original(f,sid,pat)
            origs.append(old)
            hashes[str(f)]=sha(f)
            if not has_decoy:continue
            for rank in a.ranks:
                covfile=a.match_root/sid/'coverage.json'
                cov=json.loads(covfile.read_text())
                assert cov['sample']==sid and cov['patient']==pat
                assert cov['original_completed_intervention_csv_sha256']==sha(f)
                rankstr=str(rank)
                coverage=float(cov['coverage_per_rank'][rankstr])
                support.append({'patient':pat,'sample':sid,'rank':rank,'coverage':coverage,
                                'n_original_pairs':cov['original_unique_frozen_pairs'],
                                'n_matched_pairs':cov['accepted_per_rank'][rankstr]})
                fp=a.decoy_result_root/f'rank{rank}'/f'{sid}_same_morph_intervention.csv'
                new=get_new(fp,sid,pat,rank,f,a.match_root)
                orig_sel=old.merge(new[KEY+['old_control_cell_id','delta_source','delta_source_minus_control']],
                    left_on=KEY+['control_cell_id'],right_on=KEY+['old_control_cell_id'],
                    how='inner',validate='one_to_one',suffixes=('_old','_new'))
                assert len(orig_sel)==len(new),('FROZEN_PAIR_JOIN_FAILED',sid,rank,len(orig_sel),len(new))
                assert np.max(np.abs(orig_sel.delta_source_old-orig_sel.delta_source_new))<=1e-4,('REPLAY_SOURCE_MISMATCH',sid,rank)
                # Both controls on exactly the same original (source,recipient,gene)
                # rows, with independent outcome-blind matching.
                left=old.merge(new[KEY+['old_control_cell_id']],
                    left_on=KEY+['control_cell_id'],right_on=KEY+['old_control_cell_id'],
                    how='inner',validate='one_to_one')
                assert len(left)==len(new)
                for label,d in [('ORIGINAL_SUBSET_rank'+str(rank),left),
                                ('SAME_CLASS_rank'+str(rank),new)]:
                    t=d.copy();t['comparison']=label
                    comp.append(t)
                # Exact paired difference between controls, on IDENTICAL original pairs.
                difference=left.copy()
                diff=(new[KEY+['old_control_cell_id','delta_source_minus_control']].rename(
                      columns={'delta_source_minus_control':'same_effect'}))
                difference=difference.merge(diff,left_on=KEY+['control_cell_id'],
                        right_on=KEY+['old_control_cell_id'],how='inner',validate='one_to_one',
                        suffixes=('','_check'))
                assert len(difference)==len(new)
                difference['delta_source_minus_control']=(difference.same_effect-
                                         difference.delta_source_minus_control)
                difference['comparison']='SAME_MINUS_OLD_rank'+str(rank)
                comp.append(difference)
    original=pd.concat(origs,ignore_index=True)
    original['comparison']='ORIGINAL_FULL'
    a_all=[original]+comp
    allrows=pd.concat(a_all,ignore_index=True)
    # Some same-class rank>1 sensitivity subsets will lack one of the five genes
    # in a section. This is an explicit non-estimable failure, not an invitation
    # to fill or impute; the script must stop.
    gene,section,patient=nested_means(allrows,'delta_source_minus_control')
    stats=test_patients(patient)
    support=pd.DataFrame(support)
    out.mkdir(parents=True,exist_ok=False)
    gene.to_csv(out/'gene_section_effects.csv',index=False)
    section.to_csv(out/'section_panel_effects.csv',index=False)
    patient.to_csv(out/'patient_panel_effects.csv',index=False)
    stats.to_csv(out/'patient_uncertainty.csv',index=False)
    if not support.empty:support.to_csv(out/'matched_decoy_coverage.csv',index=False)
    plot_patients(patient,stats,out)
    plot_coverage(support,out)
    plot_gene_heatmap(original,out)
    (out/'provenance.json').write_text(json.dumps({
      'patients':['P2','P3','P4'],'n_independent_new_patients':3,
      'p1_not_pooled':True,'frozen_markers':{'connective':C,'inflammatory':I},
      'estimand':'mean within cell pairs -> gene -> section -> patient, equal weights at each level',
      'source_files_sha256':hashes,'controls':'source vs historical different-morphology; plus matched same-morphology if supplied',
      'notes':['n=3, paired cell/gene section measures are not n=3 biological replicates',
               'patient-level t intervals and sign test exploratory; selection/matching are descriptive',
               'conditioned original_subset uses identical cells as same_class to prevent selection artifact',
               'P1 earlier protocol is reference only and NOT pooled as confirmatory new patient']},indent=2)+'\n')
    print('PATIENT_EFFECTS')
    print(patient.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('PATIENT_UNCERTAINTY_N3')
    print(stats.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    if not support.empty:print('MATCHING_SUPPORT',support.to_string(index=False))
    print('CVPR_PATIENT_EVIDENCE_COMPLETE',out)

if __name__=='__main__':main()
