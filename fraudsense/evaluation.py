import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
    confusion_matrix, brier_score_loss, log_loss)


def metrics(y, p, threshold=.5):
    y,p = np.asarray(y),np.asarray(p,dtype=float)
    tn,fp,fn,tp = map(int,confusion_matrix(y,p>=threshold,labels=[0,1]).ravel())
    precision = tp/max(1,tp+fp); recall = tp/max(1,tp+fn)
    return {'rows':len(y),'frauds':int(y.sum()),'precision':precision,'recall':recall,
            'f1':2*precision*recall/max(1e-15,precision+recall),
            'average_precision':float(average_precision_score(y,p)) if y.sum() else 0.,
            'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            'tn':tn,'fp':fp,'fn':fn,'tp':tp,'confusion_matrix':[[tn,fp],[fn,tp]],
            'false_alarms_per_1000':1000*fp/max(1,tn+fp),'review_rate':float(np.mean(p>=threshold)),
            'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),
            'threshold':float(threshold)}


def choose_thresholds(y,p):
    precision,recall,t = precision_recall_curve(y,p)
    f = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
    # Constrain combined reviews to 0.5% on selection; tied scores are respected.
    values,counts=np.unique(p,return_counts=True)
    rates=np.cumsum(counts[::-1])[::-1]/len(p)
    eligible=values[rates<=.005]
    floor=float(eligible[0]) if len(eligible) else 1.0000001
    feasible=t>=floor
    high=float(t[np.argmax(np.where(feasible,f,-1))]) if feasible.any() else floor
    return {'low':floor,'high':max(high,floor),'selection_review_rate':float(np.mean(p>=floor)),
            'objective':'Maximize selection F1 within 0.5% review budget; low is tie-aware budget threshold'}


def budgets(y,p):
    result=[]
    for budget in [.001,.005,.01,.02]:
        # Tie-aware thresholds, no arbitrary ordering of identical scores.
        v,c=np.unique(p,return_counts=True);rates=np.cumsum(c[::-1])[::-1]/len(p)
        t=float(v[rates<=budget][0]) if (rates<=budget).any() else 1.0000001
        result.append({'budget':budget,**metrics(y,p,t)})
    return result
