import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import os
import pickle
from scipy.spatial.distance import pdist 
import seaborn as sns

def get_curve(feature_map,ground_truth,num_min,num_max,period):
    prec_recall_curve = []
    num = len(feature_map)
    for threshold in np.arange(num_min,num_max,period):
        all_postives = 0
        for i in np.arange(1,num):
            for j in np.arange(0,i):
                if feature_map[i,j]>= threshold:
                    if  (i-100>=j):
                        all_postives= all_postives+1
        true_positives = (feature_map >= threshold)& (ground_truth == 1)

        try:
            precision = float(np.sum(true_positives))/all_postives
            recall = float(np.sum(true_positives))/np.sum(ground_truth == 1)
            prec_recall_curve.append([threshold,precision,recall])
        except:
            break
    prec_recall_curve = np.array(prec_recall_curve)
    #print(prec_recall_curve)
    return prec_recall_curve


output = open('KITTI/netvlad/feats_score_00.pkl', 'rb')
KITTI00_feats_score_00= pickle.load(output)
output.close()

print("loading KITTI00_GroundTruth")
KITTI00_GroundTruth = np.loadtxt("KITTI/netvlad/kitti00GroundTruth.txt",delimiter = ",")
print("loaded KITTI00_GroundTruth")

sns.heatmap(KITTI00_GroundTruth, cmap="Reds", cbar_kws={'label': 'score'})
fig=plt.gcf()
plt.title("datasets give the truth loop",fontsize=24)
fig.set_size_inches(12,8)
# plt.savefig('Nordland/winter_summer_cosine_distance2_'+LAYER_NAME)
plt.show()

print("doing prec_recall_curve0")
prec_recall_curve0 = get_curve(KITTI00_feats_score_00,KITTI00_GroundTruth,0.0,0.8,0.05)
print("done prec_recall_curve0")

plt.plot(prec_recall_curve0[:,2],prec_recall_curve0[:,1],'r',label="netvlad_00")
#题目
plt.title('Precision-Recall curve')
#坐标轴名字
plt.xlabel('Recall')
plt.ylabel('Precision')
#背景虚线
plt.grid(True)
plt.legend(loc='upper right')
#显示
plt.show()
print("done")

