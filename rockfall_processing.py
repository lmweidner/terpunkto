# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:39:20 2022

@author: sa
"""

import numpy as np
import cloudComPy as cc
import time
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from sklearn.neighbors import KDTree
if cc.isPluginM3C2():
    import cloudComPy.M3C2
# import colorsys
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn. impute import SimpleImputer
# from imblearn.under_sampling import RandomUnderSampler
# from skimage.color import rgb2lab
cc.initCC()  # to do once before using plugins or dealing with numpy
import joblib

# %% M3C2 CALC
def calcChange(c1,c2,m3c2params,fp):
    cc.M3C2.initTrace_M3C2()
    
    print('calculating forwardChange')
    forwardChange = cc.M3C2.computeM3C2([c1,c2],m3c2params)
    print('calculating reverseChange')
    reverseChange = cc.M3C2.computeM3C2([c2,c1],m3c2params)
    print('done with change')
    

    
    cc.SavePointCloud(forwardChange,fp+r'\forwardChange.bin')
    cc.SavePointCloud(reverseChange,fp+r'\reverseChange.bin')
        
    return forwardChange,reverseChange
# %% FILTER CHANGE BELOW LOD

def prepforcluster(forwardChange,reverseChange,clusterSubsample,lod):
    frontmin = forwardChange.getScalarField(2).getMin() #get minchange in forward change
    backmax = reverseChange.getScalarField(2).getMax() #get maxchange in reverse change
    frontFaces = cc.filterBySFValue(frontmin,-lod,forwardChange) #filter forward change to only negative above lod
    backFaces = cc.filterBySFValue(lod,backmax,reverseChange) #filter reverse change to only positive above lod
    # print(type(np.ones((frontFaces.size(),1))))
    sf1 = frontFaces.getScalarField(0)
    sf1.fromNpArrayCopy(np.float32(np.ones((frontFaces.size(),1))))
    sf2 = backFaces.getScalarField(0)
    sf2.fromNpArrayCopy(np.float32(np.ones((backFaces.size(),1)))*2)
    
    mergedFalls = backFaces.cloneThis()
    mergedFalls.fuse(frontFaces)
    
    
    
    
    print('...downsampling')
    params = cc.SFModulationParams()
    refCloud = cc.CloudSamplingTools.resampleCloudSpatially(mergedFalls, clusterSubsample, params)
    print('...finished downsampling')
    (mergedFallsSub, res) = mergedFalls.partialClone(refCloud)
    
    rockfallsPreClustering = mergedFallsSub.toNpArrayCopy()
    changePreClustering = mergedFallsSub.getScalarField(2).toNpArrayCopy()
    faces = mergedFallsSub.getScalarField(0).toNpArrayCopy()
    
    return rockfallsPreClustering,changePreClustering,faces
  # %% CLUSTER USING OPTICS
def clustering(rocks,eps,minPts):
    start = time.perf_counter()
    # clustering = OPTICS(min_samples=minPts).fit(rockfallsPreClustering)
    clustering = DBSCAN(min_samples=minPts,eps=eps).fit(rocks)
    end = time.perf_counter()
    elapsedTime = end-start
    print(elapsedTime)
    labels = clustering.labels_
    print(len(np.unique(labels)))
    
    return labels
# %% WRITE OUTPUTS
def write_outputs(rocks,change,labels,faces,filters,fp):
    cloudList = []
    for label in np.unique(labels):
        # if label!=-1:
        changeTemp = change[labels==label]
        faceTemp = faces[labels==label]
        filtTemp = filters[labels==label]
        xyz = rocks[labels==label,:]
        

        
        # if filtTemp[0]==2:
            
        tempCloud = cc.ccPointCloud()
        tempCloud.coordsFromNPArray_copy(xyz)
        tempCloud.getScalarField(tempCloud.addScalarField('change')).fromNpArrayCopy(np.float32(changeTemp))
        tempCloud.getScalarField(tempCloud.addScalarField('face')).fromNpArrayCopy(np.float32(faceTemp))
        tempCloud.getScalarField(tempCloud.addScalarField('filter')).fromNpArrayCopy(np.float32(filtTemp))
        tempCloud.setName(str(label))
        tempCloud.setCurrentDisplayedScalarField(0)
        tempCloud.showSF(True)
        cloudList.append(tempCloud)
        
        
        out = np.column_stack((xyz,filtTemp,faceTemp,changeTemp))
        f = fp + r'\rf_' + str(label) + '.txt'
        np.savetxt(f,out,delimiter=',',fmt='%.7f')
        
        
    
    notNoise = labels!=-1
    f2 = fp + r'\rockfall_fullcloud.txt'
    out_onecloud = np.column_stack((rocks[notNoise],filters[notNoise],faces[notNoise],change[notNoise],labels[notNoise]))
    np.savetxt(f2,out_onecloud,delimiter=',',fmt='%.7f')
    cc.SaveEntities(cloudList,fp+r'\clusters.bin')
    
    print('wrote all outputs!')
   
# %% ROCKFALL FEATURE CALC AND CLASSIFICATION

def classify(rocks,change,labels,classifier,threshold):
    rockFilter = np.empty_like(change)
    rockFilterProbability = np.empty_like(change)
    for label in np.unique(labels):
        rockTemp = rocks[labels==label]

        changeTemp = change[labels==label]
        pca = PCA(n_components=3)
        eigs = pca.fit(rockTemp).explained_variance_ratio_
        L1,L2,L3 = eigs
        hull=ConvexHull(rockTemp)
        stats= np.array([np.nanmean(changeTemp),
                np.nanmin(changeTemp),
                np.nanmax(changeTemp),
                np.nanstd(changeTemp),
                np.shape(rockTemp)[0],
                hull.volume,
                hull.volume/np.shape(rockTemp)[0],
                L1,
                L2,
                L3,
                (L1*L2*L3)**(1/3),
                (L2-L3)/L1,
                (L1-L2)/L1,
                L3/L1])
        
        y_pred_prob = classifier.predict_proba(stats.reshape(1, -1))[:,1]
        
        y_thresh = (y_pred_prob>threshold).astype(int)
        rockFilterProbability[labels==label]=y_pred_prob
        rockFilter[labels==label]=y_thresh
     
    
    return rockFilter

# %%
def classifyVeg(rocks,change,labels,mask):
    
    maskCloud = cc.loadPointCloud(mask)
    xyz = maskCloud.toNpArrayCopy()
    maskLabel = maskCloud.getScalarField(0).toNpArrayCopy()
    
    tree = KDTree(xyz)
    
    vegLabel = np.empty_like(labels)
    for label in np.unique(labels):
        rockTempCentroid = np.mean(rocks[labels==label],axis=0).reshape(1, -1)
        
        dist,ind = tree.query(rockTempCentroid,k=10)
        ratio = np.sum(maskLabel[ind]!=2) / 10
        if ratio > 0.5:
            vegLabel[labels==label]= 1
        else:
            vegLabel[labels==label]= 2
        
    return vegLabel

# %% PARAMETERS
# Filepaths
cloudfp = r'C:\Users\sa\Desktop\Research\EHI\US74_Projects\E\rotated'
classifierfp= r'C:\Users\sa\Desktop\Research\rockfall_clouds\clusters\RFclassifier'
ccfp = r'C:\CloudComPy39\DATA'
m3c2params = ccfp+r'\m3c2_params_GW.txt'
# cloud1  = r'Z:\Projects\LidarScans\GW\Alignments\GW_20181013\GW_20181013__merged_subsampled.txt'
# cloud2 = r'Z:\Projects\LidarScans\GW\Alignments\GW_20191012\GW_20191012__merged_subsampled.txt'
cloud1 = r'C:\CloudComPy39\DATA\E_20210621_smallest.bin'
cloud2 = r'C:\CloudComPy39\DATA\E_20210520.bin'
maskfp = r'C:\CloudComPy39\DATA\GW_mask_v3_extended.txt'

# Parameters
lod = 0.02
minPts = 35
eps = 0.1
clusterSubsample = 0.01
thresh = 0.15

#  LOAD CLOUDS
c1 = cc.loadPointCloud(cloud1)
c2 = cc.loadPointCloud(cloud2)

rf = joblib.load(classifierfp)

print('...loaded cloud with size: ',c1.size())
print('...loaded cloud with size: ',c2.size())


# START PROCESSING
start = time.perf_counter()

# returns the full point clouds of the forward and reverse change

(change1,change2) = calcChange(c1,c2,m3c2params,ccfp)

# filters the forward and reverse clouds based on a symmetric LOD, merges the
# clusters, then subsample the merged clusters base on the distance threshold
# returns the merged rockfall point cloud, the change values for each point
# AND a label indicating which points correspond to the front and back faces
(rocks,changeVals,frontBackFace) = prepforcluster(change1,change2,clusterSubsample,lod)

# performs DBSCAN clustering.
# returns a label for each point corresponding to which cluster it belongs to
clusterLabels = clustering(rocks,eps,minPts)



# filter clusters based on some criteria. currently, there are two options:
# 1. filter by ML classification, 2. filter by mask

# MLfilter = classify(rocks,changeVals,clusterLabels,rf,thresh)
vegLabel = classifyVeg(rocks,changeVals,clusterLabels,maskfp)

# write output point clouds for manually inspecting/filtering rockfall clouds


write_outputs(rocks,changeVals,clusterLabels,frontBackFace,vegLabel,ccfp + r'\rockfalls')


end = time.perf_counter()
elapsedTime = end-start
print(elapsedTime)



