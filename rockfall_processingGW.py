# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:39:20 2022
@author: sa
"""

import numpy as np
import cloudComPy as cc
import time
import os
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
# FILTER CHANGE BELOW LOD

def prepforcluster(forwardChange,reverseChange,clusterSubsample,lod):
    
    #for Heather's clouds, change is always the last scalar field 
    changeSF1 = forwardChange.getNumberOfScalarFields()-1
    changeSF2 = reverseChange.getNumberOfScalarFields()-1
    
    forwardChange.renameScalarField(changeSF1,'M3C2')
    reverseChange.renameScalarField(changeSF2,'M3C2')
    
    if (changeSF1 != changeSF2):
        print('WARNING!!!!: forward and reverse clouds had different num scalar fields. Check outputs.')
    
    assert forwardChange.getNumberOfScalarFields() > 1
    assert reverseChange.getNumberOfScalarFields() > 1
    
    
    frontmin = forwardChange.getScalarField(changeSF1).getMin() #get minchange in forward change
    backmax = reverseChange.getScalarField(changeSF2).getMax() #get maxchange in reverse change

    forwardChange.setCurrentOutScalarField(changeSF1)
    reverseChange.setCurrentOutScalarField(changeSF2)
    frontFaces = cc.filterBySFValue(frontmin,-lod,forwardChange) #filter forward change to only negative above lod
    backFaces = cc.filterBySFValue(lod,backmax,reverseChange) #filter reverse change to only positive above lod

    sf1 = frontFaces.getScalarField(0)
    sf1.fromNpArrayCopy(np.float32(np.ones((frontFaces.size(),1))))
    sf2 = backFaces.getScalarField(0)
    sf2.fromNpArrayCopy(np.float32(np.ones((backFaces.size(),1)))*2)
    
    mergedFalls = backFaces.cloneThis()
    mergedFalls.fuse(frontFaces)
    
    
    
    
    # # NOTE: THIS DOWNSAMPLING STEP IS PROBABLY NOT REQUIRED, IT WOULD ONLY BE NEEDED FOR SPEED BUT IT'S ALREADY PLENTY FAST
    # print('...downsampling')
    # params = cc.SFModulationParams()
    # refCloud = cc.CloudSamplingTools.resampleCloudSpatially(mergedFalls, clusterSubsample, params)
    # print('...finished downsampling')
    # (mergedFallsSub, res) = mergedFalls.partialClone(refCloud)
    
    rockfallsPreClustering = mergedFalls.toNpArrayCopy()
    changePreClustering = mergedFalls.getScalarField('M3C2').toNpArrayCopy()
    faces = mergedFalls.getScalarField(0).toNpArrayCopy()
    
    return rockfallsPreClustering,changePreClustering,faces
  #  CLUSTER USING OPTICS
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
#  WRITE OUTPUTS
def write_outputs(rocks,change,labels,faces,filters,fp):
    cloudList = []
    for label in np.unique(labels):
        # if label!=-1:
        changeTemp = change[labels==label]
        faceTemp = faces[labels==label]
        if len(filters)>0:
            filtTemp = filters[labels==label]
        xyz = rocks[labels==label,:]
        
            
        tempCloud = cc.ccPointCloud()
        tempCloud.coordsFromNPArray_copy(xyz)
        tempCloud.getScalarField(tempCloud.addScalarField('change')).fromNpArrayCopy(np.float32(changeTemp))
        tempCloud.getScalarField(tempCloud.addScalarField('face')).fromNpArrayCopy(np.float32(faceTemp))
        if len(filters)>0:
            tempCloud.getScalarField(tempCloud.addScalarField('filter')).fromNpArrayCopy(np.float32(filtTemp))
        tempCloud.setName(str(label))
        tempCloud.setCurrentDisplayedScalarField(0)
        tempCloud.showSF(True)
        if len(filters)>0:
            if filtTemp[0]==2:
                cloudList.append(tempCloud)
        else:
            cloudList.append(tempCloud)
        
        # if len(filters)>0:
            # if filtTemp[0]==2:
                # out = np.column_stack((xyz,filtTemp,faceTemp,changeTemp))
                # f = fp + r'\rf_' + str(label) + '.txt'
                # np.savetxt(f,out,delimiter=',',fmt='%.7f')
        # else:
        out = np.column_stack((xyz,faceTemp,changeTemp))
        f = fp + r'\rf_' + str(label) + '.txt'
        np.savetxt(f,out,delimiter=',',fmt='%.7f')
    
    
    
    notNoise = labels!=-1
    f2 = fp + r'\rockfall_fullcloud.txt'
    if len(filters)>0:
        out_onecloud = np.column_stack((rocks[notNoise],filters[notNoise],faces[notNoise],change[notNoise],labels[notNoise]))
    else:
        out_onecloud = np.column_stack((rocks[notNoise],faces[notNoise],change[notNoise],labels[notNoise]))
    np.savetxt(f2,out_onecloud,delimiter=',',fmt='%.7f')
    cc.SaveEntities(cloudList,fp+r'\clusters.bin')
    
    print('wrote all outputs!')
   
#  ROCKFALL FEATURE CALC AND CLASSIFICATION

def classify(rocks,change,labels,method,**kw):
    if method == 'ml':
        print('warning: using ML method')
        FinalFilt = np.empty_like(change)
        allStats = np.empty((0,14))
        
        for label in np.unique(labels):
            rockTemp = rocks[labels==label]
            faceTemp = kw['faces'][labels==label]
            u,n = np.unique(faceTemp,return_counts=True)
            changeTemp = change[labels==label]
            if np.sum(np.isnan(changeTemp)) == len(changeTemp):
                FinalFilt[labels==label] = 1
                continue
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
                        
            allStats = np.append(allStats,[stats],axis=0)
            
            
            
            y_pred_prob = kw['classifier'].predict_proba(stats.reshape(1, -1))[:,1]
            
            y_thresh = (y_pred_prob>kw['threshold']).astype(int)             
                        
            FinalFilt[labels==label]=y_thresh+1 #add one to make 1==false and 2==true, consistent with other filter methods
            
            if stats[5]>0.01:
                
                FinalFilt[labels==label]=2
            
            elif len(n)==1:
                
                FinalFilt[labels==label] = 1
                
            elif np.abs((n[0]-n[1])/len(faceTemp)) > 0.6:
                
                FinalFilt[labels==label] = 1
            
            else:
                FinalFilt[labels==label] = 2
            
        return FinalFilt,allStats
    
    elif method == 'mask':
        maskCloud = cc.loadPointCloud(kw['mask'])
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
cloud1 = r'D:\fromSeagateExpansionDrive\I70\GW_MM117\Comparisons\20180108-20180220\forward\GW_20180108_classified-GW_20180220_classified_change.txt'
cloud2 = r'D:\fromSeagateExpansionDrive\I70\GW_MM117\Comparisons\20180108-20180220\reverse\GW_20180220_classified-GW_20180108_classified_change.txt'
outfp = r'D:\fromSeagateExpansionDrive\I70\GW_MM117\Cluster_Luke_2022'
dirs = os.listdir(r'D:\fromSeagateExpansionDrive\I70\GW_MM117\Cluster_Luke_2022')[:-4]
# m3c2params = ccfp+r'\m3c2_params_GW.txt'
# cloud1  = r'Z:\Projects\LidarScans\GW\Alignments\GW_20181013\GW_20181013__merged_subsampled.txt'
# cloud2 = r'Z:\Projects\LidarScans\GW\Alignments\GW_20191012\GW_20191012__merged_subsampled.txt'
maskfp = r'D:\fromSeagateExpansionDrive\I70\GW_MM117\Cluster_Luke_2022\GW_mask_v3_extended.txt'
classifierfp= r'E:\fromSeagateExpansionDrive\I70\GW_MM117\Cluster_Luke_2022\RFclassifier'


# Parameters
lod = 0.02
minPts = 40
eps = 0.1
thresh = 0.15
# clusterSubsample = 0.01


#  LOAD CLOUDS
for folder in dirs:
# folder = dirs
    print(folder)
    t1,t2 = folder.split('-')
    
    cloud1 = os.path.join('D:/fromSeagateExpansionDrive/I70/GW_MM117/Comparisons',folder,'forward','GW_'+t1+'_classified-GW_'+t2+'_classified_change.txt')
    cloud2 = os.path.join('D:/fromSeagateExpansionDrive/I70/GW_MM117/Comparisons',folder,'reverse','GW_'+t2+'_classified-GW_'+t1+'_classified_change.txt')
    outfp = os.path.join('D:/fromSeagateExpansionDrive/I70/GW_MM117/Cluster_Luke_2022',folder)
    
    
    c1 = cc.loadPointCloud(cloud1)
    c2 = cc.loadPointCloud(cloud2)
    
    rf = joblib.load(classifierfp)
    
    print('...loaded cloud with size: ',c1.size())
    print('...loaded cloud with size: ',c2.size())
    
    
    # START PROCESSING
    start = time.perf_counter()
    
    # returns the full point clouds of the forward and reverse change
    
    # (change1,change2) = calcChange(c1,c2,m3c2params,ccfp)
    change1 = c1
    change2 = c2
    
    # filters the forward and reverse clouds based on a symmetric LOD, merges the
    # clusters, then subsample the merged clusters base on the distance threshold
    # returns the merged rockfall point cloud, the change values for each point
    # AND a label indicating which points correspond to the front and back faces
    (rocks,changeVals,frontBackFace) = prepforcluster(change1,change2,[],lod)
    
    # performs DBSCAN clustering.
    # returns a label for each point corresponding to which cluster it belongs to
    clusterLabels = clustering(rocks,eps,minPts)
    
    
    
    # filter clusters based on some criteria. currently, there are two options:
    # 1. filter by ML classification, 2. filter by mask
    

    # MLfilter = classify(rocks,changeVals,clusterLabels,rf,thresh)
    autofilter,stat = classify(rocks,changeVals,clusterLabels,'ml',classifier=rf,threshold=thresh,faces=frontBackFace)
    print('Original: ',len(np.unique(clusterLabels)))
    print('Filtered: ',len(np.unique(clusterLabels[autofilter==2])))
    
    # write output point clouds for manually inspecting/filtering rockfall clouds
    
    
    write_outputs(rocks,changeVals,clusterLabels,frontBackFace,autofilter,outfp)
    

    end = time.perf_counter()
    elapsedTime = end-start
    print(elapsedTime)

#%% run full workflow for an individual comparison if needed
lod = 0.02
minPts = 40
eps = 0.1
thresh = 0.15

cloud2 = r'Z:\Projects\LidarScans\GW\Alignments\GW_20200406\GW_20200406_aligned_merge_final.bin'
cloud1 = r'Z:\Projects\LidarScans\GW\Alignments\GW_20200314\GW_20200314_aligned_merge_final.bin'
ccfp = r'E:\fromSeagateExpansionDrive\I70\GW_MM117\Cluster_Luke_2022\20200314-20200406'
rf = joblib.load(r'E:\fromSeagateExpansionDrive\I70\GW_MM117\Cluster_Luke_2022\RFclassifier')
m3c2params = r'E:\fromSeagateExpansionDrive\I70\GW_MM117\Cluster_Luke_2022\m3c2_params_GW.txt'

maskfp = r'E:\fromSeagateExpansionDrive\I70\GW_MM117\Cluster_Luke_2022\GW_mask_v3_subsampled.bin'

c1 = cc.loadPointCloud(cloud1)
c2 = cc.loadPointCloud(cloud2)

(change1,change2) = calcChange(c1,c2,m3c2params,ccfp)

(rocks,changeVals,frontBackFace) = prepforcluster(change1,change2,[],lod)

clusterLabels = clustering(rocks,eps,minPts)

autofilter = classify(rocks,changeVals,clusterLabels,'mask',mask = maskfp,faces=frontBackFace)

# autofilter,stat = classify(rocks,changeVals,clusterLabels,'ml',classifier=rf,threshold=thresh,faces=frontBackFace)

write_outputs(rocks,changeVals,clusterLabels,frontBackFace,autofilter,ccfp)