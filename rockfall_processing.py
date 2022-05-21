# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:39:20 2022

@author: sa
"""

import numpy as np
import cloudComPy as cc
import time
from sklearn.cluster import OPTICS,DBSCAN
if cc.isPluginM3C2():
    import cloudComPy.M3C2
# import colorsys
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn. impute import SimpleImputer
# from imblearn.under_sampling import RandomUnderSampler
# from skimage.color import rgb2lab
cc.initCC()  # to do once before using plugins or dealing with numpy


# %% M3C2 CALC
def calcChange(c1,c2,saveIntermediate,m3c2params,fp):
    cc.M3C2.initTrace_M3C2()
    
    print('calculating forwardChange')
    forwardChange = cc.M3C2.computeM3C2([c1,c2],m3c2params)
    print('calculating reverseChange')
    reverseChange = cc.M3C2.computeM3C2([c2,c1],m3c2params)
    print('done with change')
    

    if saveIntermediate:
        cc.SavePointCloud(forwardChange,fp+r'\forwardChange.bin')
        cc.SavePointCloud(reverseChange,fp+r'\reverseChange.bin')
        
    return forwardChange,reverseChange
# %% FILTER CHANGE BELOW LOD

def prepforcluster(forwardChange,reverseChange,clusterSubsample,saveIntermediate,lod,fp):
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
    
    if saveIntermediate:
        cc.SavePointCloud(mergedFalls,fp+r'\mergedFalls.bin')
    
    
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
def write_outputs(rocks,change,labels,faces,fp):
    for label in np.unique(labels):
        # if label!=-1:
        changeTemp = change[labels==label]
        faceTemp = faces[labels==label]
        xyz = rocks[labels==label,:]
        out = np.column_stack((xyz,faceTemp,changeTemp))
        f = fp + r'\rf_' + str(label) + '.txt'
        np.savetxt(f,out,delimiter=',',fmt='%.7f')
    print('wrote all outputs!')
# %% MAIN
cloudfp = r'C:\Users\sa\Desktop\Research\EHI\US74_Projects\E\rotated'
ccfp = r'C:\CloudComPy39\DATA'
m3c2params = ccfp+r'\m3c2_params.txt'
cloud1  = cloudfp+r'\E_20210325_classified.txt'
cloud2 = cloudfp+r'\E_20210621.txt'
lod = 0.02
minPts = 30
eps = 0.1
clusterSubsample = 0.01
saveIntermediate = True

#  LOAD CLOUDS
c1 = cc.loadPointCloud(cloud1)
c2 = cc.loadPointCloud(cloud2)

print('...loaded cloud with size: ',c1.size())
print('...loaded cloud with size: ',c2.size())

start = time.perf_counter()

(change1,change2) = calcChange(c1,c2,saveIntermediate,m3c2params,ccfp)
(rocks,change,faces) = prepforcluster(change1,change2,clusterSubsample,saveIntermediate,lod,ccfp)
L = clustering(rocks,eps,minPts)
write_outputs(rocks,change,L,faces,ccfp + r'\rockfalls')

end = time.perf_counter()
elapsedTime = end-start
print(elapsedTime)