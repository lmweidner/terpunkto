# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:17:57 2022

@author: AmberHill
"""
import numpy as np
import cloudComPy as cc
if cc.isPluginPCL():
    import cloudComPy.PCL
if cc.isPluginM3C2():
    import cloudComPy.M3C2

#%%
basefp = r'Z:\Projects\LidarScans\GW\Alignments\GW_20190818\GW_20190818__merged_subsampled.txt'
outfp = r'Z:\Projects\LidarScans\GW\Alignments\GW_20200406\\'
c1fp = r'Z:\Projects\LidarScans\GW\Raw Scans\GW_20200406\GW_20200406_001.e57'
c2fp = r'Z:\Projects\LidarScans\GW\Raw Scans\GW_20200406\GW_20200406_003.e57'
c3fp = r'Z:\Projects\LidarScans\GW\Raw Scans\GW_20200406\GW_20200406_005.e57'
fps = [c1fp,c2fp,c3fp]
minDistDownsample = 0.2
radius = 1.0
#%%
cloudList = []
for fp in fps:
    c = cc.loadPointCloud(fp)
    cloudList.append(c)

basecloud = cc.loadPointCloud(basefp)
#%% downsample
cloudsDownsampled = []
for cloud in cloudList:
    
    params = cc.SFModulationParams()
    refCloud = cc.CloudSamplingTools.resampleCloudSpatially(cloud, minDistDownsample, params)
    print('...finished downsampling')
    (cdownsampled, res) = cloud.partialClone(refCloud)
    cloudsDownsampled.append(cdownsampled)
    
refCloud = cc.CloudSamplingTools.resampleCloudSpatially(basecloud, minDistDownsample, params)
print('...finished downsampling')
(basedownsampled, res) = basecloud.partialClone(refCloud)
#%% normals
cc.computeNormals(cloudsDownsampled+[basedownsampled],defaultRadius = radius)
#%% compute fast global registration on downsampled clouds.
transmats = []
for i,(dsCloud,cloud) in enumerate(zip(cloudsDownsampled,cloudList)):

    fgr = cc.PCL.FastGlobalRegistrationFilter()
    fgr.setParameters(basedownsampled,[dsCloud],0)
    res = fgr.compute()
    print(res)
    transmat = fgr.getTransformation()
    print(transmat.data())
    transmats.append(transmat)
    
    cloud.applyRigidTransformation(transmat)
    
    res = cc.SavePointCloud(cloud,outfp+f'cloud_{i}_roughAlign.bin')
    cc.SavePointCloud(dsCloud,outfp+f'cloud_{i}_roughAlign_sparse.bin')
    print(res)

# cc.SavePointCloud(basedownsampled,r'Z:\PersonalShare\Luke\alignment\baseout.bin')
# cc.SavePointCloud(basecloud,r'Z:\PersonalShare\Luke\alignment\basefull.bin')
#%% rerun failed
failed = 0
if True:
    for i,(dsCloud,cloud) in enumerate(zip(cloudsDownsampled,cloudList)):
        if i == failed: # only works for one at a time because you can't have a list of len 1
            cc.computeNormals([dsCloud],defaultRadius=radius)
            cc.invertNormals([dsCloud])
            fgr = cc.PCL.FastGlobalRegistrationFilter()
            fgr.setParameters(basedownsampled,[dsCloud],0)
            res = fgr.compute()
            print(res)
            transmat = fgr.getTransformation()
            print(transmat.data())
            transmats.append(transmat)
        
            cloud.applyRigidTransformation(transmat)
     
            res = cc.SavePointCloud(cloud,outfp+f'cloud_{i}_roughAlign_retry.bin')
            cc.SavePointCloud(dsCloud,outfp+f'cloud_{i}_roughAlign_sparse_retry.bin')
            print(res)
#%% ICP ALIGNMENT (loading roughly aligned clouds if needed. ignore this if running straight through from rough alignment)
basefp = r'Z:\Projects\LidarScans\GW\Alignments\GW_20190818\GW_20190818__merged_subsampled.txt'
outfp = r'Z:\Projects\LidarScans\GW\Alignments\GW_20200314\\'
c1fp = r'Z:\Projects\LidarScans\GW\Alignments\GW_20200314\cloud_0_roughAlign.bin'
c2fp = r'Z:\Projects\LidarScans\GW\Alignments\GW_20200314\cloud_1_roughAlign.bin'
c3fp = r'Z:\Projects\LidarScans\GW\Alignments\GW_20200314\cloud_2_roughAlign.bin'
fps = [c1fp,c2fp,c3fp]
cloudList = []
for fp in fps:
    c = cc.loadPointCloud(fp)
    cloudList.append(c)
    if c ==None:
        raise Exception('Cloud failed to load. Check filepath')

basecloud = cc.loadPointCloud(basefp)
#%% ICP ALIGNMENT

for i,cloud in enumerate(cloudList):
    
    ICPres = cc.ICP(data=cloud,model=basecloud,minRMSDecrease = 1.e-5,
                    maxIterationCount = 20,
                    randomSamplingLimit = 80000,
                    removeFarthestPoints = True,
                    method = cc.CONVERGENCE_TYPE.MAX_ERROR_CONVERGENCE,
                    finalOverlapRatio = 0.7,
                    adjustScale = False
                    )

    cloud.applyRigidTransformation(ICPres.transMat)
    res = cc.SavePointCloud(cloud,outfp+f'cloud_{i}_ICPalign.bin')
    print(res)
    
#%% compute quick M3C2 to assess results


for i,cloud in enumerate(cloudList):

    if i == 0:
        mergedCloud = cloud.cloneThis()
        
    else:
        mergedCloud.fuse(cloud)

assert mergedCloud.size() == np.sum([x.size() for x in cloudList])

cc.SavePointCloud(mergedCloud,outfp+'mergedCloud.bin')
#%%##
M3C2params = r'Z:\Projects\LidarScans\GW\Alignments\GW_20200220\m3c2params.txt'
# cc.M3C2.M3C2gessParamsToFile([basecloud,mergedCloud],M3C2params,False)

quickcompare = cc.M3C2.computeM3C2([basecloud,mergedCloud],M3C2params)
change = quickcompare.getScalarField(2).toNpArray()
filt = (change>-0.03 & change < 0.03)
print(np.nanmean(change[filt]),np.nanstd(change[filt]))

cc.SavePointCloud(quickcompare,outfp+'quickcompare.bin')

#%%
# c1 = cc.loadPointCloud(r'Z:\PersonalShare\Luke\alignment\t0.bin')
# c2 = cc.loadPointCloud(r'Z:\PersonalShare\Luke\alignment\t1.bin')
# quickcompare = cc.M3C2.computeM3C2([c1,c2],M3C2params)
# sf = quickcompare.getScalarField(2).toNpArray()
# print(np.nanmean(sf),np.nanstd(sf))
# cc.SavePointCloud(quickcompare,outfp+'quickcompare.bin')
