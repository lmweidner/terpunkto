# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 10:06:00 2022
@author: LWeidner
"""

import numpy as np
import cloudComPy as cc
import time
# import colorsys
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn. impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
from skimage.color import rgb2lab
cc.initCC()  # to do once before using plugins or dealing with numpy


# %%
fp = R'Z:\PersonalShare\Luke\terpunkto\Python\Slope3.bin'
cloud = cc.loadPointCloud(fp)
print('...loaded cloud with size: ',cloud.size())
radii = [3,1.5,0.75]
minDistDownsample = 0.3
useColor = 1
useGeometry = 1
# %% downsample raw point cloud using min space between points
print('...downsampling')
params = cc.SFModulationParams()
refCloud = cc.CloudSamplingTools.resampleCloudSpatially(cloud, minDistDownsample, params)
print('...finished downsampling')
(subsampledCloud, res) = cloud.partialClone(refCloud)
octree = subsampledCloud.computeOctree()
print('...downsampled cloud and computed octree, new cloud has size: ',subsampledCloud.size())
labelSf = subsampledCloud.getScalarField(0)
label = labelSf.toNpArrayCopy()
# %% convert colors from RGB to HSV (or LAB, let's see if it works)
if useColor:
    colRGB = subsampledCloud.colorsToNpArrayCopy()[:,0:3]
    # colHSV=np.empty_like(colRGB)
    # for i,row in enumerate(colRGB):
    #     colHSV[i,:]=colorsys.rgb_to_hsv(row[0],row[1],row[2])
    colLAB=np.empty_like(colRGB)    
    colLAB = rgb2lab(colRGB,channel_axis=1)
        
        
        
    print('...converted colors from RGB to LAB')
# %% calculate features
if useGeometry:
    print('...starting feature calculation (this will take several minutes)')
    for radius in radii:
        start = time.perf_counter()
        cc.computeFeature(cc.GeomFeature.Linearity,radius,[subsampledCloud])
        cc.computeFeature(cc.GeomFeature.Planarity,radius,[subsampledCloud])
        cc.computeFeature(cc.GeomFeature.Omnivariance,radius,[subsampledCloud])
        cc.computeFeature(cc.GeomFeature.Sphericity,radius,[subsampledCloud])
        cc.computeFeature(cc.GeomFeature.Verticality,radius,[subsampledCloud])
        end = time.perf_counter()
        elapsedTime = end-start
        print(elapsedTime)
        
    print('...finished feature calculation')
# %% convert scalar fields to Nunmpy arrays for Sklearn
nsf = subsampledCloud.getNumberOfScalarFields()
sfs = []
for i in range(nsf):
    sf = subsampledCloud.getScalarField(i)
    sfs.append(sf.toNpArrayCopy())
    
# xyz = subsampledCloud.toNpArrayCopy() # may use this in the future
allSf = np.vstack(sfs).transpose()[:,1:] # the first scalar is the label and is excluded

if useColor:
    DataMatrix = np.hstack([allSf,colLAB])
    print(np.shape(DataMatrix))
else:
    DataMatrix = np.copy(allSf)
    print(np.shape(DataMatrix))
# %% MACHINE LEARNING TIME

hasLabel = label>-1 # label of -1 indicates no label

# get only points that are labeled
labeledLabels = label[hasLabel]
labeledData = DataMatrix[hasLabel,:]


# fill nans and randomly undersample for training
imp1 = SimpleImputer(missing_values=np.nan, strategy='mean').fit(labeledData)
data_imp = imp1.transform(labeledData)

print('...undersampling')
rus = RandomUnderSampler(random_state=0,sampling_strategy='auto')
X_resampled, y_resampled = rus.fit_resample(data_imp, labeledLabels)

print('...training Random Forest')
# train shallow neural network
# nn = MLPClassifier(hidden_layer_sizes=(20,5),max_iter=300,early_stopping=True).fit(X_resampled,y_resampled)
nn = RandomForestClassifier(verbose=1,n_jobs=-1).fit(X_resampled,y_resampled)

# fill nans in main cloud data
imp2 = SimpleImputer(missing_values=np.nan, strategy='mean').fit(DataMatrix)
outFeatures = imp2.transform(DataMatrix)

print('...predicting outputs')
fullCloudLabel = nn.predict(outFeatures)
LabelProbs = nn.predict_proba(outFeatures)
Confidence = np.max(LabelProbs,axis=1)
# %% convert results to scalar fields and write output cloud

print('...writing results to BIN file')
nClasses = np.size(np.unique(labeledLabels))

for iClass in range(nClasses):
    
    classInd=subsampledCloud.addScalarField('class '+str(iClass)+' probability')
    sf = subsampledCloud.getScalarField(classInd)
    sf.fromNpArrayCopy(np.float32(LabelProbs[:,iClass]))
    
classInd=subsampledCloud.addScalarField('Predicted Label')
sf = subsampledCloud.getScalarField(classInd)
sf.fromNpArrayCopy(fullCloudLabel)  

confInd=subsampledCloud.addScalarField('Confidence')
sf = subsampledCloud.getScalarField(confInd)
sf.fromNpArrayCopy(np.float32(Confidence))

res = cc.SavePointCloud(subsampledCloud,R'Z:\PersonalShare\Luke\terpunkto\Python\Slope3Pred.bin')
print('saved cloud with result ',res)
