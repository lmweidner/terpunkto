function t = terpunkto(inname,indices,scales,subsampleflag,intensityflag,colorflag,numpts_subsample)

%{ 

Terpunkto v1

Calculate point cloud descriptors for either a full point cloud or for a
set of labeled points in the cloud. This code handles geometric, slope,
color, and lidar intensity features of point clouds. It can be modified
relatively easily to enable the calculation of radius neighborhood
features/statistics for any point cloud scalar field.

Author:
Luke Weidner
PhD Candidate, Colorado School of Mines

Copyright 2020 (CC BY 4.0)
Associated with the following publications:

(Version 2) Weidner, L., Walton, G., Kromer, R., 2020. Generalization considerations and solutions for point cloud hillslope classifiers. Geomorphology 107039. https://doi.org/10.1016/j.geomorph.2020.107039
(Version 1) Weidner, L., Walton, G., Kromer, R., 2019. Classification methods for point clouds in rock slope monitoring: A novel machine learning approach and comparative analysis. Engineering Geology 263, 105326. https://doi.org/10.1016/j.enggeo.2019.105326
(Version 3) Weidner et al., (in prep). Deriving robust and repeatable color features for classifying rock slope materials in photogrammetry point clouds.

INPUTS:

inname : filepath to input cloud .txt
outputname : filepath to output feature matrix and .ply point cloud
indices : list of integers indicating which columns in the point cloud
correspond to which items (see below)
scales : list of search radii at which to calculate multi-scale features
    e.g. [1,0.75,0.5,0.25,0.1]
subsampleflag  : 0 = balanced subsample for core points within labeled points, 1 = randomly sample the entire cloud
intensityflag : 1 = calculate Walton features for the column indicated by intensity_ind, 0 = do nothing
colorflag : 1 = calculate color features, 0 = do nothing
numpts_subsample : total number of points subsampled if subsampleflag == 1 OR number of points subsampled PER CLASS if subsampleflag == 0

Order of indices:

1 = x
2 = y
3 = z
4 = red
5 = green
6 = blue
7 = Nx
8 = Ny
9 = Nz
10 = truth label
11 = intensity

Example 1 : [1,2,3,4,5,6,8,9,10,7,0] 
Last one is zero because we're not calculating intensities so that index
doesn't matter. Also, this cloud has a truth label in column 7

Example 2 : [1,2,3,0,0,0,4,5,6,0,7]
This cloud only has xyz, normal, and intensity (i.e. typical lidar)

%}
%% PARAMETERS

xind = indices(1); % X coordinate column
yind = indices(2); % Y ... ...
zind = indices(3); % Z ... ...
rind = indices(4); % Red color column
gind = indices(5); % Green ... ...
bind = indices(6); % Blue ... ...
nxind = indices(7); % X Normal Component (Nx) column
nyind = indices(8); % Y ...
nzind = indices(9); % Z ...
truth_ind = indices(10); % Label index (optional if subsampleflag == 1)
intensity_ind = indices(11); % Single-channel intensity (from lidar)

%% LOAD POINT CLOUD
f=waitbar(0.05,'loading point cloud...');
pts_orig=dlmread(inname,',',1,0); %load point cloud to be classified
disp('loaded raw cloud')
%% CONVERT NORMALS TO SLOPE ANGLES IN 0-180 RANGE
waitbar(0.2,f,'calculating slope angles...');
newslangs=zeros(size(pts_orig,1),1);

dz=pts_orig(:,nzind);
dy=pts_orig(:,nyind);
dx=pts_orig(:,nxind);

newslangs(dz<0)=real(90+acosd((dx(dz<0).^2+dy(dz<0).^2)./sqrt(dx(dz<0).^2+dy(dz<0).^2)));
newslangs(dz>0)=real(90-acosd((dx(dz>0).^2+dy(dz>0).^2)./sqrt(dx(dz>0).^2+dy(dz>0).^2)));
disp('calculated slope')
%% Convert RGB colors to HSV and LAB color spaces
waitbar(0.4,f,'converting colors...');
if colorflag == 1
    
    RGB=pts_orig(:,[rind,gind,bind])./255;
    
    LAB=rgb2lab(RGB);
    
end

if intensityflag == 1
    intensities = pts_orig(:,intensity_ind);
end

disp('converted colors')
%% Create pointcloud objects for original cloud and core points cloud
waitbar(0.6,f,'creating core points...');

tp=pointCloud(pts_orig(:,[xind,yind,zind]));

if subsampleflag==1  % subsample points from the entire cloud (i.e. don't care if they're labeled or not. more useful for prediction on unseen data
    pts_test=datasample(pts_orig,numpts_subsample);
    tp2=pointCloud(pts_test(:,[xind,yind,zind]));
    numloop=size(pts_test,1);
    disp(['Number of core points: ',num2str(numloop)])
else % subsample set number of points for each class (up to 5 classes supported, see "balanced_sample" function)
   
    pts_test=balanced_sample(pts_orig,truth_ind,numpts_subsample); %extract points for a random stratified sample of N points per class
    tp2=pointCloud(pts_test(:,[xind,yind,zind])); %point cloud object for core cloud
    numloop=size(tp2.Location,1);
    disp(['Number of core points: ',num2str(numloop)]) 
    
end

disp('loaded core points')
%% INITIALIZE OUTPUT MATS
numscales=length(scales);
count=0;
waitbar(0.8,f,'initializing...');

geomstat=NaN(size(pts_test,1),10*numscales);
slopes=NaN([size(pts_test,1),4*numscales]);

if colorflag == 1
    LAB1stat=NaN([size(pts_test,1),3*numscales]);
    LAB2stat=NaN([size(pts_test,1),9*numscales]);
    GLCMstat=NaN([size(pts_test,1),4]);
    LABtexturestat=NaN([size(pts_test,1),192]);
    LABstd=NaN(size(pts_test,1),3*numscales);
    [gridx,gridy]=meshgrid(-0.7:0.2:0.7,-0.7:0.2:0.7); %initialize grid for color texture features
    grid1=[gridx(:),gridy(:),zeros(length(gridy(:)),1)];
    LABpoint=NaN([size(pts_test,1),3]);
end

if intensityflag == 1
    intensstat=NaN([size(pts_test,1),4*numscales]);
end

disp('initialization done. starting multi-scale feature calculation...')
%% Calculate descriptors
tic
waitbar(0.05,f,'calculating multi-scale features...');
for i=1:numloop
    %display progress
    if i==round(length(pts_test)*1/10)
        waitbar(0.1,f);
    elseif i==round(length(pts_test)*2/10)
        waitbar(0.2,f);
    elseif i==round(length(pts_test)*3/10)
        waitbar(0.3,f);
    elseif i==round(length(pts_test)*4/10)
        waitbar(0.4,f);
    elseif i==round(length(pts_test)*5/10)
        waitbar(0.5,f);
    elseif i==round(length(pts_test)*6/10)
        waitbar(0.6,f);
    elseif i==round(length(pts_test)*7/10)
        waitbar(0.7,f);
    elseif i==round(length(pts_test)*8/10)
        waitbar(0.8,f);
    elseif i==round(length(pts_test)*9/10)
        waitbar(0.9,f);
    end
    
    %FIND NEIGHBROS IN RADIUS (IN ORIGINAL POINT CLOUD), AROUND
    %CORE POINT i
    
    [ind1,dist]=findNeighborsInRadius(tp,tp2.Location(i,:),max(scales));
    
    %Initialize temporary vectors to store descriptors

    geotemp=NaN(numscales,10);
    slopetemp=NaN(numscales,4);
    intenstemp=NaN(numscales,4);
    LAB1temp=NaN(numscales,3);
    
    LABstdtemp=NaN(numscales,3);
    
    LAB2temp=NaN(numscales,9);
    RGBtxttemp=NaN(64,3);
    LABtxttemp=NaN(64,3);
    
    %calculate overall texture features (Beretta et al., 2019)
    if colorflag == 1
        if length(ind1)>3
            xyz=pts_orig(ind1,[xind,yind,zind]);
            coeffs=pca(xyz);
            samples1=grid1*coeffs'+pts_orig(ind1(1),[xind,yind,zind]);
            
            for k=1:length(samples1)
                
                ind2=findNearestNeighbors(tp,samples1(k,:),4);
                RGBtxttemp(k,:)=[mean(RGB(ind2,1)),mean(RGB(ind2,2)),mean(RGB(ind2,3))];
                LABtxttemp(k,:)=[mean(LAB(ind2,1)),mean(LAB(ind2,2)),mean(LAB(ind2,3))];
                
            end
            
            %calculation GLCM features
            I1=rgb2gray(reshape(RGBtxttemp,8,8,3));
            props1=graycoprops(graycomatrix(I1,'GrayLimits',[]));
            GLCMstat(i,:)=[props1.Contrast,props1.Correlation,props1.Energy,props1.Homogeneity];
        end
    end
    
    %calculate direct point cloud radius features     
    
    for j=1:numscales
        
        if sum((dist<scales(j))==1)>2 %only calculate if there are at least 3 points in radius
            
            %Calculate geometric features
            
            radinds=ind1(dist<scales(j)); % get point indices in radius j
            
            eigvals=sort(eig(cov(pts_orig(radinds,[xind,yind,zind]))),'descend'); %calculate eigenvalues
            
            geotemp(j,:)=calcdesc(eigvals); %convert eigenvalues to geometric descriptors
            
            
            %Calculate slope features
            
            slopetemp(j,:)=[...
                nanmean(newslangs(radinds)),...
                nanstd(newslangs(radinds)),...
                skewness(newslangs(radinds),0),...
                kurtosis(newslangs(radinds),0)];
            
            %Calculate intensity features
            
            if intensityflag == 1 
                
                intenstemp(j,:)=[...
                    nanmean(intensities(radinds)),...
                    nanmean(intensities(radinds))-intensities(ind1(1)),...
                    intensities(ind1(1))-nanmin(intensities(radinds)),...
                    nanmax(intensities(radinds))-intensities(ind1(1))];

            end
            
            if colorflag == 1
                %Calculate color features
                
                %LAB
                LAB1temp(j,:)=[nanmean(LAB(radinds,1)),nanmean(LAB(radinds,2)),nanmean(LAB(radinds,3))];
                
                LABstdtemp(j,:)=[nanstd(LAB(radinds,1)),nanstd(LAB(radinds,2)),nanstd(LAB(radinds,3))];
                
                LAB2temp(j,:)=[... %2= relative features (Walton 2016)
                    nanmean(LAB(radinds,1))-LAB(ind1(1),1),...%RED %mean-point
                    LAB(ind1(1),1)-nanmin(LAB(radinds,1)),... %point - min
                    nanmax(LAB(radinds,1))-LAB(ind1(1),1),... %max - point
                    nanmean(LAB(radinds,2))-LAB(ind1(1),2),...%GREEN
                    LAB(ind1(1),2)-nanmin(LAB(radinds,2)),...
                    nanmax(LAB(radinds,2))-LAB(ind1(1),2),...
                    nanmean(LAB(radinds,3))-LAB(ind1(1),3),...%BLUE
                    LAB(ind1(1),3)-nanmin(LAB(radinds,3)),...
                    nanmax(LAB(radinds,3))-LAB(ind1(1),3)];
            end
        else
            
            count=count+1;
            
        end
    end
    
    if colorflag == 1
        LABpoint(i,:)=LAB(ind1(1),:);    
        LAB1stat(i,:)=reshape(LAB1temp',1,numscales*3);    
        LABstd(i,:)=reshape(LABstdtemp',1,numscales*3);
        LAB2stat(i,:)=reshape(LAB2temp',1,numscales*9);
        LABtexturestat(i,:)=reshape(LABtxttemp',1,192);
    end
    
    if intensityflag == 1
        intensstat(i,:)=reshape(intenstemp',1,numscales*4);
    end
    
    geomstat(i,:)=reshape(geotemp',1,numscales*10);
    slopes(i,:)=reshape(slopetemp',1,numscales*4);    

    
end
elapsedTime=toc;
close(f)
disp(['done. time elapsed: ',num2str(elapsedTime/60/60),' hours'])
disp('calculated descriptors')
disp(['Number of core points without enough neighbors: ',num2str(count)])

%% Outputs

t=struct('geom',real(geomstat),'slope',real(slopes)); %create structure and add geom/slope features

if colorflag == 1
    
    t.LABpoint=LABpoint; %point color
    t.meanLAB=LAB1stat; %color mean
    t.LABstd=LABstd; %color std
    t.WaltonLAB=LAB2stat; %relative color (Walton 2016)
    t.GLCM=GLCMstat;
    t.LABtexture=LABtexturestat; %color texture (GLCM and Beretta 2019)
    
end

if intensityflag == 1
    t.intensity=intensstat; %single-channel intensity (lidar only) Walton 2016 
end

if subsampleflag == 0
    t.truth=pts_test(:,truth_ind); %class label
end

t.points = tp2.Location;

% save(outputname,'t','-v7.3')

% pcwrite(tp2,[outputname,'.ply'])


