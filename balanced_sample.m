function sampled_points = balanced_sample(pts,truth, numpts)
%INPUT: a vector of labels 0 through 4
%OUTPUT: a vector of samples, N=numpts per class

labels=pts(:,truth);

%class 0
if sum(labels==0) >= numpts
    
    class0=datasample(pts(labels==0,:),numpts);
    
else
    
    class0=datasample(pts(labels==0,:),sum(labels==0));
    
end

%class 1
if sum(labels==1) >= numpts
    
    class1=datasample(pts(labels==1,:),numpts);
    
else
    
    class1=datasample(pts(labels==1,:),sum(labels==1));
    
end

%class 2
if sum(labels==2) >= numpts
    
    class2=datasample(pts(labels==2,:),numpts);
    
else
    
    class2=datasample(pts(labels==2,:),sum(labels==2));
    
end

%class 3
if sum(labels==3) >= numpts
    
    class3=datasample(pts(labels==3,:),numpts);
    
else
    
    class3=datasample(pts(labels==3,:),sum(labels==3));
    
end

%class 4
if sum(labels==4) >= numpts
    
    class4=datasample(pts(labels==4,:),numpts);
    
else
    
    class4=datasample(pts(labels==4,:),sum(labels==4));
    
end

sampled_points=[class0;class1;class2;class3;class4];






