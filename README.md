# terpunkto
Calculate point cloud descriptors for machine learning.

This code is intended to be a plug and play program for calculating a variety of point cloud geometric and color features in MATLAB. This functions similarly to CloudCompare, but with the added ability to:
- calculate color and intensity statistics, 
- handle balanced sampling of labeled point clouds automatically, and 
- interface directly with machine learning workflows in MATLAB without needing to load the features from CloudCompare

Point cloud descriptors are very useful for classification and segmentation tasks, but for those without a strong background in computer science it can be difficult and time-consuming to code application-specific features. Terpunkto is a function that calculates some simple neighborhood desciptors in multiple spherical radii, including geometric (e.g. omnivariance, eigentropy, etc.), slope (i.e. normal vector), color, and texture. The user can specify the set of radii and set of features to be calculated. 

These features were originally intended for geoscience research applications and therefore computational efficiency and point density variation are not major concerns. However, I also added the ability to use heirarchical subsampling features, which somewhat restricts the features that can be calculated and reduces accuracy a bit, but in exchange is much much faster and less sensitive to variations in point density.

This code was used in the following publications:

Weidner, L., Walton, G., Krajnovich, A., 2021. Classifying rock slope materials in photogrammetric point clouds using robust color and geometric features. ISPRS Journal of Photogrammetry and Remote Sensing 176, 15–29. https://doi.org/10.1016/j.isprsjprs.2021.04.001

Weidner, Luke, Walton, G., Kromer, R., 2020. Generalization considerations and solutions for point cloud hillslope classifiers. Geomorphology 107039. https://doi.org/10.1016/j.geomorph.2020.107039

Weidner, L., Walton, G., Kromer, R., 2020. Automated Rock Slope Material Classification Using Machine Learning. Presented at the 54th U.S. Rock Mechanics/Geomechanics Symposium, American Rock Mechanics Association.

Weidner, L., Walton, G., Kromer, R., 2019. Classification methods for point clouds in rock slope monitoring: A novel machine learning approach and comparative analysis. Engineering Geology 263, 105326. https://doi.org/10.1016/j.enggeo.2019.105326
