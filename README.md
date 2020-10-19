# terpunkto
Calculate point cloud descriptors for machine learning

Point cloud descriptors are very useful for classification and segmentation tasks, but for those without a strong background in computer science it can be difficult and time-consuming to code application-specific features. Terpunkto is a function that calculates some simple neighborhood desciptors in multiple spherical radii, including geometric (e.g. omnivariance, eigentropy, etc.), slope (i.e. normal vector), color, and texture. The user can specify the set of radii and set of features to be calculated. 

These features were originally intended for geoscience research applications and therefore computational efficiency and point density variation are not major concerns, although the neighborhood search method could be modified without too much difficulty using e.g. a heirarchical subsampling scheme to mitigate any potential issues caused by this.

This code was used in the following publications:

Weidner, L., Walton, G., 2020. Monitoring and Modeling of the DeBeque Canyon Landslide Complex in Three Dimensions. Presented at the 54th U.S. Rock Mechanics/Geomechanics Symposium, American Rock Mechanics Association.

Weidner, Luke, Walton, G., Kromer, R., 2020. Generalization considerations and solutions for point cloud hillslope classifiers. Geomorphology 107039. https://doi.org/10.1016/j.geomorph.2020.107039

Weidner, L., Walton, G., Kromer, R., 2020. Automated Rock Slope Material Classification Using Machine Learning. Presented at the 54th U.S. Rock Mechanics/Geomechanics Symposium, American Rock Mechanics Association.

Weidner, L., Walton, G., Kromer, R., 2019. Classification methods for point clouds in rock slope monitoring: A novel machine learning approach and comparative analysis. Engineering Geology 263, 105326. https://doi.org/10.1016/j.enggeo.2019.105326
