#trabecular bone score(TBS)实现

## 介绍 introduction
骨小梁评分的python实现，但原版没给什么细节，故不能保证与原版完全相同。

## 计算方式 Method
原版实现  
每个像素作为初始点  
距离为1时  
对每个初始点M0，在360范围随机定一个方向θ。  
对每个初始点M0，θ方向距离为k的点，计算与初始点的灰度的方差，求平均，记为V(k)。  
距离为2-10，同上。  
最小二乘拟合二元函数，横轴为log[10](k)，纵轴为log1p[V(1)]V(k)  
原点斜率即为TBS  

## 参考文献 reference
Laurent Pothuaud, Pascal Carceller, Didier Hans, Correlations between grey-level variations in 2D projection images (TBS) and 3D microarchitecture: Applications in the study of human trabecular bone microarchitecture, Bone, Volume 42, Issue 4, 2008, Pages 775-787,
https://doi.org/10.1016/j.bone.2007.11.018.