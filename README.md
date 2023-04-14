# DeepTOF

## Result 1: real data (revision of table 5 in orginal paper)
|Variants | BP | PF|
|---------|----|---|
|DeepTOF  | 0.4596 | 0.4890|
|DeepTOF w/o C | -0.0215 (-4.7%) | -0.0246 (-5%)|
|DeepTOF w/o M | -0.0294 (-6.4%)| -0.0118 (-2.4%)|
|DeepTOF w/o S | -0.0188 (-4.1%)| -0.0147 (-3%)|
|DeepTOF w/o M,S | -0.0290 (-6.3%)| -0.0294 (-6%)|
|DeepTOF w/o C,S | -0.0222 (-4.8%)| -0.0041 (-0.8%)|
|DeepTOF w/o C,M | -0.0225 (-4.9%)| -0.0281 (-5.7%)|
|DeepTOF w/o C,M,S | -0.0280 (-6.1%)| -0.0365 (-7.5%)|

## Result 2 (simulated data)
### Task 1: Surgical group: Gaussian noise added with parameters N(0.5, 0.1), Nonoperative group: Gaussian noise added with parameters N(0, 0.1), Labels: Gaussian noise added with parameters N(0, 0.1), Mask ratio: 10%  
|Variants | BP | PF|  
|---------|----|---|   
|DeepTOF  | 0.4394 | 0.4798|
|DeepTOF w/o C | -0.0219 (-5%) | -0.0372 (-7.8%)|    
|DeepTOF w/o M | -0.0370 (-8.4%)| -0.0207 (-4.3%)|   
|DeepTOF w/o S | -0.0261 (-5.9%)| -0.0339 (-7.1%)| 
|DeepTOF w/o M,S | -0.0366 (-8.3%)| -0.0321 (-6.7%)|  
|DeepTOF w/o C,S | -0.0231 (-5.3%)| -0.0446 (-9.3%)|  
|DeepTOF w/o C,M | -0.0475 (-10.8%)| -0.0530 (-11%)|  
|DeepTOF w/o C,M,S | -0.0489 (-11.1%)| -0.0597 (-12.4%)|  


### Task 2: Surgical group: Gaussian noise added with parameters N(0.6, 0.2), Nonoperative group: Gaussian noise added with parameters N(0, 0.1), Labels: Gaussian noise added with parameters N(0, 0.2), Mask ratio: 20%    
|Variants | BP | PF|
|---------|----|---|
|DeepTOF  | 0.4312 | 0.4751|
|DeepTOF w/o C | -0.0263 (-6.1%)| -0.0399 (-8.4%)|
|DeepTOF w/o M | -0.0396 (-9.2%)| -0.0427 (-9%)|
|DeepTOF w/o S | -0.0241 (-5.6%)| -0.0341 (-7.2%)|
|DeepTOF w/o M,S | -0.0441 (-10.2%)| -0.0429 (-9%)|
|DeepTOF w/o C,S | -0.0318 (-7.4%)| -0.0437 (-9.2%)|
|DeepTOF w/o C,M | -0.0520 (-12.1%)| -0.0568 (-12%)|
|DeepTOF w/o C,M,S | -0.0532 (-12.3%)| -0.0642 (-13.5%)|

### Task 3: Surgical group: Gaussian noise added with parameters N(0.7, 0.3), Nonoperative group: Gaussian noise added with parameters N(0, 0.1), Labels: Gaussian noise added with parameters N(0, 0.3), Mask ratio: 30%
|Variants | BP | PF|  
|---------|----|---|  
|DeepTOF  | 0.4284 | 0.4724|.  
|DeepTOF w/o C | -0.0389 (-9.1%)| -0.0423 (-9%)|  
|DeepTOF w/o M | -0.0600 (-14%)|-0.0471 (-10%)|  
|DeepTOF w/o S | -0.0592 (-13.8%)| -0.0553 (-11.7%)|    
|DeepTOF w/o M,S | -0.0442 (-10.3%)| -0.0586 (-12.4%)|    
|DeepTOF w/o C,S | -0.0493 (-11.5%)| -0.0534 (-11.3%)|    
|DeepTOF w/o C,M | -0.0654 (-15.3%)| -0.0475 (-10.1%)|   
|DeepTOF w/o C,M,S | -0.0634 (-14.8%)| -0.0684 (-14.5%)|   


## Result 3
### Result 3.1, Average representation distance between treated and untreated group by DeepTOF w/ counterfactual modeling, average difference: 0.3623.

|FIELD1|surgical          |nonsurgical       |
|------|------------------|------------------|
|0     |12.8983           |13.0026           |
|1     |13.3972           |13.2094           |
|2     |11.9706           |11.8053           |
|3     |9.9960            |10.3158           |
|4     |10.8122           |11.1825           |
|5     |11.1622           |10.8628           |
|6     |11.9716           |12.0424           |
|7     |11.5357           |11.6998           |
|8     |10.7058           |10.7424           |
|9     |10.8505           |11.1801           |
|10    |10.3774           |10.6389           |
|11    |14.2044           |13.4578           |
|12    |14.0589           |14.8989           |
|13    |11.2666           |11.1015           |
|14    |11.6532           |11.1093           |
|15    |12.4525           |12.0294           |
|16    |11.0256           |11.3695           |
|17    |15.2408           |16.0530           |
|18    |16.0615           |16.9596           |
|19    |11.2941           |11.1299           | 


### Result 3.2, Average representation distance between treated and untreated group by DeepTOF w/o counterfactual modeling, average difference: 1.7109.

|FIELD1|surgical          |nonsurgical       |
|------|------------------|------------------|
|0     |10.00980568       |12.61927032       |
|1     |15.52026272       |13.17067242       |
|2     |12.64957428       |11.47766495       |
|3     |9.049478531       |12.61666679       |
|4     |10.06480026       |10.71032619       |
|5     |10.63503933       |10.31456947       |
|6     |10.71271515       |11.7099123        |
|7     |11.17810726       |11.27660179       |
|8     |9.635328293       |13.75091648       |
|9     |10.29611492       |10.76837063       |
|10    |9.479779243       |12.77809525       |
|11    |14.18276501       |13.43580818       |
|12    |11.95788288       |14.79267788       |
|13    |10.87122345       |10.67739296       |
|14    |13.81731033       |10.46208763       |
|15    |12.42627811       |10.94762802       |
|16    |10.13836193       |10.51164341       |
|17    |14.98858833       |15.77536869       |
|18    |15.15091705       |17.04966736       |
|19    |13.21783066       |10.31321049       |
  

## Result 4, Feature importance of the selected features.  
![image](https://raw.githubusercontent.com/HangtingYe/DeepTOF/main/feature%20importance/0.jpg)   
Each scatter point represents the SHAP value of a single sample. The color of the scatter points represents the value of the corresponding feature for that sample. Red points indicate higher feature values, while blue points indicate lower feature values, with the intensity of the color indicating the magnitude of the feature value. Since SHAP value provides feature importance for a specific model prediction, we only present one of our outcomes here. For the full predictions explainations, please kindly refer to the file DeepTOF/feature importance. 

## Result 5, NRMSE results.
Comparison of all models' predictive performance in terms of Bodily Pain (BP) and Physical Function (PF) with a different number of features. Feature selector was co-trained with DeepTOF to select a different number of features (i.e. M is set to 20, 30, 40, 50, 60 and 70, here, full represents 131 features). Using M features as input, the results show the comparison of all models' performance in terms of NRMSE. For each method, NRMSE scores averaged over surgical and nonoperative treatment are reported (lower is better). The reported performance is averaged over 10 independent runs. The best results are highlighted in bold. 
|            |M=20   |M=20    |M=30    |M=30    |M=40    |M=40    |M=50    |M=50    |M=60    |M=60    |M=70    |M=70    |Full    |Full   |
|------------|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-------|
|Model       |BP     |PF      |BP      |PF      | BP     |PF      |BP      |PF      |BP      |PF      |BP      |PF      |BP      |PF     |
|Lasso       |0.3407 | 0.3224 | 0.3515 | 0.3360 | 0.3407 | 0.3224 | 0.3400 | 0.3258 | 0.3361 | 0.3203 | 0.3345 | 0.3186 | 0.3343 | 0.3186|
|SVR         |0.3510 | 0.3273 | 0.3641 | 0.3522 | 0.3485 | 0.3309 | 0.3438 | 0.3310 | 0.3401 | 0.3247 | 0.3374 | 0.3237 | 0.3350 | 0.3214|
|K-NN        |0.3731 | 0.3593 | 0.3838 | 0.3820 | 0.3738 | 0.3662 | 0.3692 | 0.3673 | 0.3697 | 0.3665 | 0.3658 | 0.3654 | 0.3678 | 0.3652|
|RandomForest|0.3390 | 0.3172 | 0.3508 | 0.3312 | 0.3389 | 0.3178 | 0.3357 | 0.3214 | 0.3339 | 0.3160 | 0.3316 | 0.3157 | 0.3313 | 0.3161|               
|LightGBM    |0.3440 | 0.3290 | 0.3515 | 0.3401 | 0.3443 | 0.3294 | 0.3431 | 0.3323 | 0.3389 | 0.3275 | 0.3377 | 0.3262 | 0.3383 | 0.3255|
|ResNet      |0.3272 | 0.3046 | 0.3395 | 0.3239 | 0.3268 | 0.3082 | 0.3262 | 0.3120 | 0.3223 | 0.3060 | 0.3218 | 0.3057 | 0.3230 | 0.3098|
|DeepTOF     |**0.3098** | **0.2632** | **0.2945** | **0.2757** | **0.3043** | **0.2736** | **0.2845** | **0.2816** | **0.2977** | **0.2860** | **0.2947** | **0.2653** | **0.3012** | **0.2765** |
