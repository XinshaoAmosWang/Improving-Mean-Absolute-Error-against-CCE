

CCE_baseline = [ 47.07,  44.70, 44.60, 44.65, 46.82];
CCE_baseline = [ 44.70, 44.60, 44.65];
means(1) = mean(CCE_baseline);
stds(1) = std(CCE_baseline);

CCE_L2 = [ 50.01, 51.50, 51.95, 51.08, 51.45 ];
CCE_L2 = [ 51.50, 51.95, 51.08, 51.45 ];
means(2) = mean(CCE_L2);
stds(2) = std(CCE_L2);

CCE_Dropout = [ 46.93, 42.40, 46.19, 46.33, 47.19 ];
CCE_Dropout = [ 46.93, 46.19, 46.33, 47.19 ];
means(3) = mean(CCE_Dropout);
stds(3) = std(CCE_Dropout);


CCE_L2_Dropout = [ 53.08, 51.35, 53.06, 52.83, 52.31 ];
CCE_L2_Dropout = [ 53.08, 53.06, 52.83, 52.31];
means(4) = mean(CCE_L2_Dropout);
stds(4) = std(CCE_L2_Dropout);

