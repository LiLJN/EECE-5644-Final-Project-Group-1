clc,
clear all,

alldata = 'DATASET.xlsx';
[part3_train part3_test] = readxlsx(alldata);
part3_model = fitcsvm(part3_train.features,part3_train.class);
sv = part3_model.SupportVectors;
figure(1);
gscatter(part3_train.features(:,1),part3_train.features(:,2),part3_train.class);
hold on,
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10);
xlabel('Age');
ylabel('Money spent');
cvmodel = crossval(part3_model);
missclass = kfoldLoss(cvmodel);
disp(missclass);


part3_model_2 = fitcsvm(part3_train.features,part3_train.class,'Holdout',0.15,['' ...
    'ClassNames'],{'Died','Survived'},'Standardize',true);
compact_svm_model = part3_model_2.Trained{1};
[label score] = predict(compact_svm_model,part3_test.features);
part3_table = table(part3_test.class,label,'VariableNames',{'True Label','Predicted Label'});
figure(2);
part3_confusionMat = confusionchart(part3_test.class,label);
figure(3);
roc = rocmetrics(part3_test.class,score,part3_model_2.ClassNames);
plot(roc);
title('ROC Curve for normal SVM');


part3_model3 = fitcsvm(part3_train.features(:,1:2),part3_train.class, ...
    'KernelFunction','gaussian','Standardize',true)
[x1Grid x2Grid] = meshgrid(min(part3_train.features(:,1)):0.1:max(part3_train.features(:,1)), ...
                  min(part3_train.features(:,2)):0.1:max(part3_train.features(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,score2] = predict(part3_model3,xGrid);
figure(4);
h(1:2) = gscatter(part3_train.features(:,1),part3_train.features(:,2),part3_train.class);
hold on,
% h(3) = plot(part3_train.features(part3_model3.IsSupportVector,1),...
%     part3_train.features(part3_model3.IsSupportVector,2),'ko','MarkerSize',10);
contour(x1Grid,x2Grid,reshape(score2(:,2),size(x1Grid)),[0 0],'k');
title('Classification using Gaussian Kernel SVM');
legend({'Died','Survived','Support Vectors'},'Location','Best');
hold off;
cvmodel = crossval(part3_model3);
missclass = kfoldLoss(cvmodel);
disp(missclass);

[label3,score3] = predict(part3_model3,part3_test.features(:,1:2));
figure(5);
roc = rocmetrics(part3_test.class,score3,part3_model3.ClassNames);
plot(roc);
title('ROC Curve for Gaussian Kernel SVM');



