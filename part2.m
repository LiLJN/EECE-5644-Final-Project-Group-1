clc,
clear all,

alldata = 'DATASET.xlsx';
[part2_train part2_test] = readxlsx(alldata);
part2_model_1 = fitcnb(part2_train.features(:,1:2),part2_train.class, ...
    'ClassNames',{'Died','Survived'});
e = min(part2_train.features(:,1)):0.1:max(part2_train.features(:,1));
f = min(part2_train.features(:,2)):0.1:max(part2_train.features(:,2));
% h = min(part2_train.features(:,3)):0.01:max(part2_train.features(:,3));
[x1 x2] = meshgrid(e,f);
x = [x1(:) x2(:)];
ms = predict(part2_model_1,x);
figure(1);
gscatter(x1(:),x2(:),ms,'cy');
hold on,
gscatter(part2_test.features(:,1),part2_test.features(:,2), ...
    part2_test.class(:),'rb','xo',5);

xlabel('Age');
ylabel('Money spent');
title('Classification using Naive Bayes Classifier');
hold off;

part2_model_2 = fitcnb(part2_train.features,part2_train.class, ...
    'ClassNames',{'Died','Survived'});
[label2 scores] = predict(part2_model_2,part2_test.features);
table(part2_test.class,label2,'VariableNames',...
    {'True Label','Predicted Label'})
figure(2);
ConfusionMat = confusionchart(part2_test.class,label2);
figure(3);
roc = rocmetrics(part2_test.class,scores,part2_model_2.ClassNames);
plot(roc);
title('ROC curve for Naive Bayes Classifier');
cvmodel = crossval(part2_model_2);
missclass = kfoldLoss(cvmodel);
disp(missclass);
