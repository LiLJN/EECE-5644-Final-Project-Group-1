clc,
clear all,

train_data = readtable('train.csv');
test_data = readtable('test.csv');
survived_train = table2array(train_data(:,2));
PClass_train = table2array(train_data(:,3));
Age_train = table2array(train_data(:,6));
Fare_train = table2array(train_data(:,10));
survived_test = table2array(test_data(:,2));
PClass_test = table2array(test_data(:,3));
Age_test = table2array(test_data(:,6));
Fare_test = table2array(test_data(:,10));

Expense_train = [PClass_train Fare_train];

figure(1);
edge = [0:5:515];
h1 = histogram(Fare_train,edge);
xlabel('Money Spent ($)');
ylabel('People Counts');

figure(2);
uv = unique(PClass_train);
count = [sum(PClass_train(:) == 1);sum(PClass_train(:) == 2);sum(PClass_train(:) == 3)];
bar(uv,count);
xlabel('PClass');
ylabel('People Counts');

Compare1 = [survived_train PClass_train]';
count1 = [sum(Compare1(1,:) == 0 & Compare1(2,:) == 1),sum(Compare1(1,:) == 1 & Compare1(2,:) == 1)];
count2 = [sum(Compare1(1,:) == 0 & Compare1(2,:) == 2),sum(Compare1(1,:) == 1 & Compare1(2,:) == 2)];
count3 = [sum(Compare1(1,:) == 0 & Compare1(2,:) == 3),sum(Compare1(1,:) == 1 & Compare1(2,:) == 3)];
count = [count1;count2;count3];
label = categorical({'First Class','Second Class','Third Class'});
figure(3);
bar(label,count);
legend('Died','Survived');
ylabel('People Count');

Compare2 = [Fare_train survived_train];
[m,n] = size(Fare_train);
Fare_train = [Fare_train ones(m,1)];
beta = zeros(n+1,1);
iteration = 30000;
alpha = 0.01;
for iter = 1:iteration
    z = Fare_train * beta;
    h = 1./(1+exp(-z));
    error = h - survived_train;
    gradient = Fare_train' * error;
    beta = beta - alpha / m * gradient;
end
figure(4);
scatter(Fare_train,survived_train);
hold on,
x = linspace(-200,200,400);
y = 1./(1+exp(-(beta(1).*x+beta(2))));
plot(x,y);
xlim([-200 200]);
xlabel('Money Spent');
ylabel('Died/Survived');
title('Classification using logistic regression');
subtitle('$\beta_1=0.0563$,$\beta_2=-1.4838$,$p(x)=\frac{1}{1+e^{-\beta_1x+\beta_2}}$','Interpreter','latex');
hold off;

% Fare_train = Fare_train(:,1);
% part1_train = [Age_train Fare_train survived_train];
% part1_test = [Age_test Fare_test survived_test];
% idx = isnan(part1_train(:,1));
% part1_train(idx,:) = [];
% idx = isoutlier(part1_train(:,2));
% part1_train(idx,:) = [];
% idx = isnan(part1_test(:,1));
% part1_test(idx,:) = [];
% idx = isoutlier(part1_test(:,2));
% part1_test(idx,:) = [];


% part1_train = [PClass_train Fare_train survived_train];
% part1_survived_train = part1_train(part1_train(:,3)==1,1:2);
% part1_died_train = part1_train(part1_train(:,3)==0,1:2);
% figure(5);
% scatter(part1_survived_train(:,2),part1_survived_train(:,1),'ro');
% hold on,
% scatter(part1_died_train(:,2),part1_died_train(:,1),'bx');
% ylim([0 200]);

% part1_train_features = part1_train(:,1:2);
% part1_train_class = part1_train(:,3);
% part1_train_class = num2cell(part1_train_class);
% for i = 1:length(part1_train_class)
%     if part1_train_class{i} == 0
%         part1_train_class{i} = 'Died';
%     else
%         part1_train_class{i} = 'Survived';
%     end
% end
% part1_model = fitcnb(part1_train_features,part1_train_class,'ClassNames',{'Died','Survived'});
% diedIndex = strcmp(part1_model.ClassNames,'Died');
% estimates = part1_model.DistributionParameters{diedIndex,1};
% e = min(part1_train_features(:,1)):0.01:max(part1_train_features(:,1));
% f = min(part1_train_features(:,2)):0.01:max(part1_train_features(:,2));
% [x1 x2] = meshgrid(e,f);
% x = [x1(:) x2(:)];
% ms = predict(part1_model,x);
% figure(5);
% gscatter(x1(:),x2(:),ms,'cym');
% hold on,
% gscatter(part1_train_features(:,1),part1_train_features(:,2), ...
%     part1_train_class(:),'rg','.',5);
% xlabel('Age');
% ylabel('Money spent');


