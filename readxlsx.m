function [train_data test_data] = readxlsx(alldata)
    data = readtable(alldata);
    survived_train = table2array(data(1:891,2));
    survived_test = table2array(data(892:end,2));
    PClass_train = table2array(data(1:891,3));
    PClass_test = table2array(data(892:end,3));
    Age_train = table2array(data(1:891,6));
    Age_test = table2array(data(892:end,6));
    Fare_train = table2array(data(1:891,10));
    Fare_test = table2array(data(892:end,10));

    features_train = [Age_train Fare_train PClass_train survived_train];
    idx = isnan(features_train(:,1));
    features_train(idx,:) = [];
    idx = isoutlier(features_train(:,2));
    features_train(idx,:) = [];
    train_data.features = features_train(:,1:3);
    class_train = features_train(:,4);
    class_train = num2cell(class_train);
    for i = 1:length(class_train)
        if class_train{i} == 0
            class_train{i} = 'Died';
        else
            class_train{i} = 'Survived';
        end
    end
    train_data.class = class_train;

    features_test = [Age_test Fare_test PClass_test survived_test];
    idx = isnan(features_test(:,1));
    features_test(idx,:) = [];
    idx = isoutlier(features_test(:,2));
    features_test(idx,:) = [];
    test_data.features = features_test(:,1:3);
    class_test = features_test(:,4);
    class_test = num2cell(class_test);
    for i = 1:length(class_test)
        if class_test{i} == 0
            class_test{i} = 'Died';
        else
            class_test{i} = 'Survived';
        end
    end
    test_data.class = class_test;
end

