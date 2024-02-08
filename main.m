%% Split the data to training and testing
%Create the indices for the amounts to go to train and test
[idxTrain,idxTest] = trainingPartitions(length(Data), [0.5 0.5]);
%split to training data
DataTrain = Data(idxTrain);
LabelTrain = Labels(idxTrain);
%split to testing data
DataTest = Data(idxTest);
LabelTest = Labels(idxTest);

options = trainingOptions('sgdm', ...
    'MaxEpochs',20,...
    'InitialLearnRate',1e-4, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(DataTrain,LabelTrain,layers_1,options);