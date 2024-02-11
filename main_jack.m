%% Assignment 1 - JD Herlehy || Jacky Li
%% Feb.10.2024
%% Compare accuracy difference when giving a MLP NN SIFT Features vs  Raw images

% Using the RealWaste data set, from this using the metal and plastic
% classes

%[RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning](https://www.mdpi.com/2078-2489/14/12/633)
%https://archive.ics.uci.edu/dataset/908/realwaste

<<<<<<< HEAD
options = trainingOptions('sgdm', ...
    'MaxEpochs',300,...
    'InitialLearnRate',2e-3, ...
    'Verbose',false, ...
    'Plots','training-progress');
=======
%% Define the location of the dataset
datasetPath = ('C:\Users\JD Herlehy\OneDrive - The University of Western Ontario\Forth Year\4436 Adv Img Proc\Ass 1\WasteBinary');
imageDataset = imageDatastore(datasetPath, "IncludeSubfolders",true, "LabelSource","foldernames");
>>>>>>> bee9ec2af4572bc5995bbaeb201fc7cfba73a1d7

%Get the SIFT data points
[Data] = SIFTFeatureExtraction(imageDataset)
%Train and evaluate network with SFIT data
[AccuracyTrain, AccuracyTest] = SIFTNetwork(Data, imageDataset)
%Train and evaluate network with raw image data
[AccuracyTrainRaw, AccuracyTestRaw] = RawNetwork(imageDataset)
%Display the relevent information
DisplayInfo(AccuracyTrain, AccuracyTest, AccuracyTrainRaw, AccuracyTestRaw)
