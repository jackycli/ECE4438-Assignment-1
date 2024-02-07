%% Define the location of the dataset
datasetPath = ('C:\Users\JD Herlehy\OneDrive - The University of Western Ontario\Forth Year\4436 Adv Img Proc\Ass 1\WasteBinary');
imageDataset = imageDatastore(datasetPath, "IncludeSubfolders",true, "LabelSource","foldernames");

%% Get dataset size
datasetSize = size(imageDataset.Files);

%% Organize Data
%for i = 1:datasetSize(1)
for i = 1:4
    
    %Convert to grayscale
    image = rgb2gray(imread(imageDataset.Files{i}));
    %Get strongest SIFT features
    imgSIFTs = detectSIFTFeatures(image);
    strongest = imgSIFTs.selectStrongest(200);
    %Put in custom table
    Data(i, :) = {strongest.Metric};
    
    %if not as much points, pad with 0
    if(length(Data{i}) < 200)
        pad = zeros(200 - length(Data{i}), 1);
        Data{i} = vertcat(Data{i}, pad);
    end

    Data{i} = Data{i}';

end