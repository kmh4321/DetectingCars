%% R-CNN (Regions with Convolutional Neural Networks)

%% Download CIFAR-10 Image Data
% Download the CIFAR-10 data set [3]. This dataset contains 50,000 training
% images that will be used to train a CNN.

% Download CIFAR-10 data to a temporary directory
cifar10Data = tempdir;
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
helperCIFAR10Data.download(url, cifar10Data);

% Load the CIFAR-10 training and test data.
[trainingImages, trainingLabels, testImages, testLabels] = helperCIFAR10Data.load(cifar10Data);
%% Create A Convolutional Neural Network (CNN)

% Create the image input layer for 32x32x3 CIFAR-10 images
[height, width, numChannels, ~] = size(trainingImages);
imageSize = [height width numChannels];
inputLayer = imageInputLayer(imageSize);

%%
% Convolutional layer parameters
filterSize = [5 5];
numFilters = 32;

middleLayers = [
    
convolution2dLayer(filterSize, numFilters, 'Padding', 2)
reluLayer()
maxPooling2dLayer(3, 'Stride', 2)

convolution2dLayer(filterSize, numFilters, 'Padding', 2)
reluLayer()
maxPooling2dLayer(3, 'Stride',2)

convolution2dLayer(filterSize, 2 * numFilters, 'Padding', 2)
reluLayer()
maxPooling2dLayer(3, 'Stride',2)
];

%%
finalLayers = [
    
fullyConnectedLayer(64)

reluLayer

fullyConnectedLayer(numImageCategories)

softmaxLayer
classificationLayer
];

%% Combine the input, middle, and final layers.
layers = [
    inputLayer
    middleLayers
    finalLayers
    ];

%%
% Initialize the first convolutional layer weights using normally
% distributed random numbers with standard deviation of 0.0001.

layers(2).Weights = 0.0001 * randn([filterSize numChannels numFilters]);

%% Train CNN Using CIFAR-10 Data
% Set the network training options
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 128, ...
    'Verbose', true);

cifar10Net = trainNetwork(trainingImages, trainingLabels, layers, opts);

%% Train R-CNN using our bounding boxes data to learn region proposals
options = trainingOptions('sgdm', ...
        'MiniBatchSize', 300, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 600, ...
        'MaxEpochs', 10, ...
        'Verbose', true,...
        'ExecutionEnvironment', 'auto',...
        'CheckpointPath','C:\Users\kmh43\Documents\Fall 2017\EECS 498\checkpoint_rcnn');

% Train an R-CNN object detector. This will take several minutes.
% Call attached "extractLearningData.m" function to generate learnData.
rcnn = trainRCNNObjectDetector(learnData, cifar10Net , options, ...
'NegativeOverlapRange', [0.1 0.35], 'PositiveOverlapRange',[0.5 1], 'NumStrongestRegions',2000);

%% Testing the R-CNN on test data

files = dir('rob599_dataset_deploy/test/*/*_image.jpg');
N = zeros(numel(files),1);
bboxes = cell(numel(files),1);
score = cell(numel(files),1);
guid=cell(numel(files),1);

for coutnFile =1:1:numel(files)
    tic
    disp(['Current Progress = ' num2str(coutnFile) '/' num2str(numel(files))]);
    snapshot = [files(coutnFile).folder, '\', files(coutnFile).name];
    thr = 0.97;
    %Resize and turn into grayscale
    testImage = rgb2gray(imread(snapshot));
    testImage=imresize(testImage, 0.5);
    
    %get fileName
    guid{coutnFile,1}=[files(coutnFile).folder(find(files(1).folder=='\',1,'last')+1:end)...
        '/' files(coutnFile).name(1:find(files(coutnFile).name=='_')-1)];
    
    %Detect cars in image
    [bboxes{coutnFile,1}, score{coutnFile,1}, ~] = detect(rcnn, testImage,'MiniBatchSize', 128);
    
    N(coutnFile) = sum(score{coutnFile,1}(:,1) > thr);    
    toc
end
result = table(bboxes,score);


%% Extracting test results into a file
output=table(guid,N,'VariableNames',{'guid_image','N'});
txtName=['Results_t' num2str(thr*100)];
writetable(output,txtName);
fileID = fopen([txtName '.txt'],'r+');
fseek(fileID,0,'cof');
newTitle='guid/image';
fprintf(fileID, newTitle);
fclose(fileID);