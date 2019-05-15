outputFolder = fullfile('caltech101');
rootFolder=fullfile(outputFolder, '101_ObjectCategories');
%loading the entire path of the folder
path = char('E:\study\PROJECTS\IMAGE PROCESSING\OBJECT-CLASSIFICATION\caltech101\101_ObjectCategories')
categories = {'car_side', 'cougar_body', 'joshua_tree'};
imds = imageDatastore(fullfile(path,categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});

imds = splitEachLabel(imds,minSetCount,'randomize');

% car_side = find(imds.Labels == 'car_side',1);
% cougar_body = find(imds.Labels == 'cougar_body',1);
% joshua_tree = find(imds.Labels == 'joshua_tree',1);
%visualising the data loaded
% figure
% subplot(2,2,1);
% imshow(readimage(imds,car_side));
% subplot(2,2,2);
% imshow(readimage(imds,cougar_body));
% subplot(2,2,3);
% imshow(readimage(imds,joshua_tree));
%loading the neural network
net = resnet50();
%visualising the neural network
% figure
% plot(net)
% title('Resnet-50 Architecture')
%resizing the network to a suitable size
% set(gca,'YLim',[150 170]);
%examining the network input properties
net.Layers(1);
%examining the network output properties
net.Layers(end);
%or we can write
numel(net.Layers(end).ClassNames);
%splitting the data into training and test set
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');
%extracting the inputsize for the network as we have examined
inputimagesize = net.Layers(1).InputSize;
%resizing and converting on a copy of the images
augmentedTrainingSet = augmentedImageDatastore(inputimagesize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(inputimagesize, testSet, 'ColorPreprocessing', 'gray2rgb');
%examining the weights for image recognition
w1 = net.Layers(2).Weights;
w1 = mat2gray(w1);
% figure
% montage(w1)
% title('First CNN layer weight')
featureLayer = 'fc1000';
trainingFeatures = activations(net,...
    augmentedTrainingSet, featureLayer, 'MiniBatchSize',32, 'OutputAs', 'columns');
trainLabels = trainingSet.Labels;
classifier = fitcecoc(trainingFeatures,...
    trainLabels, 'Learner', 'Linear', 'Coding', 'onevsall','ObservationsIn', 'columns');
testFeatures = activations(net,...
    augmentedTestSet, featureLayer, 'MiniBatchSize',32, 'OutputAs', 'columns');

predictLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

testsetLabels = testSet.Labels;
confmat = confusionmat(testsetLabels, predictLabels);

compmat = bsxfun(@rdivide, confmat, sum(confmat,2));
mean(diag(compmat));

newImage = imread(fullfile('test.jpg'));
augmentednewimage = augmentedImageDatastore(inputimagesize, newImage, 'ColorPreprocessing', 'gray2rgb');

newImageFeatures = activations(net,...
   augmentednewimage, featureLayer, 'MiniBatchSize',32, 'OutputAs', 'columns');

thelabel = predict(classifier, newImageFeatures, 'ObservationsIn', 'columns');
sprintf('The loaded image belongs to %s class' ,thelabel)

