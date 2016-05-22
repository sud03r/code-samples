%===============================================
% The Main function: SfM2
%===============================================

function SfM2(img1, img2, intrinsicParamsFile, outputFile)
	I1 = imread(img1);
	I2 = imread(img2);
	
	% Load the camera intrinsic Parameters
	disp('Loading Intrinsics...');
	params = importdata(intrinsicParamsFile);
	K =  params(1:3, 1:3);

	% Extract sparse features
	[matchedPoints1, matchedPoints2] = extractFeatureMatchPoints(I1, I2, false);

	disp('Computing Projection Matrices...');
	[P1, P2] = estimateProjectionMatrices(matchedPoints1, matchedPoints2, K);
	
	% Extract dense features
	disp('Computing Dense features...');
	[matchedPoints1, matchedPoints2] = extractFeatureMatchPoints(I1, I2, true);
	
	disp('Computing 3D points...');
	points3D = triangulate(matchedPoints1, matchedPoints2, P1', P2');

	% Extract color from one of the images.
	ptColors = extractColor(I1, matchedPoints1);
	
	% Create the point cloud
	ptCloud = pointCloud(points3D, 'Color', ptColors);
	pcwrite(ptCloud, outputFile);
	disp('Done!');
	
	% Visualize if needed!
	displayPC(outputFile);
end

%====================================================
% Matlab function to extract the features that match
% in two images. We use SURF features for now..
%====================================================
function [mPts1, mPts2] = extractFeatureMatchPoints(I1, I2, isDense)
	visualize = false;
	th2 = 0.05;
	if isDense
		th2 = 0.0045;
	end
	
	% Detect feature points
	iPts1 = detectMinEigenFeatures(rgb2gray(I1), 'MinQuality', th2);

	% Create the point tracker
	tracker = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 5);

	% Initialize the point tracker
	iPts1 = iPts1.Location;
	initialize(tracker, iPts1, I1);

	% Track the points
	[iPts2, vIdx] = step(tracker, I2);
	mPts1 = iPts1(vIdx, :);
	mPts2 = iPts2(vIdx, :);

	if visualize
		% Visualize correspondences
		figure
		showMatchedFeatures(I1, I2, mPts1, mPts2);
	end
end

%=================================================
% Extract the colors from I for matchedPoints
%=================================================
function color = extractColor(I, matchedPoints)
	% Get the color of each reconstructed point
	numPixels = size(I, 1) * size(I, 2);
	allColors = reshape(I, [numPixels, 3]);
	colorIdx = sub2ind([size(I, 1), size(I, 2)],... 
					round(matchedPoints(:,2)), round(matchedPoints(:, 1)));
	color = allColors(colorIdx, :);
end

%===========================================
% Display the 3D point cloud!
%===========================================
function displayPC(fName)
	ptCloud = pcread(fName);
	
	% Visualize the camera locations and orientations
	cameraSize = 0.3;
	figure
	plotCamera('Size', cameraSize, 'Color', 'r', 'Label', '1', 'Opacity', 0);
	hold on
	grid on

	% Visualize the point cloud
	pcshow(ptCloud, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
					'MarkerSize', 45);

	% Rotate and zoom the plot
	camorbit(0, -30);
	camzoom(1.5);
end
