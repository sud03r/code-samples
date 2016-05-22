%=================================================================
% Estimate the camera projection matrices P1, P2
%=================================================================
function [P1, P2] = estimateProjectionMatrices(imgLoc1, imgLoc2, K)
	pointCnt = size(imgLoc1, 1);
    assert(size(imgLoc2, 1) == pointCnt);
	[P1, P2] = computeRansacE(imgLoc1, imgLoc2, K);
end

%=============================================================
% Compute the Essential Matrix using RANSAC
%=============================================================
function [P1, P2] = computeRansacE(imgLoc1, imgLoc2, K)
	% Add the solver to path.
	addpath(genpath('fivepointSolver'));

	K_inv = inv(K);
	numPoints = size(imgLoc1, 1);
	bestE = eye(3);
	minSampsonError = inf;

	% Try twice as many times as the number of times.
	maxTrials = round(numPoints * 2);
	msg = horzcat('Will be Running for ' , num2str(maxTrials), ' iterations.');
	disp(msg);
	
	for i = 1:maxTrials	
		randomIdxs = randperm(numPoints, 5);
		Q1 = imgLoc1(randomIdxs, :);
		Q2 = imgLoc2(randomIdxs, :);
		
		% Make homogenous
		Q1 = K_inv * [Q1, ones(5, 1)]';
		Q2 = K_inv* [Q2, ones(5, 1)]';

		Evec = calibrated_fivepoint(double(Q2), double(Q1));
		for j=1:size(Evec,2)
			E = reshape(Evec(:,j), 3, 3);
			err = computeEstimationError(imgLoc1, imgLoc2, E, K_inv);
			if minSampsonError > err
				minSampsonError = err;
				bestE = E;
			end
		end
		
		if mod(i, 500) == 0
			msg = horzcat('Completed ' , num2str(i), ' iterations');
			disp(msg);
		end
	end
	
	msg = horzcat('After ' , num2str(maxTrials), ' iterations, we settle for an error :' , num2str(minSampsonError));
	disp(msg);
	Rt = splitFourWays(bestE);
	[P1, P2] = findBest(Rt, K, imgLoc1, imgLoc2);
end

%====================================================================
% Compute the best projection matrix depending on how many 3D points
% happen to be in front of the camera
%====================================================================
function [P1, P2] = findBest(Rt, K, inliers1, inliers2)

	P1 = K * [eye(3) [0;0;0]];
	validPts = zeros(1, 4);
	for i = 1:4
		% Uncompress R, T
		R = Rt(:, 1:3, i);
		T = Rt(:, 4, i);

		P2 = K * Rt(:, :, i);
		X = triangulate(inliers1, inliers2, P1', P2');
		% Need to find out how many of these points
		% are in front of both cameras
		R_3 = R(3, :)';
		% Camera Center = -R'T
		C = -R'*T;
		pointLoc = (X(:,:) - repmat(C', size(X,1), 1)) * R_3;
		validPts(i) = sum(X(:,3)>0 & pointLoc > 0);
	end
	[~, bIdx] = max(validPts);
	P2 = K * Rt(:, :, bIdx);
end

%====================================================
% Decompose E into possible four R, t 
%====================================================

function Rt = splitFourWays(E)
	% Slides 12SfM : #58
	W = [0 -1 0; 1 0 0; 0 0 1];
	[U, S, V] = svd(E);

	% Two possible rotation matrices
	R1 = U*W*V';
	R2 = U*W'*V';
	% Two possible translation vectors
	t1 = U(:,3);
	t2 = -U(:,3);
	% Check determinant
	if det(R1) < 0
		R1 = -R1;
	end
	
	if det(R2) < 0
		R2 = -R2;
	end
	
	% Total 4 combinations
	Rt(:, :, 1) = [R1 t1];
	Rt(:, :, 2) = [R1 t2];
	Rt(:, :, 3) = [R2 t1];
	Rt(:, :, 4) = [R2 t2];
end


%====================================================
% Sampson's distance to estimate reprojection error
%====================================================
function dv = computeEstimationError(x1, x2, E, K_inv)
	% x1 <-> x2 are matched image points.
	numPts = size(x1, 1);
	% Convert to homogenous coordinates.
	x1h = K_inv * [x1 ones(numPts, 1)]';
	x2h = K_inv * [x2 ones(numPts, 1)]';
	pfp = (x2h' * E)';
	pfp = pfp .* x1h;
	d = sum(pfp, 1) .^ 2;

	epl1 = E * x1h;
	epl2 = E' * x2h;
	d = d ./ (epl1(1,:).^2 + epl1(2,:).^2 + epl2(1,:).^2 + epl2(2,:).^2);
	dv = sum(d);
end


%=============================================================
% Compute the Essential Matrix using Matlab built-in function.
% Note: We don't use it anymore since our RANSAC already works
% so well! [OBSELETE]
%=============================================================
function [P1, P2] = computeEssentialBuiltIn(imgLoc1, imgLoc2, K) 
	% Estimate the fundamental matrix
	[F, inlierIdx] = estimateFundamentalMatrix(imgLoc1, imgLoc2,... 
							'Method', 'MSAC', 'NumTrials', 10000);
	% Compute Essential matrix from F
	% Using this relation.
	E = K' * F * K;
	Rt = splitFourWays(E);
	
	% inlierIdx is a boolean array that identifies which points were used
	inliers1 = imgLoc1(inlierIdx, :);
	inliers2 = imgLoc2(inlierIdx, :);

	% The first camera has parameters K * [I | 0]
	% The second camera is K * [bestR | bestT]
	[P1, P2] = findBest(Rt, K, inliers1, inliers2);
end


