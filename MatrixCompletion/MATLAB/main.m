clearvars;
clc;
clf;

%% Inputs
testAvailable = 0;
tau = 1500;
ell = 50;

%% Choose one for the observed data
% rowN = 100;
% colN = 100;
% observedData = createRandomMatrix( rowN, colN, 0.1 );

[ observedData, testData ] = readMovieLensData( "ml-100k", 1 );
testAvailable = 1;

% observedData = readImage( "Images/MPF.jpg", 20, 0.5 );

%% Dual conditional gradient for matrix completion
tic;
[ U, S, V, rho ] = completeMatrix( observedData, tau, ell, 100 * tau );
toc;

%% Rank estimation as proposed by MPF
estimatedRank = estimateRank( U, S, V );

%% Post-processing
predictedData = U * S * V';

% Evaluation on training set
indices = find( observedData );
trainSize = size( observedData, 1 ) * size( observedData, 2 );
trainSparsity = size( indices, 1 ) / trainSize;
objFunction = norm( predictedData( indices ) - observedData( indices ), 'fro' );
trainError_RMSE = rmse( predictedData( indices ), observedData( indices ) );
trainError_NMAE = sum( abs( predictedData( indices ) - observedData( indices ) ) ) / size( indices, 1 ) / range( observedData( indices ) );

% Evaluation on test set
if ( testAvailable )
    indices = find( testData );
    testSparsity = size( indices, 1 ) / trainSize;
    testError_RMSE = rmse( predictedData( indices ), testData( indices ) );
    testError_NMAE = sum( abs( predictedData( indices ) - testData( indices ) ) ) / size( indices, 1 ) / range( testData( indices ) );
end

%% Plots
figure( 1 );
semilogy( abs( rho ) );
xlabel( "Iteration" );
ylabel( "Optimality gap" );

figure( 2 );
subplot( 1, 2, 1 );
spy( observedData );
subplot( 1, 2, 2 );
spy( predictedData );

figure( 3 );
subplot( 1, 2, 1 );
imshow( full( observedData ) );
subplot( 1, 2, 2 );
imshow( full( predictedData ) );