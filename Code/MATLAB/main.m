clearvars;
clc;
clf;

%% Choose one for the observed data
rowN = 100;
colN = 100;
% observedData = createRandomMatrix( rowN, colN, 0.5 );
% [ observedData, testData ] = readMovieLensData( "ml-100k", 1 );
% observedData = readTorabData( "TorabData" );
observedData = readImage( "Images/torab.jpg", 5, 0.8 );

testAvailable = 0;
%% Dual conditional gradient for matrix completion
[ U, S, V, rho ] = completeMatrix( observedData, 4000, 5 );

%% Rank estimation as proposed by MPF
estimatedRank = estimateRank( U, S, V );

%% Post-processing
predictedData = U * S * V';
% rounding to have more meaningful data
% predictedData = scaleData( observedData, predictedData );

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
subplot( 1, 3, 1 );
spy( observedData );
subplot( 1, 3, 2 );
spy( predictedData );
% subplot( 1, 3, 3 );
% spy( testData );

figure( 2 );
subplot( 1, 3, 1 );
plot( observedData );
subplot( 1, 3, 2 );
plot( predictedData );

figure( 3 );
semilogy( abs( rho ) )

figure( 4 );
subplot( 1, 2, 1 );
imshow( full( observedData ) );
subplot( 1, 2, 2 );
imshow( full( predictedData ) );
% subplot( 1, 3, 3 );

