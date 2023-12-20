clearvars;
clc;
clf;

%% Choose one for the observed data
rowN = 1000;
colN = 1000;
observedData = createRandomMatrix( rowN, colN, 0.1 );
% [ observedData, testData ] = readMovieLensData( "ml-100k", 1 );
% observedData = readTorabData( "TorabData" );

testAvailable = 0;
%% Dual conditional gradient for matrix completion
[ U, S, V, rho ] = completeMatrix( observedData, 100, 1 );

%% Rank estimation as proposed by MJF
estimatedRank = estimateRank( U, S, V );

%% Post-processing
constructedData = U * S * V';
% rounding to have more meaningful data
% constructedData = round( constructedData );

% Evaluation on training set
indices = find( observedData );
trainSize = size( observedData, 1 ) * size( observedData, 2 );
trainSparsity = size( indices, 1 ) / trainSize;
trainError_RMSE = rmse( constructedData( indices ), observedData( indices ) );
trainError_NMAE = sum( abs( constructedData( indices ) - observedData( indices ) ) ) / size( indices, 1 ) / mean( observedData( indices ) );
% lossFunction = 0.5 * norm( constructedData - observationData, 'fro' );

% Evaluation on test set
if ( testAvailable )
    indices = find( testData );
    testSparsity = size( indices, 1 ) / trainSize;
    testError_RMSE = rmse( constructedData( indices ), testData( indices ) );
    testError_NMAE = sum( abs( constructedData( indices ) - testData( indices ) ) ) / size( indices, 1 ) ./ mean( testData( indices ), 1 );
end

%% Plots
figure( 1 );
subplot( 1, 3, 1 );
spy( observedData );
subplot( 1, 3, 2 );
spy( constructedData );
subplot( 1, 3, 3 );
% spy( testData );


figure( 2 );
subplot( 1, 3, 1 );
plot( observedData );
subplot( 1, 3, 2 );
plot( constructedData );

figure( 3 );
semilogy( abs( rho ) )



