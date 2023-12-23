clearvars;
clc;
close all;

%% Choose one for the observed data
imageRank = 20;
originalData = readImage( "Images/MPF.jpg", imageRank, 1. );
observedData = readImage( "Images/MPF.jpg", imageRank, 0.6 );
tau = 90.8;
%% Dual conditional gradient for matrix completion
[ Z, rho ] = completeMatrix( observedData, tau );

ell = 20;
[ U, S, V ] = solveLS( observedData, Z, 1000*tau, ell );
predictedData = U * S * V';
indices = find( observedData );
traceS = trace(S);
NMAE = sum( abs( predictedData( indices ) - observedData( indices ) ) ) / size( indices, 1 ) / range( observedData( indices ) );

figure( 1 );
subplot( 1, 3, 1 );
imshow( full( originalData ) );
subplot( 1, 3, 2 );
imshow( full( observedData ) );
subplot( 1, 3, 3 );
imshow( full( predictedData ) );

figure( 2 );
semilogy( abs( rho ) )