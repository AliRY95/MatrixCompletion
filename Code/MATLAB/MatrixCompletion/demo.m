clearvars;
clc;
clf;

lamnbda_tol = 10;
tol = 1e-8;
N = 500;
[ observedData, testData ] = readMovieLensData( "../ml-100k", 1 );
observedData = readImage( "../Images/torab.jpg", 5, 0.1 );
[predictedData, ier] = MatrixCompletion( full( observedData ), N, lamnbda_tol, tol, 0 );

% fprintf('\n Corrupted matrix nuclear norm (initial): %g \n',sum(svd(A.*B)));
% fprintf('Restored matrix nuclear norm (final): %g \n',sum(svd(CompletedMat)));
% Diff_sq = abs(CompletedMat-A).^2;
% fprintf('MSE on known entries: %g \n',sqrt(sum2(Diff_sq.*B)/sum(B(:)) ));

figure( 1 );
subplot( 1, 2, 1 );
imshow( full( observedData ) );
subplot( 1, 2, 2 );
imshow( full( predictedData ) );