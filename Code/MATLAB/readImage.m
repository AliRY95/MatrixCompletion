function observedData = readImage( pathToImage, desiredRank, sparsity )
    % Reading the image
    image = imread( pathToImage );
    % Grayscale image
    image = rgb2gray( image );
    % Making an array of doubles
    image = im2double( image );
    % Resizing
    image = imresize( image, 0.5 );

    % Changing the rank
    [ U, S, V ] = svd( image );
    image = U( :, 1:desiredRank ) * ... 
            S( 1:desiredRank, 1:desiredRank ) * ...
            V( :, 1:desiredRank )';
    
    % Eliminating sum elements
    imageSize = size( image, 1 ) * size( image, 2 );
    randomId = randperm( imageSize, ceil( ( 1 - sparsity ) * imageSize ) );
    image( randomId ) = 0;

    % Output
    observedData = sparse( image );
end