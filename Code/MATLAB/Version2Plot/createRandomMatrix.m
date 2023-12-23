% Create random sparseMatrix of observation for matrixCompletion.
function sparseData = createRandomMatrix( rowN, colN, sparsity )
    trueRank = ceil( rowN / 100 );
    U = randn( rowN, trueRank );
    V = randn( colN, trueRank );
    N = randn( rowN, colN );
    omega = ones( rowN, colN );
    omega( 1:ceil( ( 1 - sparsity ) * rowN * colN ) ) = 0;
    omega( randperm( numel( omega ) ) ) = omega;
    sparseData = sparse( omega .* ( U * V' + 0.1 * N ) );
end