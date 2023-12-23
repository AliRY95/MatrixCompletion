function rank = estimateRank( U, S, V )
    froNorm = norm( U * S * V', 'fro' );
    sizeS = size( S );
    truncatedS = zeros( sizeS );
    for rank = 1:size( diag( S ) )
        truncatedS( rank, rank ) = S( rank, rank );
        estimatedMatrix = U * truncatedS * V';
        if norm( estimatedMatrix, 'fro' ) >= 0.9 * froNorm
            break
        end
    end
end