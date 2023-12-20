function rank = estimateRank( U, S, V )
    froNorm = norm( U * S * V', 'fro' );
    truncatedS = zeros( 'like', S );
    for rank = 1:size( diag( S ) )
        truncatedS( rank, rank ) = S( rank, rank );
        estimatedMatrix = U * truncatedS * V';
        if norm( estimatedMatrix, 'fro' ) >= 0.9 * froNorm
            break
        end
    end
end