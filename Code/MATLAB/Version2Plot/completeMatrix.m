function [ Z, rho ] = completeMatrix( observedData, tau )
    maxIter = 10000;
    epsilon = 1.e-15;

    [ rowN, colN ] = size( observedData );
    mask = sparse( observedData ~= 0 );
    R = observedData;
    Q = sparse( rowN, colN );
    rho = zeros( maxIter, 1 ); % optimality gap
    for k = 1:maxIter        
        Z = mask .* R;
        [ u, ~, v ] = svds( Z, 1 );
        delR = mask .* ( tau * u * v' ) - Q;
        rho( k ) = dot( delR(:), R(:) );
        if ( abs( rho( k ) ) < epsilon )
            disp( "Converged in " + num2str( k ) + ...
                " iterations with optimality gap " + num2str( rho( k ) ) + ...
                "! Doing SDP now!" );
            break;
        end
        if rem( k, 100 ) == 0
            disp( "Iteration " + num2str( k ) + ...
                " passed with optimality gap " + num2str( rho(k) ) + "." );
        end
        theta = min( 1, rho( k ) / norm( delR, 'fro' ) ^ 2 );
        R = R - theta * delR;
        Q = Q + theta * delR;
    end
    rho = rho( rho ~= 0 );
end
