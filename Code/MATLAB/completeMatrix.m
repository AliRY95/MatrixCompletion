function [ U, S, V, rho ] = completeMatrix( observedData, tau, maxNumSVs )
    maxIter = 1000;
    epsilon = 1.e-12;

    [ rowN, colN ] = size( observedData );
    mask = sparse( observedData ~= 0 );
    R = observedData;
    Q = sparse( rowN, colN );
    rho = zeros( maxIter, 1 ); % optimality gap
    for k = 1:maxIter
        Z = mask .* R;
        [ u, ~, v ] = svds( Z, 1 );
        delR = mask .* ( tau * u * v' ) - Q;
        rho( k ) = sum( dot( delR, R ) );
        if ( abs( rho( k ) ) < epsilon )
            disp( "aaaaaa" );
            break;
        end
        theta = min( 1, rho( k ) / norm( delR, 'fro' ) ^ 2 );
        R = R - theta * delR;
        Q = Q + theta * delR;
    end
    [ U, Sigma, V ] = svds( Z, maxNumSVs );
    % multiplictyLargestSV = size( find( abs( diag( Sigma ) - Sigma( 1 ) ) < 1. ), 1 );
    multiplictyLargestSV = maxNumSVs;
    U = U( :, 1:multiplictyLargestSV );
    V = V( :, 1:multiplictyLargestSV );

    % Solving SDP
    options = optimoptions( 'fmincon', ...
                            'Display', ...
                            'iter-detailed', ...
                            'Algorithm', ...
                            'sqp' );
    fun = @(S) ( 0.5 * ...
                 norm( full ( mask .* ( U * diag( S ) * V' - observedData ) ) ) ...
                 ^ 2 );
    % Linear constraint trace(S) < tau
    A = ones( 1, multiplictyLargestSV );
    b = tau;
    % No equality constraints
    Aeq = [];
    beq = [];
    % Lower and upper bounds of the variables
    lb = zeros( multiplictyLargestSV, 1 );
    ub = tau * ones( multiplictyLargestSV, 1 );
    % Nonlinear constraint
    nonlcon = [];
    % Initial guess
    S0 = zeros( multiplictyLargestSV, 1 );
    % Solve
    S = fmincon( fun, S0, A, b, Aeq, beq, lb, ub, nonlcon, options );

    % Reshaping for the output
    rho = rho( find( rho ) );
    S = diag( S );
end
