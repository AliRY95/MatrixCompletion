%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matrix Completion Function Using DualGC.
%
% Inputs:
% 1. observedData: Known data. Binary mask is created based on 0 elements
% of observedData.
% 2. tau: Nuclear norm constraint.
% 3. maxNumSVs: Maximum No. of singular values to retrieve.
%
% Outputs:
% [U, S, V]: SVD decomposition of the completed matrix.
% rho: Optimality gap at each iteration.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ U, S, V, rho ] = completeMatrix( observedData, tau, maxNumSVs, tauStar )
    maxIter = 10000;
    epsilon = 1.e-7;

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
    [ U, Sigma, V ] = svds( Z, maxNumSVs );
    % multiplictyLargestSV = size( find( abs( diag( Sigma ) - Sigma( 1 ) ) < 10 ), 1 );
    multiplictyLargestSV = maxNumSVs;
    U = U( :, 1:multiplictyLargestSV );
    V = V( :, 1:multiplictyLargestSV );
    
    % Solving LS
    % Objective function + reshaping for better performance in optimization
    indices = find( observedData );
    observedData_obj = observedData( indices );
    UVT = zeros( size( indices, 1 ), multiplictyLargestSV );
    for i = 1:multiplictyLargestSV
        temp = U( :, i ) * V( :, i )';
        UVT( :, i ) = temp( indices );
    end
    
    
    % Linear constraint trace(S) < tau
    % If you want them unsorted
    % A = ones( 1, multiplictyLargestSV );
    % b = tau;
    % If you want them sorted
    A = diag( +1 * ones( 1, multiplictyLargestSV) ) + ...
        diag( -1 * ones( 1, multiplictyLargestSV - 1 ), -1 );
    A( 1, : ) = 1;
    b = zeros( multiplictyLargestSV, 1 );
    b( 1 ) = tauStar;
    % No equality constraints
    Aeq = [];
    beq = [];
    % Lower and upper bounds of the variables
    lb = zeros( multiplictyLargestSV, 1 );
    ub = [];
    % Initial guess
    S0 = zeros( multiplictyLargestSV, 1 );
    S0( 1 ) = tau;
    % Solve using lsqlin
    options = optimoptions( 'lsqlin', ...
                            'Algorithm', 'interior-point', ...
                            'Display', 'iter' );
    S = lsqlin( UVT, observedData_obj, A, b, Aeq, beq, lb, ub, S0, options );
    % Solve using fmincon
    % Nonlinear constraint
    % nonlcon = [];
    % fun = @(S) ( ( 0.5 * norm( UVT * S - observedData_obj ) ^ 2 ) );
    % options = optimoptions( 'fmincon', ...
    %                         'Display', 'iter-detailed', ...
    %                         'Algorithm', 'sqp', ...
    %                         'UseParallel', true );
    % S = fmincon( fun, S0, A, b, Aeq, beq, lb, ub, nonlcon, options );

    % Reshaping for the output
    rho = rho( rho ~= 0 );
    S = diag( S );
end
