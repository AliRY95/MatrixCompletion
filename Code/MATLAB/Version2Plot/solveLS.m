function [U, S, V] = solveLS( observedData, Z, tau, ell )
    [ U, Sigma, V ] = svds( Z, ell );
    % multiplictyLargestSV = size( find( abs( diag( Sigma ) - Sigma( 1 ) ) < 10 ), 1 );
    multiplictyLargestSV = ell;
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
    A = ones( 1, multiplictyLargestSV );
    b = tau;
    % If you want them sorted
    % A = diag( +1 * ones( 1, multiplictyLargestSV) ) + ...
    %     diag( -1 * ones( 1, multiplictyLargestSV - 1 ), -1 );
    % A( 1, : ) = 1;
    % b = zeros( multiplictyLargestSV, 1 );
    % b( 1 ) = tau;
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
    S = diag( S );
end