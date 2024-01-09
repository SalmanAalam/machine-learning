% BPN_XOR.m

function BPN_XOR()

    % Initialize weights and biases
    V = [0.197 0.3191 -0.1448 0.3394; 0.3099 0.1904 -0.0347 -0.4861];
    B1 = [-0.3378 0.2771 0.2859 -0.3329];
    B2 = 0.1401;
    W = [0.4919; -0.2913; -0.3979; 0.3581];

    % Learning parameters
    Alpha = 0.02;
    Mf = 0.9;

    % Input and target data
    X = [1 1 0 0; 1 0 1 0];
    T = [0 1 1 0];

    % Initialize momentum-related variables
    V1 = zeros(size(V));
    B1_momentum = zeros(size(B1));
    W1 = zeros(size(W));
    B2_momentum = 0;

    % Training loop
    Con = 1;
    Epoch = 0;

    % Define nested functions
    function y = binsig(x)
        y = 1 / (1 + exp(-x));
    end

    function y = binsig1(x)
        y = binsig(x) * (1 - binsig(x));
    end

    while Con
        E = 0;

        for I = 1:4
            % Feedforward
            Zin = B1;
            for j = 1:4
                Zin(j) = Zin(j) + X(1, I) * V(1, j) + X(2, I) * V(2, j);
                Z(j) = binsig(Zin(j));
            end

            Yin = B2 + Z * W;
            Y = binsig(Yin);

            % Backpropagation of error
            Delk = (T(I) - Y) * binsig1(Yin);
            Delw = Alpha * Delk * Z' + Mf * (W - W1);
            Delb2 = Alpha * Delk;
            Delinj = Delk * W;
            
            for j = 1:4
                Delj(j) = Delinj(j) * binsig1(Zin(j));
            end

            for j = 1:4
                for i = 1:2
                    Delv(i, j) = Alpha * Delj(j) * X(i, I) + Mf * (V(i, j) - V1(i, j));
                end
            end

            Delb1 = Alpha * Delj;

            % Weight updation
            W1 = W;
            V1 = V;
            W = W + Delw;
            B2 = B2 + Delb2;
            V = V + Delv;
            B1 = B1 + Delb1;

            E = E + (T(I) - Y)^2;
        end

        if E < 0.005
            Con = 0;
        end
clc

clc
clc


        Epoch = Epoch + 1;
    end

    % Display results
    disp('BPN for XOR function with binary input and output');
    disp('Total epochs performed:');
    disp(Epoch);
    disp('Final error:');
    disp(E);
    disp('Final weight matrix and bias:');
    disp('V:');
    disp(V);
    disp('B1:');
    disp(B1);
    disp('W:');
    disp(W);
    disp('B2:');
    disp(B2);

end
