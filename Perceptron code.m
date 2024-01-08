% Perceptron for AND function with bipolar inputs and targets

clear;
clc;

X = [1 1 -1 -1; 1 -1 1 -1];
t = [1 -1 -1 -1];

w = [0 0];
b = 0;

alpha = input('Enter learning rate: ');
theta = input('Enter threshold value: ');

con = 1;
epoch = 0;

while con
    con = 0;
    
    for i = 1:4
        yin = b + X(1, i) * w(1) + X(2, i) * w(2);
        
        if yin > theta
            y = 1;
        elseif yin >= -theta && yin <= theta
            y = 0;
        elseif yin < -theta
            y = -1;
        end
        
        if y ~= t(i)
            con = 1;
            for j = 1:2
                w(j) = w(j) + alpha * t(i) * X(j, i);
            end
            b = b + alpha * t(i);
        end
    end
    
    epoch = epoch + 1;
end

disp('Perceptron for AND function:');
disp('Final weight matrix:');
disp(w);
disp('Final bias:');
disp(b);
disp(['Number of epochs: ', num2str(epoch)]);
