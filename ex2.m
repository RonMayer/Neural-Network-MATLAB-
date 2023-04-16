%% uploading the data and rearranging it to vectors
% first import the train file, and change the variables to "x", "y", and
% spiral. to check if it worked, let's plot the points:
data_train=readtable('DATA_TRAIN.csv');
data_train.Properties.VariableNames{'Var1'} = 'x';
data_train.Properties.VariableNames{'Var2'} = 'y';
data_train.Properties.VariableNames{'Var3'} = 'spiral';
index1=data_train.spiral==1;
index2=data_train.spiral==0;
%plot to see if it worked:
figure(1);
scatter(data_train.x(index1),data_train.y(index1));
hold on
scatter(data_train.x(index2),data_train.y(index2));
X1=data_train.x;
X2=data_train.y;
X=[X1 X2];
Y=data_train.spiral;
Data=[X Y];

vidObj = VideoWriter('nn_optimization.avi');

%% Set-up the parameters with Xavier's Initialization (randn in comments)
input_layer_size = 2;
hidden_layer_size = 16;
variance = 2/(input_layer_size + hidden_layer_size); 
stddev = sqrt(variance);
learning_rate=0.12;
initial_weights1=zeros(input_layer_size,hidden_layer_size); %weights for X->HL1
initial_weights2=zeros(hidden_layer_size,hidden_layer_size); %weights for HL1->HL2
initial_weights3=zeros(hidden_layer_size,2); %weights for HL2->FL

for i=1:hidden_layer_size*input_layer_size
      initial_weights1(i) = normrnd(0, stddev); 
      initial_weights3(i) = normrnd(0, stddev);
end
for i=1:hidden_layer_size*hidden_layer_size
    initial_weights2(i) = normrnd(0, stddev);
end

initial_b1=zeros(1,hidden_layer_size);
initial_b2=zeros(1,hidden_layer_size);
initial_b3=zeros(1,2);

% % for initialization with randn (normal standard distribution)uncomment
% initial_weights1=randn(input_layer_size,hidden_layer_size); %weights for X->HL1
% initial_weights2=randn(hidden_layer_size,hidden_layer_size); %weights for HL1->HL2
% initial_weights3=randn(hidden_layer_size,2); %weights for HL2->FL

%% training
W1=initial_weights1;    W2=initial_weights2;    W3=initial_weights3;
B1=initial_b1;    B2=initial_b2;   B3=initial_b3;
error_vec=[];
iter_size=600;

    %for momentum of deltaW(n-1)
dW1_n_1=0;   dW2_n_1=0;   dW3_n_1=0;   dB1_n_1=0;   dB2_n_1=0;   dB3_n_1=0;
alpha=0.12;

    % for regularization
gamma=0.001;

  % for video uncomment
open(vidObj);
fh = figure;

%forward
for iteration=1:iter_size
    
    %randomize data order every step
    Data = Data(randperm(size(Data, 1)), :);
    X = Data(:,1:2);
    Y = Data(:,3);
    
    HL1 = X*W1+B1; %hidden layer value before activation
    HL1 = sin(HL1); %sin activation function
    
    HL2 = HL1*W2+B2; %hidden layer value before activation    
%     % for RELU uncomment
%      HL2 = max(0, HL2);     
    % for tanh uncomment
    HL2 = tanh(HL2); 
    
    FL = HL2*W3+B3; %final layer value
    FL = exp(FL)./sum(exp(FL), 2); %apply softmax 
    
    % now we will calculate the cost. for every sample, the cost will be
    % added to the total cost. the Y vector has values of 0 or 1 so we will
    % add +1 to fit value of 1 or 2.
    data_error = 0;
    for i = 1:size(X,1)
        data_error = data_error + -1*log(FL(i, Y(i)+1));
    end
    data_error = data_error/size(X,1);
    error_vec(end+1) = data_error;
    
        % For making the movie showing decision boundaries uncomment
    if mod(iteration,5) == 0
         visualize_decision_boundaries2(X, Y, W1,B1, W2,B2, W3,B3, fh, vidObj);
    end
    
%backward
    derror_dFL = FL;
    for i = 1:size(X,1)
        derror_dFL(i, Y(i)+1) = derror_dFL(i, Y(i)+1) - 1;
    end
    derror_dFL = derror_dFL/size(X,1); 
    % for ridge regression regularization uncomment. 1st row with L1, second L2: 
%     derror_dFL = derror_dFL+ (gamma/2)*((sum(sum(abs(W3))))^2)/size(X,1);
    derror_dFL = derror_dFL+ (gamma/2)*(sum(sum(W3.*W3)))/size(X,1);
    
    derror_dW3 = HL2'*derror_dFL; 
    derror_dB3 = ones(1, size(X,1))*derror_dFL;
    
    % back propagation through hidden layer 2
    dHL2 = derror_dFL*W3';
    
    % for ridge regression regularization uncomment. 1st row with L1, second L2: 
%     dHL2 = dHL2+ (gamma/2)*((sum(sum(abs(W2))))^2)/size(X,1);
    dHL2 = dHL2+ (gamma/2)*(sum(sum(W2.*W2)))/size(X,1);
    
%     % for RELU uncomment
%     dHL2(HL2 == 0) = 0;
    
    % for tanh uncomment
    dHL2 = dHL2.*(1./(cosh(HL1*W2+B2)).^2);
    
    derror_dW2 = HL1'*dHL2;
    derror_dB2 = ones(1, size(X,1))*dHL2;
    
    % back propagation through hidden layer 1
    dHL1 = dHL2*W2';
    dHL1 = dHL1.*(cos(X*W1+B1));
    
     % for ridge regression regularization uncomment. 1st row with L1, second L2: 
%     dHL1 = dHL1+ (gamma/2)*((sum(sum(abs(W1))))^2)/size(X,1); 
    dHL1 = dHL1+ (gamma/2)*(sum(sum(W1.*W1)))/size(X,1);
    
    derror_dW1 = X'*dHL1;
    derror_dB1 = ones(1, size(X,1))*dHL1;

    %update weights    
    W3 = W3 - (derror_dW3*learning_rate+(alpha.*dW3_n_1));
%     B3 = B3 - (derror_dB3*learning_rate+(alpha.*dB3_n_1));
    W2 = W2 - (derror_dW2*learning_rate+(alpha.*dW2_n_1));
%     B2 = B2 - (derror_dB2*learning_rate+(alpha.*dB2_n_1));
    W1 = W1 - (derror_dW1*learning_rate+(alpha.*dW1_n_1));  
%     B1 = B1 - (derror_dB1*learning_rate+(alpha.*dB1_n_1));
%for a momentum learning method (Q2) uncomment
dW1_n_1=derror_dW1;  dB1_n_1=derror_dB1;  dW2_n_1=derror_dW2;  dB2_n_1=derror_dB2;  dW3_n_1=derror_dW3;  dB3_n_1=derror_dB3;   
end

%% evaluate training accuracy
% find indicies where the max probability for each class occurs
[val,idx] = max(FL, [], 2);
% correct classification is when idx matches
num_correct_classifications = sum(idx == Y+1);
% classification accuracy
acc = num_correct_classifications/size(X,1);
fprintf(['accuracy =' , num2str(num_correct_classifications),'/1400' ]);
figure(3);
plot(error_vec);
title('error vs. iteration');
xlabel('iteration');
ylabel('error');
%% test on data validation set
%upload
data_valid=readtable('DATA_valid.csv');
data_valid.Properties.VariableNames{'Var1'} = 'x';
data_valid.Properties.VariableNames{'Var2'} = 'y';
data_valid.Properties.VariableNames{'Var3'} = 'spiral';
X1=data_valid.x;
X2=data_valid.y;
X=[X1 X2];
Y=data_valid.spiral;
Data=[X Y];

%run one time
 HL1 = X*W1+B1; %hidden layer value before activation
 HL1 = sin(HL1); %sin activation function
    
 HL2 = HL1*W2+B2; %hidden layer value before activation    
 
% % for RELU uncomment
% HL2 = max(0, HL2); 

% for tanh uncomment
HL2 = tanh(HL2);

FL = HL2*W3+B3; %final layer value
FL = exp(FL)./sum(exp(FL), 2); %apply softmax cross entropy

% now we will calculate the cost. for every sample, the cost will be
% added to the total cost. the Y vector has values of 0 or 1 so we will
% add +1 to fit value of 1 or 2.
data_error = 0;
for i = 1:size(X,1)
    data_error = data_error + -1*log(FL(i, Y(i)+1));
end
data_error = data_error/size(X,1);
% find indicies where the max probability for each class occurs
[val,idx] = max(FL, [], 2);
% correct classification is when idx matches
num_correct_classifications = sum(idx == Y+1);
% classification accuracy
acc = num_correct_classifications/size(X,1);
fprintf([newline, 'accuracy =' , num2str(num_correct_classifications),'/400' ]);