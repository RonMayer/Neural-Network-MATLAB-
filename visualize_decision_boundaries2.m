function visualize_decision_boundaries2(X, classes, W1, B1, W2, B2, W3, B3, fh, vidObj)
% draw decision boundaries
clf(fh);
xrange = [-15 15];
yrange = [-15 15];
% step size for how finely you want to visualize the decision boundary.
inc = 0.05;

% save current true class info in another variable
yy = classes;
% generate grid coordinates. this will be the basis of the decision
% boundary visualization.
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
 
% size of the (x, y) image, which will also be the size of the 
% decision boundary image that is used as the plot background.
rows = size(x,1);
cols = size(x,2);
N = rows*cols;
 
xy = [x(:) y(:)]; % make (x,y) pairs as a bunch of row vectors.
X2 = xy;
hidden_input1 = X2*W1+ones(N,1)*B1;
% apply sin non-linearity
hidden_output1 = sin(hidden_input1);

hidden_input2 = hidden_output1*W2+ones(N,1)*B2;
% % for RELU uncomment
% hidden_output2 = max(0, hidden_input2); % size: N*H
 % for tanh uncomment
hidden_output2 = tanh(hidden_input2); % size: N*H
% compute scores
scores = hidden_output2*W3+ones(N,1)*B3;
% get un-normalized probabilities
exp_scores = exp(scores);
probs = exp_scores./sum(exp_scores, 2);
[val idx] = max(probs, [], 2);
% decisions
decisions = idx;
decision_map = reshape(decisions, rows, cols); 

imagesc(xrange,yrange,decision_map);
hold on;
set(gca,'ydir','normal');
 
% colormap for the classes:
% class 1 = light red, 2 = light green, 3 = light blue
cmap = [1 0.8 0.8; 0.9 0.9 1];
colormap(cmap);

hold on;
ind = find(yy == 0); 
plot(X(ind,1), X(ind,2), 'ro');

ind = find(yy == 1); 
plot(X(ind,1), X(ind,2), 'bo');

currFrame = getframe;
writeVideo(vidObj,currFrame);
end