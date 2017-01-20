close all;
clear all;

load data.mat;
X=data;

inp1 = X(:,1);
out1 = X(:,1);
inp2 = X(:,2);
out2 = X(:,2);
inp3 = X(:,3);
out3 = X(:,3);
inp4 = X(:,4);
out4 = X(:,4);
inp5 = X(:,5);
out5 = X(:,5);

[net1 per1 tr1] = TimeSeriesNN(inp1(500:2890),inp1(500:2890),10,[10 10 10], 0.9, 0.05);
[accuracy1, y_pred] = Test(net1,inp1(end:end),inp1(end:end));

[net2 per2 tr2] = TimeSeriesNN(inp2(500:2890),out2(500:2890),10,[10 10 10], 0.9, 0.05);
[accuracy2, y_pred] = Test(net2,inp2(3001:end),out2(3001:end))

[net3 per3 tr3] = TimeSeriesNN(inp3(500:2890),out3(500:2890),20,[5 5 5], 0.9, 0.05);
[accuracy3, y_pred] = Test(net3,inp3(3001:end),out3(3001:end))

[net4 per4 tr4] = TimeSeriesNN(inp4(500:2890),out4(500:2890),3,[10 10 10], 0.9, 0.05)
[accuracy4, y_pred] = Test(net4,inp4(3001:end),out4(3001:end))

[net5 per5 tr5] = TimeSeriesNN(inp5(500:2890),out5(500:2890),15,[5 5 5], 0.9, 0.05);
[accuracy5, y_pred] = Test(net5,inp5(3001:end),out5(3001:end))
