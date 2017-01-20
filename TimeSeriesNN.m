function [ net, performance, tr ] = TimeSeriesNN( inp_train, out_train, delay, LayerArray, train_ratio, val_ratio )
  %   inp_train - input time series.
  %   out_train - target time series.

  X = tonndata(inp_train,false,false);
  T = tonndata(out_train,false,false);

  % Choose a Training Function
  % 'trainlm' is usually fastest.
  % 'trainbr' takes longer but may be better for challenging problems.
  % 'trainscg' uses less memory. Suitable in low memory situations.
  trainFcn = 'trainbr';  % Levenberg-Marquardt backpropagation.

  % Create a Time Delay Network
  inputDelays = 1:delay;
  hiddenLayerArray = LayerArray;
  net = timedelaynet(inputDelays,hiddenLayerArray,trainFcn);

  % Prepare the Data for Training and Simulation
  [x,xi,ai,t] = preparets(net,X,T);

  % Setup Division of Data for Training, Validation, Testing
  net.divideParam.trainRatio = train_ratio;
  net.divideParam.valRatio = val_ratio;
  net.divideParam.testRatio = 1-train_ratio-val_ratio;

  % Train the Network
  [net,tr] = train(net,x,t,xi,ai);

  % Test the Network
  y = net(x,xi,ai);
  e = gsubtract(t,y);
  performance = [perform(net,t,y)];

  % View the Network
  % view(net);

  %Plot the data versus timesteps
  y_pred = cell2mat(y);
  figure;
  plot([1:length(y_pred)],y_pred);
  title('Obtained Values VS Timesteps');
  figure;
  plot([1:length(out_train)-delay],out_train(1+delay:end));
  title('Actual Values VS Timesteps');
  figure;
  plot([1:length(y_pred)],y_pred,'r',[1:length(out_train)-delay],out_train(1+delay:end),'b');

  % Performance for prediction of rise or fall of Stock Market
  performance =[ performance  DirectionPerformance(inp_train, out_train, y_pred')];

end
