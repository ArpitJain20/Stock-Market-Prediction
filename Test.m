function [ performance, y_pred ] = Test( net, inp_test, out_test )

  X = tonndata(inp_test,false,false);
  T = tonndata(out_test,false,false);

  [x,xi,ai,t] = preparets(net,X,T);

  y = net(x,xi,ai);
  y_pred = cell2mat(y);
  y_tar = cell2mat(t);

  performance = DirectionPerformance(inp_test,y_tar',y_pred');
  figure;
  plot(y_pred);
  title('Obtained Values VS Timesteps');
  figure;
  plot(y_tar);
  title('Actual Values VS Timesteps');
  figure;
  plot([1:length(y_pred)],y_pred(1:length(y_pred)),'r',[1:length(y_tar)],y_tar(1:end),'b');
  title('Comparing the results');
  xlabel('Days');
  ylabel('NASDAQ COMPOSITE');

end

