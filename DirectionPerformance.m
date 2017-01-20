function [ performance ] = DirectionPerformance( inp_test, out_test, out_pred )
  out_test = out_test(end-length(out_pred)+1:end);
  inp_test = inp_test(end-length(out_pred):end-1);

  dir_train = out_test - inp_test;
  dir_pred = out_pred - inp_test;

  dir_error = sign(dir_train .* dir_pred);

  performance = sum(dir_error(dir_error()>0))/length(out_pred);
end

