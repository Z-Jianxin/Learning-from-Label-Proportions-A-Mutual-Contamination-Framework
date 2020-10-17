function [result] = run_exp(func, data_path, output_path)
tStart = tic;

result =  func(data_path);

tEnd = toc(tStart);
result.time_elapsed = tEnd;
save([output_path,], 'result');
end

