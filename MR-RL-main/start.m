fid=fopen('statandstop.txt','w');
fprintf(fid,'%d',5);  % 写入方式： d是整数型，s是字符串
fclose(fid);
