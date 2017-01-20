function kernal = rbf(x,y)
kernal = exp(-dot(x-y,x-y)/20);