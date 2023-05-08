for i=1:10000
A1 = rand(3,3)*2-2;
A2 = 2*A1;
A3 = rand(3,3)*2-2;

x0 = rand(3,1)*2-2;

for j=1:10000
t = rand(3,1);

v1 = expm(A3*t(3))*expm(A2*t(2))*A1*expm(A1*t(1))*x0;
v2 = expm(A3*t(3))*A2*expm(A2*t(2))*expm(A1*t(1))*x0;
v3 = A3*expm(A3*t(3))*expm(A2*t(2))*expm(A1*t(1))*x0;

res = null([v1,v2,v3]);

if ~isempty(res)
    disp('stop')
end
end
end