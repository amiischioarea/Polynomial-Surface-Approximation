clc
clear all
close all

load("proj_fit_11.mat");

N1 = length(id.X{1});
N2 = length(id.X{2});
v_N1 = length(val.X{1});
v_N2 = length(val.X{2});

grade = 1 : 34;
mse_minim = 10000000;
for m = grade
    
    phi_id = []; %se initializeaza matricea
    phi_id = generareRegresori(N1, N2, m, id);
    
    %construirea setului de antrenare
    Y = reshape(id.Y, [], 1);   
    theta = phi_id \ Y;
    Y_trans = phi_id * theta;

    mse_id(m) = mean((Y_trans - Y).^2);

    %pentru validare

    phi_val = [];
    phi_val = generareRegresori(v_N1, v_N2, m, val);

    Yv = reshape(val.Y, [], 1);
    Y_transv = phi_val * theta;

    mse_val(m) = mean((Y_transv - Yv).^2);

    %salvam valori pentru gradul optim
    if mse_val(m) < mse_minim
        mse_minim = mse_val(m);
        grad_optim = m;
        Y_trans_optim = Y_trans;
        Y_transv_optim = Y_transv;
    end

end

figure
plot(grade, mse_id, '-o');
hold on;
plot(grade, mse_val, '-o');
xlabel('grade'); ylabel('mse'); legend('MSEid', 'MSEval');
grid on;

m=grad_optim;

figure
subplot(2, 1, 1)
mesh(id.X{1},id.X{2},id.Y);
title("antrenarea modelului (id), cu grad = ", m);
xlabel('x1'); ylabel('x2'); zlabel('y');
    
Y_id_optim = reshape(Y_trans_optim, N2, N1);

subplot(2, 1, 2)
mesh(id.X{1}, id.X{2}, Y_id_optim);
xlabel('x1'); ylabel('x2'); zlabel('y');

%pentru validare
figure
subplot(2, 1, 1)
mesh(val.X{1}, val.X{2}, val.Y);
title("date validare (val), cu grad m = ", m);
xlabel('x1'); ylabel('x2'); zlabel('y');

Y_val_optim = reshape(Y_transv_optim, v_N2, v_N1);

subplot(2, 1, 2)
mesh(val.X{1}, val.X{2}, Y_val_optim);
xlabel('x1'); ylabel('x2'); zlabel('y');


function phi = generareRegresori(N1, N2,m, id)
    row = 1;
    for i = 1 : N1
        for j = 1 : N2
            k = 1; %contor pt randuri
            for p = 0 : m
                for q = 0 : (m - p)
                    phi(row, k) = id.X{1}(i)^p * id.X{2}(j)^q;
                    k = k + 1;
                end
            end
            row = row + 1;
        end
    end
end
