%% Constantes
d2r = pi/180;

%% Faz derivada simbolica
syms th1 th2 th3 th4 th5 th6 th7 th8 th9 a q de;
xs = [a; q];
us = [de];
p0 = [th1; th2; th3; th4; th5; th6; th7; th8; th9];
fs = f_attas_sp(xs, us, p0);
gs = g_attas_sp(xs, us, p0);
df_dths = jacobian(fs, p0);
dg_dths = jacobian(gs, p0);
df_dxs = jacobian(fs, xs);
dg_dxs = jacobian(gs, xs);

fgen = matlabFunction(fs, df_dxs, df_dths, 'Vars', {xs, us, p0}, ...
    'File', 'f_attas_sp_gen');
ggen = matlabFunction(gs, dg_dxs, dg_dths, 'Vars', {xs, us, p0}, ...
    'File', 'g_attas_sp_gen');

%% Carrega os dados

url = 'https://arc.aiaa.org/doi/suppl/10.2514/4.102790/suppl_file/flt_data.zip';
unzip(url);
data = load('flt_data/fAttasElv1.mat');

t = data.fAttasElv1(:, 1);
u = data.fAttasElv1(:, 22) * d2r;
z = data.fAttasElv1(:, [13, 8]) * d2r;

fltdata = struct;
fltdata.t = t;
fltdata.u = u;
fltdata.z = z;

z_pre = mean(z(t<5,:), 1).'; % z pre excitacao
u_pre = mean(u(t<5,:), 1).'; % u pre excitacao
%% Cria as funções do modelo
x0 = [z_pre(1); 0];

R0 = diag(var(z(t<5, :), 0, 1));
p0 = [-1; 0; 0; 0; -1; 0; z_pre(1); u_pre(1); 0];

f = @f_attas_sp;
g = @g_attas_sp;

mdlsim = @(theta) euler_sim(x0, t, u, f, g, theta);
mdlsim_gen = @(theta) euler_sim(x0, t, u, fgen, ggen, theta);

oemopt = oemoptions();
oemopt.R = '';
fun = @(p) oem_obj(p, z, mdlsim, oemopt);

model = struct;
model.f = fgen;
model.g = ggen;
model.x0 = x0;
model.R = '';

oemoptRev = oemRevOptions();
obj = @(p) oem_obj_rev(p, fltdata, model, oemoptRev);
%% Chama o otimizador

optopt = optimoptions('fminunc');
optopt.Algorithm = 'trust-region';
optopt.Display = 'iter';
optopt.SpecifyObjectiveGradient = true;
optopt.HessianFcn = 'objective';

optoptRev = optimoptions('fminunc');
optoptRev.Algorithm = 'quasi-newton';%trust-region quasi-newton
optoptRev.Display = 'iter';
optoptRev.SpecifyObjectiveGradient = true;

thetaopt = fminunc(fun, p0, optopt);
popt = fminunc(obj, p0, optoptRev);
%%
oemopt.deriv = 'user-fcn';
[~, grad] = oem_obj(thetaopt, z, mdlsim_gen, oemopt);

nsteps = 1000;
steps = logspace(0, -300, nsteps);
err_complex = zeros(nsteps, 1);
err_fwd = zeros(nsteps, 1);
err_central = zeros(length(steps), 1);

for i = 1:nsteps
    oemopt.diff_step = steps(i);
    
    oemopt.deriv = 'complex-step';
    [~, grad_complex] = oem_obj(thetaopt, z, mdlsim, oemopt);
    err_complex(i) = mean(abs(grad - grad_complex));

    oemopt.deriv = 'fwd-diff';
    [~, grad_fwd] = oem_obj(thetaopt, z, mdlsim, oemopt);
    err_fwd(i) = mean(abs(grad - grad_fwd));
    
    oemopt.deriv = 'central-diff';
    [~, grad_central] = oem_obj(thetaopt, z, mdlsim, oemopt);
    err_central(i) = mean(abs(grad - grad_central));
end

figure(1);
tiled = tiledlayout(1,5,'TileSpacing','tight');
ay1 = nexttile(1, [1 4]);
loglog(ay1, steps, err_complex , '.', steps, err_fwd , '.', steps, ... 
    err_central , '.')
set(gca, 'xdir', 'rev')
legend('Passo Complexo','Passo a frente', 'Passo central', 'location', ...
    'best','FontSize', 14)
xlim([1e-19 1e0 ])
ylim([1e-15 1e6 ])
ay2 = nexttile;
loglog(ay2, steps, err_complex , '.', steps, err_fwd , '.', steps, ... 
    err_central , '.')
set(gca, 'xdir', 'rev')
xlim([1e-300 1e-296])
ylim([1e-15 1e6 ])
xlabel(tiled, 'Passo de diferenciação h','FontSize', 18)
yticklabels({})
ylabel(tiled, 'error médio absoluto','FontSize', 18)

%%
y0 = mdlsim(p0);
[~, ~, ~, yopt_rev] = obj(popt);

oemopt.deriv = 'complex-step';
yopt = mdlsim(thetaopt);

figure(2)
tiled = tiledlayout(2,1,'TileSpacing','tight');
ax1 = nexttile;
plot(ax1,t, rad2deg(z(:,1)), '.', t, rad2deg(yopt(:,1)), t, rad2deg(y0(:, 1)), '--')
subtitle('Angulo de ataque ','FontSize', 18)
legend({'medicoes', 'estimado', 'inicial'},'FontSize', 15)%, 'location', 'best'
ylabel('\alpha (\circ)','FontSize', 18);

ax2 = nexttile;
plot(ax2,t, rad2deg(z(:,2)), '.', t, rad2deg(yopt(:,2)), t, rad2deg(y0(:, 2)), '--')
subtitle('Velocidade de Arfagem','FontSize', 18)
ylim([-9 9])
legend({'medicoes', 'estimado', 'inicial'},'FontSize', 15)
ylabel(ax2, 'q (deg/s)','FontSize', 18);
xlabel('tempo (s)', 'FontSize', 18);
%linkaxes([ax1,ax2],'x');
xticklabels(ax1,{})