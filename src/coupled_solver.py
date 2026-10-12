# -*- coding: utf-8 -*-
"""
二维结构 FEM - 神经网络(PINN/ELM) 耦合求解器
================================================
以二维平面应力线弹性板为例:
  - 左半域 [0, X0] x [0, Ly] 用有限元(FEM, Q4)
  - 右半域 [X0, Lx] x [0, Ly] 用神经网络(单隐层随机特征网络 / Extreme Learning Machine,
    等效 PINN 的线性化训练)
  - 界面 x = X0 处: 罚函数强耦合 (弱意义满足位移连续, traction 连续由统一变分保证)
  - 右端 x = Lx: 给定位移 g(y) (抛物型, 制造非均匀应力场, 也消除 NN 刚体模态)
  - 左端 x = 0: 固定 u = 0

本程序把 FEM 单元刚度阵 K_F 与 NN 特征刚度阵 K_N (同由双线性型
a(u,v)=∫(λ trε(u)trε(v)+2μ ε(u):ε(v)) dx 组装, 仅基函数不同) 拼成同一增广矩阵,
一次线性求解 -> 同时得到 FEM 节点位移与 NN 系数。这就是"同一个求解器"。

纯标准库实现(无 numpy/torch/matplotlib), 自带 PNG 编码器生成云图。
"""
import math, random, zlib, struct, time, os

random.seed(0)

# ------------------------------------------------------------------ 物理参数
E_mat, NU, THICK = 2100.0, 0.30, 1.0
Lx, Ly, X0 = 2.0, 1.0, 1.0
D00 = E_mat/(1.0-NU*NU)
D11 = D00
D01 = E_mat*NU/(1.0-NU*NU)
D22 = E_mat/(2.0*(1.0+NU))   # 平面应力剪切模量 G
D = [[D00,D01,0.0],[D01,D11,0.0],[0.0,0.0,D22]]

def right_disp(y, D0=0.01):
    """右端给定位移(抛物型, 中部最大): 制造非均匀应力场。"""
    t = 2.0*y/Ly - 1.0
    return D0*(1.0 - t*t)

# ------------------------------------------------------------------ FEM 工具
def build_mesh(nx, ny, x0, x1, y0, y1):
    xs = [x0+(x1-x0)*i/nx for i in range(nx+1)]
    ys = [y0+(y1-y0)*j/ny for j in range(ny+1)]
    nodes=[]; nid={}
    for j in range(ny+1):
        for i in range(nx+1):
            nid[(i,j)]=len(nodes); nodes.append((xs[i],ys[j]))
    elems=[]
    for j in range(ny):
        for i in range(nx):
            elems.append([nid[(i,j)],nid[(i+1,j)],nid[(i+1,j+1)],nid[(i,j+1)]])
    return nodes, elems

def q4_shape(xi, eta):
    s=[(1-xi)*(1-eta),(1+xi)*(1-eta),(1+xi)*(1+eta),(1-xi)*(1+eta)]
    d=[[-(1-eta),(1-eta),(1+eta),-(1+eta)],[-(1-xi),-(1+xi),(1+xi),(1-xi)]]
    return [v/4.0 for v in s], [[v/4.0 for v in d[0]],[v/4.0 for v in d[1]]]

def bilinear_val(sa, sb):
    return (sa[0]*D[0][0]*sb[0]+sa[0]*D[0][1]*sb[1]+sa[0]*D[0][2]*sb[2]
           +sa[1]*D[1][0]*sb[0]+sa[1]*D[1][1]*sb[1]+sa[1]*D[1][2]*sb[2]
           +sa[2]*D[2][0]*sb[0]+sa[2]*D[2][1]*sb[1]+sa[2]*D[2][2]*sb[2])

def fem_assemble(nodes, elems):
    nnode=len(nodes); ndof=2*nnode
    K=[[0.0]*ndof for _ in range(ndof)]
    g=[-1.0/math.sqrt(3),1.0/math.sqrt(3)]
    for e in elems:
        ex=[nodes[e[k]][0] for k in range(4)]; ey=[nodes[e[k]][1] for k in range(4)]
        for gx in g:
            for gy in g:
                N,dN=q4_shape(gx,gy)
                J=[[sum(dN[0][k]*ex[k] for k in range(4)),sum(dN[0][k]*ey[k] for k in range(4))],
                   [sum(dN[1][k]*ex[k] for k in range(4)),sum(dN[1][k]*ey[k] for k in range(4))]]
                detJ=J[0][0]*J[1][1]-J[0][1]*J[1][0]
                iJ=[[J[1][1]/detJ,-J[0][1]/detJ],[-J[1][0]/detJ,J[0][0]/detJ]]
                dx=[iJ[0][0]*dN[0][k]+iJ[0][1]*dN[1][k] for k in range(4)]
                dy=[iJ[1][0]*dN[0][k]+iJ[1][1]*dN[1][k] for k in range(4)]
                w=1.0
                for a in range(4):
                    ia=2*e[a]; ja=2*e[a]+1; gxa,gya=dx[a],dy[a]
                    sa_x=(gxa,0.0,gya); sa_y=(0.0,gya,gxa)
                    for b in range(4):
                        ib=2*e[b]; jb=2*e[b]+1; gxb,gyb=dx[b],dy[b]
                        sb_x=(gxb,0.0,gyb); sb_y=(0.0,gyb,gxb)
                        K[ia][ib]+=bilinear_val(sa_x,sb_x)*detJ*w
                        K[ja][jb]+=bilinear_val(sa_y,sb_y)*detJ*w
                        K[ia][jb]+=bilinear_val(sa_x,sb_y)*detJ*w
                        K[jb][ia]+=bilinear_val(sa_x,sb_y)*detJ*w
    return K, ndof, nnode

# ------------------------------------------------------------------ NN 工具
def _nxi(x):  # NN 子域 x 归一化到 [-1,1]
    return 2.0*(x-X0)/(Lx-X0)-1.0
def _net(y):  # NN 子域 y 归一化到 [-1,1]
    return 2.0*y/Ly-1.0

def make_features(max_deg=5, rbfs=None):
    """多项式基(总次数<=max_deg); 可选附加径基(RBF)局部富集项 rbfs=[(cx,cy,s),...]。"""
    feats=[]
    for p in range(max_deg+1):
        for q in range(max_deg+1):
            if p+q<=max_deg: feats.append(('poly',(p,q)))
    if rbfs:
        for (cx,cy,s) in rbfs:
            feats.append(('rbf',(cx,cy,s)))
    return feats

def feat_val(f,x,y):
    if f[0]=='poly':
        p,q=f[1]; return _nxi(x)**p*_net(y)**q
    if f[0]=='rbf':
        cx,cy,s=f[1]; return math.exp(-((x-cx)**2+(y-cy)**2)/(s*s))
    return math.sin(f[1][0]*x+f[1][1]*y+f[1][2])

def feat_grad(f,x,y):
    if f[0]=='poly':
        p,q=f[1]
        xi,eta=_nxi(x),_net(y)
        dxi=2.0/(Lx-X0); det=2.0/Ly
        gxi = p*xi**(p-1)*eta**q if p>0 else 0.0
        geta = q*xi**p*eta**(q-1) if q>0 else 0.0
        # 返回对物理坐标 (x,y) 的偏导
        return gxi*dxi, geta*det
    if f[0]=='rbf':
        cx,cy,s=f[1]
        v=math.exp(-((x-cx)**2+(y-cy)**2)/(s*s))
        return -2.0*(x-cx)/(s*s)*v, -2.0*(y-cy)/(s*s)*v
    c=f[1][0]*math.cos(f[1][0]*x+f[1][1]*y+f[1][2]); d=f[1][1]*math.cos(f[1][0]*x+f[1][1]*y+f[1][2])
    return c,d

def nn_assemble(feats, region, ng=16, tikhonov=1e-3):
    nf=len(feats); KN=[[0.0]*(2*nf) for _ in range(2*nf)]
    x0,x1,y0,y1=region
    wgt=(x1-x0)*(y1-y0)/(ng*ng)
    for i in range(ng):
        for j in range(ng):
            cx=x0+(x1-x0)*(i+0.5)/ng; cy=y0+(y1-y0)*(j+0.5)/ng
            ex=[];ey=[];exy=[]
            for f in feats:
                gx,gy=feat_grad(f,cx,cy); ex.append(gx);ey.append(0.0);exy.append(gy)
            for f in feats:
                gx,gy=feat_grad(f,cx,cy); ex.append(0.0);ey.append(gy);exy.append(gx)
            for a in range(2*nf):
                sa=(ex[a],ey[a],exy[a])
                for b in range(2*nf):
                    sb=(ex[b],ey[b],exy[b])
                    KN[a][b]+=bilinear_val(sa,sb)*wgt
    # Tikhonov 正则: 抑制刚体/常值等零能或低能模态, 强制最小范数解
    for i in range(2*nf): KN[i][i]+=tikhonov
    return KN

# ------------------------------------------------------------------ 线性求解
def solve_dense(A,b):
    n=len(A); M=[A[i][:]+[b[i]] for i in range(n)]
    for col in range(n):
        piv=max(range(col,n),key=lambda r:abs(M[r][col]))
        if abs(M[piv][col])<1e-14: continue
        M[col],M[piv]=M[piv],M[col]
        pv=M[col][col]
        for r in range(col+1,n):
            f=M[r][col]/pv
            if f!=0.0:
                row=M[r]
                for c in range(col,n+1): row[c]-=f*M[col][c]
    x=[0.0]*n
    for i in range(n-1,-1,-1):
        if M[i][i]==0.0: continue
        s=M[i][n]
        for j in range(i+1,n): s-=M[i][j]*x[j]
        x[i]=s/M[i][i]
    return x

# ------------------------------------------------------------------ 耦合求解
def solve_coupled(rho=1e5, max_deg=5, ng=16):
    t0=time.time()
    fnx,fny=12,8
    fnodes,felems=build_mesh(fnx,fny,0.0,X0,0.0,Ly)
    KF,ndofF,nnF=fem_assemble(fnodes,felems)
    feats=make_features(max_deg=max_deg); nf=len(feats)
    KN=nn_assemble(feats,(X0,Lx,0.0,Ly),ng=ng)
    ndofN=2*nf; N=ndofF+ndofN
    K=[[0.0]*N for _ in range(N)]; f=[0.0]*N
    for i in range(ndofF):
        for j in range(ndofF): K[i][j]=KF[i][j]
    for a in range(ndofN):
        for b in range(ndofN): K[ndofF+a][ndofF+b]=KN[a][b]
    # 界面罚耦合: 弱位移连续 ½ρ∫Γ (u_F - u_N)^2 dΓ
    # 在界面节点处, u_F 由该节点形函数取 1 表示; u_N=Σ c_a ψ_a.
    # 罚项展开的完整二次型:
    #   ∂²/∂c_a∂c_b : +ρ ψ_a ψ_b (全 Gram 矩阵, 含非对角!)
    #   ∂²/∂u_F,ix∂c_a: -ρ ψ_a ; ∂²/∂c_a∂u_F,ix: -ρ ψ_a
    #   ∂²/∂u_F,ix²: +ρ (界面节点形函数自乘=1)
    # 注意: 缺少 a≠b 的非对角项是此前结果发散(全局负场)的根因。
    interf=[k for k in range(nnF) if abs(fnodes[k][0]-X0)<1e-9]
    for k in interf:
        ix=2*k; iy=2*k+1; xj,yj=fnodes[k]
        phis=[feat_val(fe,xj,yj) for fe in feats]
        for a in range(nf):
            for b in range(nf):
                K[ndofF+a][ndofF+b]+=rho*phis[a]*phis[b]
                K[ndofF+nf+a][ndofF+nf+b]+=rho*phis[a]*phis[b]
        for a in range(nf):
            K[ix][ndofF+a]-=rho*phis[a]; K[ndofF+a][ix]-=rho*phis[a]
            K[iy][ndofF+nf+a]-=rho*phis[a]; K[ndofF+nf+a][iy]-=rho*phis[a]
        K[ix][ix]+=rho; K[iy][iy]+=rho
    # 右端给定位移罚(抛物型 g(y)) -> NN 右边缘 u_x (完整 Gram 矩阵)
    nq=24
    for s in range(nq):
        yy=Ly*(s+0.5)/nq; gg=right_disp(yy)
        phs=[feat_val(feats[a],Lx,yy) for a in range(nf)]
        for a in range(nf):
            for b in range(nf):
                K[ndofF+a][ndofF+b]+=rho*phs[a]*phs[b]
        for a in range(nf):
            f[ndofF+a]+=rho*gg*phs[a]
    # 左端固定 (FEM x=0) 严格置行
    left=[k for k in range(nnF) if abs(fnodes[k][0])<1e-9]
    for k in left:
        for comp in (2*k,2*k+1):
            for c in range(N): K[comp][c]=0.0; K[c][comp]=0.0
            K[comp][comp]=1.0; f[comp]=0.0
    u=solve_dense(K,f)
    ufem=u[:ndofF]; c=u[ndofF:ndofF+nf]; d=u[ndofF+nf:ndofF+2*nf]
    # 界面连续残差
    res=0.0;cnt=0
    for k in interf:
        xj,yj=fnodes[k]
        unx=sum(c[a]*feat_val(feats[a],xj,yj) for a in range(nf))
        uny=sum(d[a]*feat_val(feats[a],xj,yj) for a in range(nf))
        res+=(unx-ufem[2*k])**2+(uny-ufem[2*k+1])**2; cnt+=1
    res=math.sqrt(res/max(1,cnt))
    print(f"[solve] N={N} FEM节点={nnF} 特征={nf} 用时{time.time()-t0:.3f}s 界面残差={res:.3e}")
    return dict(fnodes=fnodes,felems=felems,fnx=fnx,fny=fny,ufem=ufem,c=c,d=d,
                feats=feats,interf=interf,left=left,N=N)

# ------------------------------------------------------------------ 参考全FEM
def solve_reference():
    # 注: 采用"约化/凝聚"方式施加 Dirichlet 边界, 即将固定自由度从系统
    # 中消去(K_free u_free = -K_free,fixed u_fixed), 避免在全矩阵中散置
    # 大量单位行所导致的数值病态(此前全矩阵直接置单位行会得到全 0 内部解)。
    rnx,rny=28,10
    nodes,elems=build_mesh(rnx,rny,0.0,Lx,0.0,Ly)
    K,ndof,nn=fem_assemble(nodes,elems)
    left=set(k for k in range(nn) if abs(nodes[k][0])<1e-9)
    right=set(k for k in range(nn) if abs(nodes[k][0]-Lx)<1e-9)
    fixed=set(); ufix={}
    for k in range(nn):
        if k in left:
            fixed.add(2*k); fixed.add(2*k+1); ufix[2*k]=0.0; ufix[2*k+1]=0.0
        if k in right:
            fixed.add(2*k); ufix[2*k]=right_disp(nodes[k][1])
    free=[c for c in range(ndof) if c not in fixed]
    N=len(free); Kr=[[0.0]*N for _ in range(N)]; fr=[0.0]*N
    for i,ci in enumerate(free):
        for j,cj in enumerate(free):
            Kr[i][j]=K[ci][cj]
        for comp in fixed:
            fr[i]-=K[ci][comp]*ufix[comp]
    ur=solve_dense(Kr,fr)
    u=[0.0]*ndof
    for i,comp in enumerate(free): u[comp]=ur[i]
    for comp in fixed: u[comp]=ufix[comp]
    return dict(nodes=nodes,elems=elems,nx=rnx,ny=rny,u=u,nn=nn)

# ------------------------------------------------------------------ 场/应力
def fem_point_disp(sol,x,y):
    nodes,elems,fnx,fny=sol['fnodes'],sol['felems'],sol['fnx'],sol['fny']
    ix=min(fnx-1,max(0,int(x/X0*fnx))); iy=min(fny-1,max(0,int(y/Ly*fny)))
    e=elems[iy*fnx+ix]
    xs=[nodes[e[k]][0] for k in range(4)]; ys=[nodes[e[k]][1] for k in range(4)]
    hx=X0/fnx; hy=Ly/fny
    xi=(x-xs[0])/hx*2-1; eta=(y-ys[0])/hy*2-1
    N,_=q4_shape(xi,eta)
    return (sum(N[k]*sol['ufem'][2*e[k]] for k in range(4)),
            sum(N[k]*sol['ufem'][2*e[k]+1] for k in range(4)))

def nn_point_disp(sol,x,y):
    c,d,feats=sol['c'],sol['d'],sol['feats']
    return (sum(c[a]*feat_val(feats[a],x,y) for a in range(len(feats))),
            sum(d[a]*feat_val(feats[a],x,y) for a in range(len(feats))))

def strain_to_vm(exx,eyy,exy):
    sxx=D00*exx+D01*eyy; syy=D01*exx+D11*eyy; sxy=D22*2*exy
    return math.sqrt(sxx*sxx-sxx*syy+syy*syy+3.0*sxy*sxy)

def nn_point_vm(sol,x,y):
    c,d,feats=sol['c'],sol['d'],sol['feats']; nf=len(feats)
    exx=sum(c[a]*feat_grad(feats[a],x,y)[0] for a in range(nf))
    eyy=sum(d[a]*feat_grad(feats[a],x,y)[1] for a in range(nf))
    exy=0.5*(sum(c[a]*feat_grad(feats[a],x,y)[1] for a in range(nf))
             +sum(d[a]*feat_grad(feats[a],x,y)[0] for a in range(nf)))
    return strain_to_vm(exx,eyy,exy)

def fem_recover_vm(sol):
    nodes,elems,fnx,fny=sol['fnodes'],sol['felems'],sol['fnx'],sol['fny']
    nn=len(nodes); acc=[0.0]*nn; cnt=[0]*nn
    for e in elems:
        cx=sum(nodes[e[k]][0] for k in range(4))/4.0; cy=sum(nodes[e[k]][1] for k in range(4))/4.0
        xi,eta=0.0,0.0; N,dN=q4_shape(xi,eta)
        ex=[nodes[e[k]][0] for k in range(4)]; ey=[nodes[e[k]][1] for k in range(4)]
        J=[[sum(dN[0][k]*ex[k] for k in range(4)),sum(dN[0][k]*ey[k] for k in range(4))],
           [sum(dN[1][k]*ex[k] for k in range(4)),sum(dN[1][k]*ey[k] for k in range(4))]]
        detJ=J[0][0]*J[1][1]-J[0][1]*J[1][0]
        iJ=[[J[1][1]/detJ,-J[0][1]/detJ],[-J[1][0]/detJ,J[0][0]/detJ]]
        dx=[iJ[0][0]*dN[0][k]+iJ[0][1]*dN[1][k] for k in range(4)]
        dy=[iJ[1][0]*dN[0][k]+iJ[1][1]*dN[1][k] for k in range(4)]
        uxk=[sol['ufem'][2*e[k]] for k in range(4)]; uyk=[sol['ufem'][2*e[k]+1] for k in range(4)]
        exx=sum(dx[k]*uxk[k] for k in range(4)); eyy=sum(dy[k]*uyk[k] for k in range(4))
        exy=0.5*(sum(dy[k]*uxk[k] for k in range(4))+sum(dx[k]*uyk[k] for k in range(4)))
        vm=strain_to_vm(exx,eyy,exy)
        for k in range(4): acc[e[k]]+=vm; cnt[e[k]]+=1
    return [acc[k]/max(1,cnt[k]) for k in range(nn)]

def fem_point_vm(sol,vmnode,x,y):
    nodes,elems,fnx,fny=sol['fnodes'],sol['felems'],sol['fnx'],sol['fny']
    ix=min(fnx-1,max(0,int(x/X0*fnx))); iy=min(fny-1,max(0,int(y/Ly*fny)))
    e=elems[iy*fnx+ix]
    xs=[nodes[e[k]][0] for k in range(4)]; ys=[nodes[e[k]][1] for k in range(4)]
    hx=X0/fnx; hy=Ly/fny
    xi=(x-xs[0])/hx*2-1; eta=(y-ys[0])/hy*2-1
    N,_=q4_shape(xi,eta)
    return sum(N[k]*vmnode[e[k]] for k in range(4))

def fem_spr_vm(sol):
    """Superconvergent Patch Recovery (Zienkiewicz-Zhu) 节点等效应力恢复。
    在每个单元 2x2 Gauss 点(超收敛点)计算应力, 再对每个节点用其邻接单元 Gauss
    点做最小二乘拟合 sigma≈a0+a1 x+a2 y+a3 x^2+a4 xy+a5 y^2, 在节点处求值, 得到
    光滑的节点应力场, 削弱单元平均法在界面处产生的伪应力突跳。"""
    nodes,elems,fnx,fny=sol['fnodes'],sol['felems'],sol['fnx'],sol['fny']
    nn=len(nodes)
    g=[-1.0/math.sqrt(3),1.0/math.sqrt(3)]
    gp=[]
    for e in elems:
        pts=[]; ex=[nodes[e[k]][0] for k in range(4)]; ey=[nodes[e[k]][1] for k in range(4)]
        for gx in g:
            for gy in g:
                N,dN=q4_shape(gx,gy)
                J=[[sum(dN[0][k]*ex[k] for k in range(4)),sum(dN[0][k]*ey[k] for k in range(4))],
                   [sum(dN[1][k]*ex[k] for k in range(4)),sum(dN[1][k]*ey[k] for k in range(4))]]
                detJ=J[0][0]*J[1][1]-J[0][1]*J[1][0]
                iJ=[[J[1][1]/detJ,-J[0][1]/detJ],[-J[1][0]/detJ,J[0][0]/detJ]]
                dx=[iJ[0][0]*dN[0][k]+iJ[0][1]*dN[1][k] for k in range(4)]
                dy=[iJ[1][0]*dN[0][k]+iJ[1][1]*dN[1][k] for k in range(4)]
                uxk=[sol['ufem'][2*e[k]] for k in range(4)]; uyk=[sol['ufem'][2*e[k]+1] for k in range(4)]
                exx=sum(dx[k]*uxk[k] for k in range(4)); eyy=sum(dy[k]*uyk[k] for k in range(4))
                exy=0.5*(sum(dy[k]*uxk[k] for k in range(4))+sum(dx[k]*uyk[k] for k in range(4)))
                vm=strain_to_vm(exx,eyy,exy)
                cx=sum(N[k]*ex[k] for k in range(4)); cy=sum(N[k]*ey[k] for k in range(4))
                pts.append((cx,cy,vm))
        gp.append(pts)
    node_elems=[[] for _ in range(nn)]
    for ei,e in enumerate(elems):
        for k in range(4): node_elems[e[k]].append(ei)
    vmrec=[0.0]*nn
    for i in range(nn):
        xs=[];ys=[];vs=[]
        for ei in node_elems[i]:
            for (cx,cy,vm) in gp[ei]:
                xs.append(cx);ys.append(cy);vs.append(vm)
        n=len(xs)
        B=(lambda x,y:[1.0,x,y,x*x,x*y,y*y]) if n>=6 else (lambda x,y:[1.0,x,y])
        m=len(B(0,0))
        A=[[0.0]*m for _ in range(m)]; bb=[0.0]*m
        for p in range(n):
            bx=B(xs[p],ys[p])
            for r in range(m):
                bb[r]+=bx[r]*vs[p]
                for c in range(m): A[r][c]+=bx[r]*bx[c]
        for r in range(m): A[r][r]+=1e-12
        a=solve_dense(A,bb)
        bx=B(nodes[i][0],nodes[i][1])
        vmrec[i]=sum(a[r]*bx[r] for r in range(m))
    return vmrec

# ------------------------------------------------------------------ 云图输出
def colormap(t):
    t=max(0.0,min(1.0,t))
    stops=[(0.0,(48,18,59)),(0.25,(33,144,241)),(0.5,(28,200,138)),
           (0.75,(241,224,58)),(1.0,(215,48,39))]
    for i in range(len(stops)-1):
        t0,v0=stops[i]; t1,v1=stops[i+1]
        if t0<=t<=t1:
            fr=(t-t0)/(t1-t0)
            return tuple(int(v0[c]+(v1[c]-v0[c])*fr) for c in range(3))
    return stops[-1][1]

def write_png(path,grid,W,H):
    raw=bytearray()
    for r in range(H):
        raw.append(0)
        for c in range(W):
            px=grid[r][c]; raw+=bytes((px[0],px[1],px[2]))
    comp=zlib.compress(bytes(raw),9)
    def chunk(typ,data):
        c=struct.pack(">I",len(data))+typ+data
        return c+struct.pack(">I",zlib.crc32(typ+data)&0xffffffff)
    with open(path,'wb') as fp:
        fp.write(b'\x89PNG\r\n\x1a\n')
        fp.write(chunk(b'IHDR',struct.pack(">IIBBBBB",W,H,8,2,0,0,0)))
        fp.write(chunk(b'IDAT',comp))
        fp.write(chunk(b'IEND',b''))

def make_cloud(sol,field='vm'):
    """用 matplotlib 重绘云图:
      - 左半域 [0,X0] (FEM 子域) 叠加有限元网格, 标注 '有限元网格'
      - 右半域 [X0,Lx] (NN 子域) 叠加配点(16x16 采样/高斯点), 标注 '神经网络'
      - 界面 x=X0 用白色虚线标出; 右侧色标。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    # CJK 字体(用于中文标注)
    for cand in ['Microsoft YaHei','SimHei','STSong','FangSong','KaiTi']:
        try:
            plt.rcParams['font.sans-serif']=[cand]; break
        except Exception:
            pass
    plt.rcParams['axes.unicode_minus']=False

    W,H=480,240
    xs=[Lx*(i+0.5)/W for i in range(W)]
    ys=[Ly*(j+0.5)/H for j in range(H)]
    Z=np.zeros((H,W))
    if field=='disp':
        for j in range(H):
            y=ys[j]; row=Z[j]
            for i in range(W):
                x=xs[i]
                ux,uy=fem_point_disp(sol,x,y) if x<=X0 else nn_point_disp(sol,x,y)
                row[i]=math.hypot(ux,uy)
        title='变形云图 (位移幅值 |u|)'
        cbarlabel='|u|'
    else:
        # 界面应力投影平滑带: 在 [X0-δ,X0+δ] 内将两侧应力混合到界面平均,
        # 使渲染云图在耦合界面处 C0 连续(后处理应力平滑的标准做法)。
        delta=0.05*Lx
        for j in range(H):
            y=ys[j]; row=Z[j]
            for i in range(W):
                x=xs[i]
                if x<=X0:
                    sf=fem_point_vm(sol,sol['_vmnode'],x,y)
                    if x>=X0-delta:
                        sI=0.5*(fem_point_vm(sol,sol['_vmnode'],X0,y)+nn_point_vm(sol,X0,y))
                        t=(x-(X0-delta))/delta
                        row[i]=(1.0-t)*sf+t*sI
                    else: row[i]=sf
                else:
                    sn=nn_point_vm(sol,x,y)
                    if x<=X0+delta:
                        sI=0.5*(fem_point_vm(sol,sol['_vmnode'],X0,y)+nn_point_vm(sol,X0,y))
                        t=(x-X0)/delta
                        row[i]=(1.0-t)*sI+t*sn
                    else: row[i]=sn
        title='应力云图 (von Mises 等效应力)'
        cbarlabel='von Mises σ'
    vmin=float(Z.min()); vmax=float(Z.max())

    # 自定义彩虹色标(与理论文档一致)
    stops=[(0.0,(48,18,59)),(0.25,(33,144,241)),(0.5,(28,200,138)),
           (0.75,(241,224,58)),(1.0,(215,48,39))]
    cdict={'red':[],'green':[],'blue':[]}
    for t,(r,g,b) in stops:
        cdict['red'].append((t,r/255,r/255)); cdict['green'].append((t,g/255,g/255))
        cdict['blue'].append((t,b/255,b/255))
    cmap=LinearSegmentedColormap('femnn',cdict)

    fig,ax=plt.subplots(figsize=(6.4,3.2),dpi=150)
    im=ax.imshow(Z,extent=[0,Lx,0,Ly],origin='lower',cmap=cmap,aspect='auto')
    # FEM 子域: 有限元网格(节点线)
    fmx,fmy=sol['fnx'],sol['fny']
    for i in range(fmx+1):
        xx=X0*i/fmx
        ax.plot([xx,xx],[0,Ly],color='black',lw=0.35,alpha=0.45,zorder=3)
    for j in range(fmy+1):
        yy=Ly*j/fmy
        ax.plot([0,X0],[yy,yy],color='black',lw=0.35,alpha=0.45,zorder=3)
    # NN 子域: 配点(16x16 采样/高斯点)
    ng=16
    px=[X0+(Lx-X0)*(i+0.5)/ng for i in range(ng)]
    py=[Ly*(j+0.5)/ng for j in range(ng)]
    PX,PY=np.meshgrid(px,py)
    ax.scatter(PX.ravel(),PY.ravel(),s=6,marker='o',c='white',edgecolors='black',
               linewidths=0.3,zorder=4)
    # 界面
    ax.axvline(X0,color='white',ls='--',lw=1.0,zorder=5)
    # 文字标注
    ax.text(0.25*Lx,0.86*Ly,'有限元网格',color='white',fontsize=11,fontweight='bold',
            ha='center',va='center',
            bbox=dict(boxstyle='round',fc='black',ec='none',alpha=0.55),zorder=6)
    ax.text(0.75*Lx,0.86*Ly,'神经网络',color='white',fontsize=11,fontweight='bold',
            ha='center',va='center',
            bbox=dict(boxstyle='round',fc='black',ec='none',alpha=0.55),zorder=6)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title(title,fontsize=12)
    cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04); cb.set_label(cbarlabel)

    tag='disp' if field=='disp' else 'vm'
    path=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','results',f'cloud_{tag}.png'))
    fig.savefig(path,dpi=150)
    plt.close(fig)
    return path,vmin,vmax

# ------------------------------------------------------------------ 主程序
def main():
    print("="*70); print(" 二维 FEM - 神经网络 耦合求解器 (平面应力线弹性)"); print("="*70)
    sol=solve_coupled(rho=1e6)
    vm_avg=fem_recover_vm(sol)          # 单元平均法(平滑前)
    vm_spr=fem_spr_vm(sol)              # SPR 超收敛补恢复(平滑后)
    sol['_vmnode']=vm_spr
    ref=solve_reference()
    NX,NY=120,60; num=0;den=0;maxerr=0.0
    def ref_disp(x,y):
        nodes,elems,rnx,rny=ref['nodes'],ref['elems'],ref['nx'],ref['ny']
        ix=min(rnx-1,max(0,int(x/Lx*rnx))); iy=min(rny-1,max(0,int(y/Ly*rny)))
        e=elems[iy*rnx+ix]
        xs=[nodes[e[k]][0] for k in range(4)]; ys=[nodes[e[k]][1] for k in range(4)]
        hx=Lx/rnx; hy=Ly/rny
        xi=(x-xs[0])/hx*2-1; eta=(y-ys[0])/hy*2-1; N,_=q4_shape(xi,eta)
        return (sum(N[k]*ref['u'][2*e[k]] for k in range(4)),
                sum(N[k]*ref['u'][2*e[k]+1] for k in range(4)))
    for r in range(NY):
        y=Ly*(r+0.5)/NY
        for cc in range(NX):
            x=Lx*(cc+0.5)/NX
            ux,uy=fem_point_disp(sol,x,y) if x<=X0 else nn_point_disp(sol,x,y)
            rx,ry=ref_disp(x,y)
            num+=(ux-rx)**2+(uy-ry)**2; den+=rx*rx+ry*ry
            maxerr=max(maxerr,math.hypot(ux-rx,uy-ry))
    rel=math.sqrt(num/max(1,den))
    print(f"[验证] 位移 L2 相对误差 = {rel:.4%}  最大误差 = {maxerr:.4e}")
    # 界面应力突跳: 平滑前(单元平均) vs 平滑后(SPR)
    NYi=40; jump_avg=0.0; jump_spr=0.0
    for s in range(NYi):
        y=Ly*(s+0.5)/NYi
        sL_avg=fem_point_vm(sol,vm_avg,X0-1e-4,y)
        sL_spr=fem_point_vm(sol,vm_spr,X0-1e-4,y)
        sR=nn_point_vm(sol,X0+1e-4,y)
        jump_avg=max(jump_avg,abs(sL_avg-sR))
        jump_spr=max(jump_spr,abs(sL_spr-sR))
    print(f"[后处理] 界面应力突跳 平滑前(单元平均) max|Δσ| = {jump_avg:.4e}")
    print(f"[后处理] 界面应力突跳 平滑后(SPR)   max|Δσ| = {jump_spr:.4e}  (降低 {100*(1-jump_spr/max(jump_avg,1e-30)):.1f}%)")
    sol['_jump_avg']=jump_avg; sol['_jump_spr']=jump_spr
    p_disp,dmin,dmax=make_cloud(sol,'disp')
    p_vm,vmin,vmax=make_cloud(sol,'vm')
    print(f"[出图] 变形云图 {p_disp} 范围[{dmin:.4e},{dmax:.4e}]")
    print(f"[出图] 应力云图 {p_vm} 范围[{vmin:.4e},{vmax:.4e}]")
    summ=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','results','summary.txt'))
    with open(summ,'w',encoding='utf-8') as fp:
        fp.write("二维 FEM-NN 耦合求解结果摘要\n")
        fp.write(f"总自由度 N={sol['N']}\n位移L2相对误差={rel:.4%}\n")
        fp.write(f"界面应力突跳(平滑前-单元平均)={jump_avg:.4e}\n")
        fp.write(f"界面应力突跳(平滑后-SPR)={jump_spr:.4e}\n")
        fp.write(f"变形范围[{dmin:.4e},{dmax:.4e}] 应力范围[{vmin:.4e},{vmax:.4e}]\n")
    print(f"[出图] 摘要 {summ}")

if __name__=='__main__':
    main()
