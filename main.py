import numpy as np
import math, copy, os, yaml, subprocess
import initial,obstacle,boundary,stgg,cfxy,diffusion,rhs
import newgrd,cip2d,uxvx,fric,mkzero,hcal,sgcurve,grid,uniform,dryck
import ros

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib._version import get_versions as mplv

cn='config.yml'

# Open Config File
with open(cn,'r', encoding='utf-8') as yml:   
    config = yaml.safe_load(yml)

lam=float(config['lam'])
nx0=int(config['nx0']);ny=int(config['ny'])
nym=int(ny/2+.5)
ds0=lam/float(nx0-1); dds=ds0/10.

wn=int(config['wn']); nx=(nx0-1)*wn+1

nx1=nx+1; ny1=ny+1; chl=lam*wn
nx2=nx1+1; ny2=ny1+1

chb=float(config['chb'])
t0_degree=float(config['t0_degree'])
slope=float(config['slope'])
xsize=float(config['xsize'])

amp=float(config['amp']); amp_0=float(config['amp_0'])
beta_0=float(config['beta_0'])
delta=float(config['delta'])

iskip=int(config['iskip'])
jskip=int(config['jskip'])
vcolor=config['vcolor']
jscale=int(config['jscale'])

rrmax=float(config['rrmax'])

s0=0.; x0=0.; y0=0.0 #始点のsおよびx,y座標

xpos=np.zeros([nx1]); ypos=np.zeros([nx1]); thepos=np.zeros([nx1]) #センターのx,y座標, 偏角
zpos=np.zeros([nx1]) #センターの初期基準河床
spos=np.zeros([nx1]) #センターのs軸距離

spos_c=np.zeros([nx1]) #セルセンター・河道センターの縦断距離
hpos_c=np.zeros([nx1]) #センターの初期水位(node)
c_area=np.zeros([nx1]) # 断面積(cell)
e_slope=np.zeros([nx1]) # エネルギー勾配
alf_f=np.zeros([nx1]) # エネルギー補正係数

pval=np.zeros([nx1]) #センターに沿ってプロットする値
xr=np.zeros([nx1]); xl=np.zeros([nx1]); yr=np.zeros([nx1]); yl=np.zeros([nx1]) #左右岸のx,y座標 
xgrid=np.zeros([nx1,ny1]); ygrid=np.zeros([nx1,ny1]); zgrid=np.zeros([nx1,ny1]); dz=np.zeros([nx1,ny1]) # x,y座標とzの初期攪乱

#センターライン上の座標の計算
xpos,ypos,thepos,spos=sgcurve.center(nx,t0_degree,s0,x0,y0,lam,dds,xpos,ypos,thepos,spos)

#格子全体の座標の計算
xr,yr,xl,yl,xgrid,ygrid,zgrid=sgcurve.sggrid(nx,chb,chl,slope,lam,amp,delta,xpos,ypos,spos,thepos,ny,dz,xr,yr,xl,yl,xgrid,ygrid,zgrid,beta_0,amp_0)

xset1=0.1;xset2=0.95;yset1=0.2;yset2=0.8
xslen=xset2-xset1;yslen=yset2-yset1

xmin=np.min(xgrid); xmax=np.max(xgrid)
ymin=np.min(ygrid); ymax=np.max(ygrid)
ylen=ymax-ymin; xlen=xmax-xmin
#print(ymin,ymax)

if ylen<xlen*0.6:
    yct=(ymax+ymin)/2.
    ymax=yct+xlen*.6*.5;ymin=yct-xlen*.6*.5
    ylen=ymax-ymin
ysize=xsize*xslen/yslen/xlen*ylen
ysize0=ysize
xmin0=xmin; ymin0=ymin; xmax0=xmax; ymax0=ymax;xlen0=xlen;ylen0=ylen
ysize0=float(int(ysize0))

#print(ymin,ymax);exit()
if np.mod(int(ysize0),2) == 1:
    ysize0=ysize0+1



# 砂州コンターの描画

fig,ax=plt.subplots(figsize = (xsize, ysize0))
im=ax.set_ylim(ymin0,ymax0)
im=ax.set_xlim(xmin0,xmax0)
im=ax.set_title("Sin-Generated Curve",size='25')
im=ax.set_xlabel("x(m)",size='20')
im=ax.set_ylabel("y(m)",size='20')
im=ax.contourf(xgrid,ygrid,dz,cmap='viridis')
fig.colorbar(im)
fig.savefig("ini_contour.png")
plt.clf(); plt.close()

#格子サイズ, 曲率半径などの計算
ds=np.zeros([nx1,ny1]);dn=np.zeros([nx1,ny1]);dsi=np.zeros([nx1,ny1]);dnj=np.zeros([nx1,ny1])

# rho_r:座標の曲率  rho_s:座標の曲率(uの計算点)  rho_n: 座標の曲率(vの計算点)
rho_r=np.zeros([nx1,ny1]);rho_s=np.zeros([nx1,ny1]);rho_n=np.zeros([nx1,ny1])
coss=np.zeros([nx1,ny1]);sins=np.zeros([nx1,ny1]);area=np.zeros([nx1,ny1])

# rhos_r: 流線の曲率 rhos_s:流線の曲率(uの計算点) rhos_n: 座標の曲率(vの計算点)
rhos_r=np.zeros([nx1,ny1]);rhos_s=np.zeros([nx1,ny1]);rhos_n=np.zeros([nx1,ny1])

ds,dn,dsi,dnj,rho_r,rho_s,rho_n,coss,sins,area=grid.dsdncal(nx,ny,ds,dn,dsi,dnj,rho_r,rho_s,rho_n,coss,sins,area,xgrid,ygrid)
spos_c=grid.center(nx,nym,dsi,spos_c)
rmin=min(np.min(ds),np.min(dn))

xmin=np.min(spos); xmax=np.max(spos)
pval[:]=rho_r[:,nym]
ymin=np.min(pval)*2.; ymax=np.max(pval)*2.
ymin=min(ymin,-rrmax);ymax=max(ymax,rrmax)
ysize=xsize*.8
ymin_r=ymin; ymax_r=ymax

# 流路曲率のプロット

fig,ax=plt.subplots(figsize = (xsize, ysize))
im=ax.set_ylim(ymin,ymax)
im=ax.set_xlim(xmin,xmax)
im=ax.set_title("Curvature",size='25')
im=ax.set_xlabel("s(m)",size='20')
im=ax.set_ylabel("1/r(1/m)",size='20')
im=ax.plot(spos,pval,label='1/r',linewidth=4)
fig.savefig("curvature.png")
plt.clf(); plt.close()

# 水理条件の読み込み

snu_0=float(config['snu_0'])
hmin=float(config['hmin'])
cw=float(config['cw'])
ep_alpha=float(config['ep_alpha'])
qp=float(config['qp'])
snm=float(config['snm'])
g=float(config['g'])

j_rep=int(config['j_rep']); alpha_upu=float(config['alpha_upu'])
j_west=int(config['j_west']);j_east=int(config['j_east'])
j_hdown=int(config['j_hdown'])
j_inih=int(config['j_inih'])

alh=float(config['alh']); lmax=int(config['lmax'])
etime=float(config['etime']);tuk=float(config['tuk'])
stime=float(config['stime'])
dt=float(config['dt'])

j_obst=int(config['j_obst'])
jo_type=int(config['jo_type'])
jo_method=int(config['jo_method'])
f_dike_pos=float(config['f_dike_pos'])
dike_dis=float(config['dike_dis'])
dike_length=float(config['dike_length'])
dike_thick=float(config['dike_thick'])
num_dike=int(config['num_dike'])

g_sqrt=np.sqrt(g)
errmax=hmin
it_out=int(tuk/dt)
qmax=1.5*qp; qmin=0.

## 主要変数の配列

u=np.zeros([nx1,ny1]); v=np.zeros([nx1,ny1]); un=np.zeros([nx1,ny1]); vn=np.zeros([nx1,ny1])
hs=np.zeros([nx1,ny1]);  h=np.zeros([nx1,ny1]); hn=np.zeros([nx1,ny1])
ijh=np.zeros([nx2,ny2],dtype=int)
v_up=np.zeros([nx1,ny1]); hs_up=np.zeros([nx1,ny1]); u_vp=np.zeros([nx1,ny1]); hs_vp=np.zeros([nx1,ny1])
eta=np.zeros([nx1,ny1]); ep=np.zeros([nx1,ny1]); ep_x=np.zeros([nx1,ny1]); usta=np.zeros([nx1,ny1])
u_node=np.zeros([nx1,ny1]); v_node=np.zeros([nx1,ny1])
qu=np.zeros([nx1,ny1]); qv=np.zeros([nx1,ny1]); qc=np.zeros([nx1])

gux=np.zeros([nx1,ny1]); guy=np.zeros([nx1,ny1]); gvx=np.zeros([nx1,ny1]); gvy=np.zeros([nx1,ny1])
gux_n=np.zeros([nx1,ny1]); guy_n=np.zeros([nx1,ny1])
gvx_n=np.zeros([nx1,ny1]); gvy_n=np.zeros([nx1,ny1])
cfx=np.zeros([nx1,ny1]); cfy=np.zeros([nx1,ny1]); qbx=np.zeros([nx1,ny1]); qby=np.zeros([nx1,ny1])
ctrx=np.zeros([nx1,ny1]); ctry=np.zeros([nx1,ny1])
uvis=np.zeros([nx1,ny1]);uvis_x=np.zeros([nx1,ny1]);uvis_y=np.zeros([nx1,ny1]);uvis_c=np.zeros([nx1,ny1])
vvis=np.zeros([nx1,ny1]);vvis_x=np.zeros([nx1,ny1]);vvis_y=np.zeros([nx1,ny1]);vvis_c=np.zeros([nx1,ny1])
fn=np.zeros([nx1,ny1]);gxn=np.zeros([nx1,ny1]);gyn=np.zeros([nx1,ny1])
ux=np.zeros([nx1,ny1]);vx=np.zeros([nx1,ny1]); uv2=np.zeros([nx1,ny1])
hx=np.zeros([nx1,ny1]);hsx=np.zeros([nx1,ny1]);vor=np.zeros([nx1,ny1])

uinp=np.zeros([ny1])

xf=np.zeros([5]);yf=np.zeros([5])

#Initial Bed Elevation
eta=initial.eta_init(eta,nx,ny,ds,dn,zgrid)  
#fig,ax=plt.subplots(figsize = (xsize, ysize))
#im=ax.set_title("Sin-Generated Curve",size='25')
#im=ax.set_xlabel("x(m)",size='20')
#im=ax.set_ylabel("y(m)",size='20')
#im=ax.set_ylim(ymin,ymax)
#im=ax.set_xlim(xmin,xmax)
#im=ax.contourf(xgrid,ygrid,eta,cmap='viridis') 
#plt.show()     
#fig.savefig("ini_eta.png")
#plt.clf(); plt.close()

#縦断形のプロット
pval[:]=zgrid[:,nym] #センター

ysize=xsize*.8
fig,ax=plt.subplots(figsize = (xsize, ysize))
im=ax.set_title("Longitudinal Proile",size='25')
im=ax.set_xlabel("s(m)",size='20')
im=ax.set_ylabel("Elevation(m)",size='20')
im=ax.plot(spos,pval,label='Cnter',linewidth=4)
pval[:]=zgrid[:,0] #右岸
im=ax.plot(spos,pval,label='Right',linewidth=4)
pval[:]=zgrid[:,ny] #左岸
im=ax.plot(spos,pval,label='Left',linewidth=4)


#初期境界条件

hs0=(snm*qp/chb/math.sqrt(slope))**(3./5.)
u0=qp/(hs0*chb)
qu0=u0*hs0*dn[1,nym]

hlstep=0.005 #水深表示のステップ
hlmin=int(hs0*0/hlstep)*hlstep
hlmax=int(hs0*4./hlstep)*hlstep
levels=np.arange(hlmin,hlmax,hlstep)

ulstep=0.05 #流速表示のステップ
ulmin=int(u0*0./ulstep)*ulstep
ulmax=int(u0*3./ulstep)*ulstep
ulevels=np.arange(ulmin,ulmax,ulstep)

vlstep=0.5 #渦度表示のステップ
vlmax=8.; vlmin=-vlmax
vlevels=np.arange(vlmin,vlmax,vlstep)

m=len(vlevels)

vlevels[0]=vlevels[0]-2.
vlevels[m-1]=vlevels[m-1]+2.

#Downstream Uniform Flow Depth
if j_east==0 and j_hdown<=1:
    h_down=uniform.down(nx,ny,dn,eta,qp,snm,hs0,slope)

#Initial Water Surface Elevation along the Center Line
if j_inih==0:
    hpos_c=uniform.h_line(hpos_c,eta,spos_c,h_down,slope,nx,nym,hmin)

elif j_inih==1:
    hpos_c=uniform.h_uniform(hpos_c,nx,ny,dn,eta,qp,snm,hs0,h_down,hmin,slope)
elif j_inih==2:
    hpos_c=uniform.h_nonuni(hpos_c,c_area,e_slope,alf_f,eta,qp,spos_c,hs0,h_down,nx,ny,nym,ds,dn,snm,hmin,g)

hpos_c[0]=hpos_c[1]

#初期水位のプロット
xmin=np.min(spos_c); xmax=np.max(spos_c)
ymin=np.min(eta); ymax=np.max(hpos_c)+hs0
zmin=ymin;zmax=ymax
im=ax.set_ylim(ymin,ymax)
im=ax.set_xlim(xmin,xmax)

im=ax.plot(spos_c,hpos_c,label='Wataer Surface',linewidth=4)


lg=ax.legend(fontsize=20)

fig.savefig("longitudinal_prof.png")
plt.clf(); plt.close()

# Initial Depth and Water Surface Elevation
h,hs=initial.h_init(h,hs,hpos_c,eta,nx,ny,hmin)
h,hs=boundary.h_bound(h,hs,eta,nx,ny,j_west,j_east,j_hdown,h_down,hmin,j_rep,alpha_upu)
hn=copy.copy(h) 

hs_up=stgg.hs_up_c(hs_up,hs,nx,ny)
hs_vp=stgg.hs_vp_c(hs_vp,hs,nx,ny)
#Initial Velocities
u=initial.u_init(u,qu,qc,dn,hs_up,nx,ny,snm,slope,hmin,qp)
uinp,u,qu,qc=boundary.u_upstream(uinp,u,hs,qu,qc,snm,slope,dn,qp,nx,ny,hmin,j_rep,alpha_upu)
un=copy.copy(u)

# Initial Eddy Viscosity
ep,ep_x=initial.ep_init(ep,ep_x,nx,ny,snu_0)

#障害物セルの設定
if j_obst==1:
    if jo_type==1: #障害物ファイルから読み込み
        ijh=obstacle.ob_ini(ijh,nx,ny)  # Setup Obstacle Cells
    elif jo_type==2: #障害物条件入力して設定
        dsmin=np.min(ds)
        igs=max(1,int(dike_thick/dsmin+.5))
        dnmin=np.min(dn)
        jgs=max(1,int(dike_length/dnmin+.5))

        ijh=obstacle.ob_cond(ijh,nx,ny,jo_type,jo_method,igs,jgs,num_dike,dike_dis,f_dike_pos,spos)

#格子+センターラインの描画+障害物

fig,ax=plt.subplots(figsize = (xsize, ysize0))
im=ax.set_title("Sin-Generated Curve",size='25')
im=ax.set_xlabel("x(m)",size='20')
im=ax.set_ylabel("y(m)",size='20')
im=ax.set_ylim(ymin0,ymax0)
im=ax.set_xlim(xmin0,xmax0)

# Center Line Plot

im=ax.plot(xpos,ypos,label='theta0='+str(t0_degree)+' degree',color='red',linewidth=2)

# Grid Plot

for j in np.arange(0,ny+1):
    im=ax.plot(xgrid[:,j],ygrid[:,j],color='black')
for i in np.arange(0,nx+1):
    im=ax.plot(xgrid[i,:],ygrid[i,:],color='black')

for i in np.arange(0,nx+1,nx):
    im=ax.plot(xgrid[nx-1,:],ygrid[nx-1,:],color='green',linewidth=4)
for j in np.arange(0,ny+1,ny):
    im=ax.plot(xgrid[:,j],ygrid[:,j],color='green',linewidth=4)

# Obstacle Plot

for i in np.arange(1,nx+1):
    for j in np.arange(1,ny+1):
        if ijh[i,j]>=0.1:
            xf[0]=xgrid[i-1,j-1];   yf[0]=ygrid[i-1,j-1]
            xf[1]=xgrid[i,j-1];     yf[1]=ygrid[i,j-1]
            xf[2]=xgrid[i,j];       yf[2]=ygrid[i,j]
            xf[3]=xgrid[i-1,j];     yf[3]=ygrid[i-1,j]
            xf[4]=xgrid[i-1,j-1];   yf[4]=ygrid[i-1,j-1]
            im=ax.fill(xf,yf,color = "blue")

#lg=ax.legend(fontsize=20)
fig.savefig("grid.png")
plt.clf(); plt.close()

u=boundary.u_bound(u,nx,ny,j_west,j_east,ijh,uinp,hs,hmin,j_rep); un=u
v=boundary.v_bound(v,nx,ny,ijh,j_rep)       ; vn=v



qu,qc=rhs.qu_cal(qu,qc,u,nx,ny,dn,hs_up,hmin,ijh)
qv=rhs.qv_cal(qv,v,nx,ny,ds,hs_vp,hmin,ijh)


#gux,guy,gvx,gvy=initial.diffs_init(gux,guy,gvx,gvy,u,v,nx,ny,dx,dy)
#gux,guy=boundary.gbound_u(gux,guy,ijh,nx,ny)
#gvx,gvy=boundary.gbound_v(gvx,gvy,ijh,nx,ny)
#print(gux)

u_vp=stgg.u_vp_c(u_vp,u,nx,ny)
hs_vp=stgg.hs_vp_c(hs_vp,hs,nx,ny)

v_up=stgg.v_up_c(v_up,v,nx,ny)
hs_up=stgg.hs_up_c(hs_up,hs,nx,ny)
#print(hs_up)

time=0.
icount=0



nfile=0
os.system("del /Q .\\png\\*.png")
os.system("del /Q .\\png_q\\*.png"); os.system("del /Q .\\png_r\\*.png")


#iskip=2; jskip=2
l=0
########### Main ############

while time<= etime:
    usta,ep,ep_x=fric.us_cal(usta,ep,ep_x,u,v,hs,nx,ny,snm,g_sqrt,hmin,ep_alpha)
    if icount%it_out==0 and time>stime:
        print('time=',np.round(time,3),l)
#        print(np.round(qc[0:20],3))
#        print('qc=',qc)
        nfile=nfile+1
#        print(np.round(vn,5))

# Contour and Vector
        ux,vx,uv2,hx,hsx=uxvx.uv(ux,vx,uv2,hx,hsx,u,v,h,hs,nx,ny,coss,sins)
        vor=uxvx.vortex(vor,ux,vx,nx,ny,ds,dn)

        fig,ax=plt.subplots(figsize = (xsize, ysize0))
        im=ax.set_ylim(ymin0,ymax0)
        im=ax.set_xlim(xmin0,xmax0)

        cont=ax.contourf(xgrid[0:nx,:], ygrid[0:nx,:], hsx[0:nx,:],levels, cmap='viridis') 
#        cont=ax.contourf(xgrid, ygrid, uv2,ulevels) 
#        cont=ax.contourf(xgrid, ygrid, vor, vlevels, cmap='coolwarm') 
#        cont=ax.contourf(xgrid, ygrid, uv2) 
        vect=ax.quiver(xgrid[0:nx:iskip,0:nx:jskip], ygrid[0:nx:iskip,0:nx:jskip], ux[0:nx:iskip,0:nx:jskip], vx[0:nx:iskip,0:nx:jskip], \
           scale=jscale, width=0.001,headwidth=1,color=vcolor)

        cb=fig.colorbar(cont)
        cb.set_label('Depth(m)', size=30)

        im=ax.set_title("Depth and Velocity Vecotors",size='30')
#        im=ax.set_title("Stream Line",size='30')

        plt.subplots_adjust(left=xset1, right=xset2, bottom=yset1, top=yset2)
        for i in np.arange(1,nx+1):
            for j in np.arange(1,ny+1):
                if ijh[i,j]>=0.1:
                    xf[0]=xgrid[i-1,j-1];   yf[0]=ygrid[i-1,j-1]
                    xf[1]=xgrid[i,j-1];     yf[1]=ygrid[i,j-1]
                    xf[2]=xgrid[i,j];       yf[2]=ygrid[i,j]
                    xf[3]=xgrid[i-1,j];     yf[3]=ygrid[i-1,j]
                    xf[4]=xgrid[i-1,j-1];   yf[4]=ygrid[i-1,j-1]
                    ax.fill(xf,yf,color = "red")
        ax.text(xmin0+xlen0*.8,ymin0+ylen0*0.95,"Time="+str(np.round(time,3))+"sec",size=25)
        
        fname="./png/" + 'f%04d' % nfile + '.png'
#        print(fname)
#        plt.show()
        im=plt.savefig(fname)
        plt.clf()
        plt.close()

# 流量縦断と流速縦断のプロット
        ysize=xsize*1.5
        fig=plt.figure(figsize = (xsize, ysize))

        ax1=fig.add_subplot(4,1,1)
        im1=ax1.set_title("Longitudinal Discharge Proile",size='35')
        xvalues=np.arange(-1,int(spos[nx]+1.))

        im1=ax1.set_ylim(qmin,qmax)
        im1=ax1.set_xlabel("s(m)",size='30')
        im1=ax1.set_ylabel("Discharge(m3/s)",size='30')
        im1=ax1.set_xticklabels(xvalues,fontsize=25)
        for tick in ax1.get_yticklabels():
            tick.set(fontsize=25)

        im1=ax1.plot(spos,qc,label='Descahrge',linewidth=4,color='green')
    #    ax1.legend(fontsize=30)
        ax1.text(0.,qc[0],"Discharge",size=30)
        ax1.text(0.,qmin+(qmax-qmin)*0.9,"Time="+str(np.round(time,3))+"sec",size=30)
# 流速縦断
        ax5=ax1.twinx() #右軸
        im5=ax5.set_ylim(ulmin,ulmax)
        im5=ax5.set_ylabel("Velocoty(m/s)",size='30')
        im5=ax5.plot(spos,u[0:nx+1,nym],label='Velocity',linewidth=4,color='red')
        ax5.text(0.,u[0,nym],"Velocity",size=30)
#        ax5.legend(fontsize=30)

# 水位(センター)
#        ax2=ax1.twinx() #y軸右選択
        ax2=fig.add_subplot(4,1,2)
        im2=ax2.set_ylim(zmin,zmax)
        im2=ax2.set_xlabel("s(m)",size='30')
        im2=ax2.set_ylabel("Elevation(m)",size='30')

        im2=ax2.set_xticklabels(xvalues,fontsize=25)
        for tick in ax2.get_yticklabels():
            tick.set(fontsize=25)
        im2=ax2.plot(spos,hx[:,nym],label='Water Surface (Center)',linewidth=4,color='blue')
        im2=ax2.plot(spos,zgrid[:,nym],label='z-center',linewidth=4,color='black')
#        im2=ax2.plot(spos_c,h[:,nym],label='Water Surface (Center)',linewidth=4,color='blue')
#        im2=ax2.plot(spos_c,eta[:,nym],label='z-center',linewidth=4,color='black')
        ax2.legend(fontsize=30)

# 水位(右)
        ax3=fig.add_subplot(4,1,3)
        im3=ax3.set_ylim(zmin,zmax)
        im3=ax3.set_xlabel("s(m)",size='30')
        im3=ax3.set_ylabel("Elevation(m)",size='30')
        im3=ax3.set_xticklabels(xvalues,fontsize=25)
        for tick in ax3.get_yticklabels():
            tick.set(fontsize=25)
        im3=ax3.plot(spos,hx[:,0],label='Water Surface (Right)',linewidth=4,color='blue')
        im3=ax3.plot(spos,zgrid[:,0],label='bed-right',linewidth=4,color='black')
        ax3.legend(fontsize=30)
# 水位(左)
        ax4=fig.add_subplot(4,1,4)
        im4=ax4.set_ylim(zmin,zmax)
        im4=ax4.set_xlabel("s(m)",size='30')
        im4=ax4.set_ylabel("Elevation(m)",size='30')
        im4=ax4.set_xticklabels(xvalues,fontsize=25)
        for tick in ax4.get_yticklabels():
            tick.set(fontsize=25)
        im4=ax4.plot(spos,hx[:,ny],label='Water Surface (Left)',linewidth=4,color='blue')
        im4=ax4.plot(spos,zgrid[:,ny],label='bed-left',linewidth=4,color='black')
        ax4.legend(fontsize=30)


        fname="./png_q/" + 'q%04d' % nfile + '.png'
#        print(fname)
        fig.savefig(fname)
#        plt.show()
        plt.clf(); plt.close()

# 流路の曲率と流線の曲率

        ysize=xsize*.5
        fig,ax=plt.subplots(figsize = (xsize, ysize))
        im=ax.set_ylim(ymin_r,ymax_r)
        im=ax.set_xlim(xmin0,xmax0)
        im=ax.set_title("Curvature",size='25')
        im=ax.set_xlabel("s(m)",size='20')
        im=ax.set_ylabel("1/r(1/m)",size='20')
        im=ax.plot(spos,rho_r[:,nym],label='1/r',linewidth=4)
        
        u_node,v_node=ros.uv_node(u_node,v_node,u,v,nx,ny)
        rhos_r=ros.roscal(nx,ny,ds,dn,rhos_r,u_node,v_node)
        im=ax.plot(spos,rho_r[:,nym]+rhos_r[:,nym],label='1/rs',linewidth=4)
        ax.legend(fontsize=30)
        ax.text(0.,ymin_r+(ymax_r-ymin_r)*0.9,"Time="+str(np.round(time,3))+"sec",size=30)

        fname="./png_r/" + 'r%04d' % nfile + '.png'
#        print(fname)
        fig.savefig(fname)
#        plt.show()
        
        plt.clf(); plt.close()
# 

#Velocities in Non Advection Phase
    l=0
    while l<lmax:

        v_up=stgg.v_up_c(v_up,vn,nx,ny)
        hs_up=stgg.hs_up_c(hs_up,hs,nx,ny)
        cfx=cfxy.cfxc(cfx,nx,ny,hs,un,g,snm,v_up,hs_up,hmin)
        ctrx=cfxy.centrix(ctrx,nx,ny,un,v_up,rho_s)
        un=rhs.un_cal(un,u,nx,ny,dsi,cfx,ctrx,hn,g,dt,hmin,hs,eta,ijh)
        un=boundary.u_bound(un,nx,ny,j_west,j_east,ijh,uinp,hs,hmin,j_rep)
        qu,qc=rhs.qu_cal(qu,qc,un,nx,ny,dn,hs_up,hmin,ijh)
    
        u_vp=stgg.u_vp_c(u_vp,un,nx,ny)
        hs_vp=stgg.hs_vp_c(hs_vp,hs,nx,ny)
        cfy=cfxy.cfyc(cfy,nx,ny,hs,vn,g,snm,u_vp,hs_vp,hmin)
        ctry=cfxy.centriy(ctry,nx,ny,u_vp,rho_n)
        vn=rhs.vn_cal(vn,v,nx,ny,dnj,cfy,ctry,hn,g,dt,hmin,hs,eta,ijh)
        vn=boundary.v_bound(vn,nx,ny,ijh,j_rep)
        qv=rhs.qv_cal(qv,vn,nx,ny,ds,hs_vp,hmin,ijh)

        hn,hs,err=hcal.hh(hn,h,hs,eta,qu,qv,ijh,area,alh,hmin,nx,ny,dt)
        hn,hs=boundary.h_bound(hn,hs,eta,nx,ny,j_west,j_east,j_hdown,h_down,hmin,j_rep,alpha_upu)

        if err<errmax:
            break
        l=l+1

#Diffusion 

    un=diffusion.diff_u(un,uvis,uvis_x,uvis_y,nx,ny,ds,dsi,dn,dt,ep,ep_x,cw,uvis_c,rho_s)
    un=boundary.u_bound(un,nx,ny,j_west,j_east,ijh,uinp,hs,hmin,j_rep)
    vn=diffusion.diff_v(vn,vvis,vvis_x,vvis_y,nx,ny,ds,dn,dnj,dt,ep,ep_x,vvis_c,rho_n)
    vn=boundary.v_bound(vn,nx,ny,ijh,j_rep)

#Differentials in Non Advection Phase
    gux,guy=newgrd.ng_u(gux,guy,u,un,nx,ny,dsi,dn)
    gux,guy=boundary.gbound_u(gux,guy,ijh,nx,ny,j_rep)
    gvx,gvy=newgrd.ng_v(gvx,gvy,v,vn,nx,ny,ds,dnj)
    gvx,gvy=boundary.gbound_v(gvx,gvy,ijh,nx,ny,j_rep)

#Advection Phase
    fn,gxn,gyn=mkzero.z0(fn,gxn,gyn,nx,ny)
    v_up=stgg.v_up_c(v_up,v,nx,ny) 
    fn,gxn,gyn=cip2d.u_cal1(un,gux,guy,u,v_up,fn,gxn,gyn,nx,ny,dsi,dn,dt)
    un,gux,guy=cip2d.u_cal2(fn,gxn,gyn,u,v_up,un,gux,guy,nx,ny,dsi,dn,dt)
    un=boundary.u_bound(un,nx,ny,j_west,j_east,ijh,uinp,hs,hmin,j_rep)
    gux,guy=boundary.gbound_u(gux,guy,ijh,nx,ny,j_rep)

    fn,gxn,gyn=mkzero.z0(fn,gxn,gyn,nx,ny)
    u_vp=stgg.u_vp_c(u_vp,u,nx,ny)
    fn,gxn,gyn=cip2d.v_cal1(vn,gvx,gvy,u_vp,v,fn,gxn,gyn,nx,ny,ds,dnj,dt)
    vn,gvx,gvy=cip2d.v_cal2(fn,gxn,gyn,u_vp,v,vn,gvx,gvy,nx,ny,ds,dnj,dt)
    vn=boundary.v_bound(vn,nx,ny,ijh,j_rep)
    gvx,gvy=boundary.gbound_v(gvx,gvy,ijh,nx,ny,j_rep)

    uinp,un,qu,qc=boundary.u_upstream(uinp,un,hs,qu,qc,snm,slope,dn,qp,nx,ny,hmin,j_rep,alpha_upu)
    un,qu,gux,guy=dryck.u_ck(un,qu,hs,hmin,gux,guy,nx,ny,ijh)
    vn,qv,gvx,gvy=dryck.v_ck(vn,qv,hs,hmin,gvx,gvy,nx,ny,ijh)
    qc=rhs.qc_cal(qu,qc,nx,ny)
    h=copy.copy(hn); u=copy.copy(un); v=copy.copy(vn)

#流線の曲率の計算
    u_node,v_node=ros.uv_node(u_node,v_node,u,v,nx,ny)
    rhos_r=ros.roscal(nx,ny,ds,dn,rhos_r,u_node,v_node)

#Time Step Update
    time=time+dt
    icount=icount+1


os.system("del /Q *.mp4")
os.system("del /Q *.gif")

subprocess.call('ffmpeg -framerate 30 -i png/f%4d.png -r 60 -an -vcodec libx264 -pix_fmt yuv420p vec.mp4',  shell=True)
os.system('ffmpeg -i vec.mp4 vec.gif -loop 0')

subprocess.call('ffmpeg -framerate 30 -i png_q/q%4d.png -r 60 -an -vcodec libx264 -pix_fmt yuv420p longi.mp4', shell=True)
os.system("ffmpeg -i longi.mp4 longi.gif -loop 0")

subprocess.call('ffmpeg -framerate 30 -i png_r/r%4d.png -r 60 -an -vcodec libx264 -pix_fmt yuv420p radius.mp4', shell=True)
os.system("ffmpeg -i radius.mp4 radius.gif -loop 0")

