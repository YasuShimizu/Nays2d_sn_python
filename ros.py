from numba.types.npytypes import NumpyNdIterType
import numpy as np
from numba import jit

@jit(nopython=True)
def uv_node(u_node,v_node,u,v,nx,ny):
    for i in np.arange(0,nx+1):
        for j in np.arange(1,ny):
            u_node[i,j]=(u[i,j]+u[i,j+1])*.5
        u_node[i,0]=u[i,1]; u_node[i,ny]=u[i,ny]
    
    
    for j in np.arange(0,ny+1):
        for i in np.arange(1,nx):    
            v_node[i,j]=(v[i,j]+v[i+1,j])*.5
        v_node[0,j]=v[1,j]; v_node[nx,j]=v[nx,j]
    return u_node,v_node

@jit(nopython=True)
def roscal(nx,ny,ds,dn,rhos_r,u_node,v_node):
    for i in np.arange(1,nx):
        for j in np.arange(1,ny):
            u2=u_node[i,j]**2; v2=v_node[i,j]**2; uv=u_node[i,j]*v_node[i,j]
            u2v2=u2+v2
            if u2v2>1e-7:
                v3=(u2v2)**(3./2.)
                duds=(u_node[i+1,j]-u_node[i-1,j])/(ds[i+1,j]+ds[i,j])
                dudn=(u_node[i,j+1]-u_node[i,j-1])/(dn[i,j+1]+dn[i,j])
                dvds=(v_node[i+1,j]-v_node[i-1,j])/(ds[i+1,j]+ds[i,j])
                dvdn=(v_node[i,j+1]-v_node[i,j-1])/(dn[i,j+1]+dn[i,j])
                rhos_r[i,j]=(u2*dvds-v2*dudn+uv*(dvdn-duds))/v3
            else:
                rhos_r[i,j]=0.
    return rhos_r
