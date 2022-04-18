///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	Copyright (c) 2008, State Key Laboratory of Multi-Phase Complex System
//					Institute of Process Engineering (IPE)
//					Chinese Academy of Sciences (CAS)
//	All rights reserved.
//	Author: Shuli Shu
//	Name: Three-dimensional Lattice Boltzmann Method on CUDA for single phase 
// 	Parallel version, space decomosion in three dimensions with boundary lattice 
//	transfered after collosion, no overlapped computation (referred to Bo Li's 
//	parallel implementation based on my original parallel version)
//	Abstract: MRT scheme with D3Q19 for Shell to compute the permeablity of porous media)
//	The distribution function is in 3D array to reduce the number of parameters in LBCollProp
//	Version: 1.0
//	Date: 2013/1/26
// refer to implementation of Jonas Tolke "Towards 3D teraflop CFD computing on a desktop PC using graphic Hardware",
// in International journal of computational fluid dynamics,vol(22),Issue 7, 2008,page 443-456.
// the block is full of lattice in X direction,withe grid partitioned along Y and Z grid
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//includes, system
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
using namespace std;

/* the following part is added for time statistics */
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>
#include <cutil.h>

#define NDIM 3
#define THREAD_NUM 256		//number of thread in block
#define GPU_NUM 6
#define GPU_BGNID 0
#define COMPONENT_NUM 1
#define DIRECTION_NUM 19
// The following part is got from the 
char initinputPath[30]="Init/", FlowPath[30]="Flow/",StreamPath[30]="Str/", ParaPath[30]="Para/";
char InitputPath[60],Flowtmppath[60],strpath[60],paratmppath[60],pararotornodepath[60],parawallnodepath[60];

#define FLUID 0
#define SOLID 1
#define SETU 2
#define ANOU 3
#define WALL 4

struct timeval tv;
struct timezone tz;

double totaltime=0, colltime=0, propagatetime=0, commtime=0;
double time0,time1;
/* global variables */
int nx, ny, nz;		        /* number of real grid nodes in each direction */
int parallelnx, parallelny, parallelnz;  /* number of grid nodes in each direction */
int tbegin, tend, postproc_intervall;	
float meshnum,inletsize,outletsize,inletp[3],outletp[3];
float *conin_h;
float *disf;

int x, y, z, k;
int NR,NW;
unsigned int *h_geoD,*h_geoDT;
unsigned int *geoD,*geoDT;
float *f0_d, *f1_d;
float *ux_d,*uy_d,*uz_d,*rho_h,*con_h,*vt_h;
float *nodeR_d,*nodeW_d,*Fw_d,*Fr_d,*Fsf;
float *ux_h,*uy_h,*uz_h,*rho_d,*con_d,*vt;
float *nodeR_h,*nodeW_h;
float *M_h,*MI_h;

size_t pitch;
int totallatticenum, reallatticenum;
int pitchnum, diffnum;
int xstartoff, ystartoff, zstartoff; // offset in each dimension for parallel version

//MPI parameters
int h_procn[NDIM], h_numofproc=1;//process number
int h_procid, procpos[NDIM];//id and coord of the process
int h_nbprocid[NDIM][2];//id of neighbour process

float *h_sendbuf, *h_recvbuf; 
float *latticebuf;

int h_GlobalLatticeNum[NDIM], baseLatticeNum[NDIM];
int h_procLatNum[NDIM];
int *d_GlobalLatticeNum, *d_baseLatticeNum;

float rho_0;
float U0_h; //rotational speed
float S_h[19];
float sct_h,h_bodyf;
__constant__ float U0;
__constant__ float rho0_d;
__constant__ float S_d[19];
__constant__ float con_in;
__constant__ float sct;
__constant__ float bodyf;

MPI_Status status;

int gpu_bgnid, gpu_num;
float visocity,latticelen;

__host__ void init()   /* initialise distributions for u=v=0 */
{
  int k1;	
  for(z = 0 ; z< parallelnz ; z++)
  {
   for(y = 0; y < parallelny ; y++)
   {
    for( x = 0 ; x < parallelnx ; x++)// 
    {
	 k = z*parallelnx*parallelny + y*parallelnx + x;
	   
	 disf[k] = rho_0/3.0;
	 disf[pitchnum+k] = disf[2*pitchnum+k] = disf[3*pitchnum+k] = disf[4*pitchnum+k] = disf[5*pitchnum+k] = disf[6*pitchnum+k] = rho_0/18.0;
	   
	 disf[7*pitchnum+k] = disf[8*pitchnum+k] = disf[9*pitchnum+k] = disf[10*pitchnum+k] = disf[11*pitchnum+k] = disf[12*pitchnum+k]=
	 disf[13*pitchnum+k] = disf[14*pitchnum+k] = disf[15*pitchnum+k] = disf[16*pitchnum+k] = disf[17*pitchnum+k] = disf[18*pitchnum+k]
	   = rho_0/36.0;
	   
	     rho_h[k]=rho_0; 
     	 ux_h[k]=0.0;
     	 uy_h[k]=0.0;
     	 uz_h[k]=0.0;
	     vt_h[k]=0.0;	 
     }
   }	 
  }
 int e[DIRECTION_NUM][NDIM]={{ 0, 0, 0},
			{ 1, 0, 0},{-1, 0, 0},{ 0, 1, 0},{ 0,-1, 0},{ 0, 0, 1},{ 0, 0, -1},
			{ 1, 1, 0},{-1, 1, 0},{-1,-1, 0},{ 1,-1, 0},
                        { 0, 1, 1},{ 0,-1, 1},{ 0,-1,-1},{ 0, 1,-1},
			{ 1, 0, 1},{ 1, 0,-1},{-1, 0,-1},{-1, 0, 1}};

 for(x=0;x<DIRECTION_NUM;x++)
 {
	float temp=e[x][0]*e[x][0]+e[x][1]*e[x][1]+e[x][2]*e[x][2];
	k=x+0*DIRECTION_NUM;
	M_h[k]=1;	
	k=x+1*DIRECTION_NUM;
	M_h[k]=-30+19*temp;
	k=x+2*DIRECTION_NUM;
	M_h[k]=0.5*(24-53*temp+21*temp*temp);
	k=x+3*DIRECTION_NUM;
	M_h[k]=e[x][0];
	k=x+4*DIRECTION_NUM;
	M_h[k]=e[x][0]*(5*temp-9);
	k=x+5*DIRECTION_NUM;
	M_h[k]=e[x][1];
	k=x+6*DIRECTION_NUM;
	M_h[k]=e[x][1]*(5*temp-9);
	k=x+7*DIRECTION_NUM;
	M_h[k]=e[x][2];
	k=x+8*DIRECTION_NUM;
	M_h[k]=e[x][2]*(5*temp-9);
	k=x+9*DIRECTION_NUM;
	M_h[k]=3*e[x][0]*e[x][0]-temp;
	k=x+10*DIRECTION_NUM;
	M_h[k]=(3*temp-5)*(3*e[x][0]*e[x][0]-temp); 
	k=x+11*DIRECTION_NUM;
	M_h[k]=e[x][1]*e[x][1]-e[x][2]*e[x][2];
	k=x+12*DIRECTION_NUM;
	M_h[k]=(3*temp-5)*(e[x][1]*e[x][1]-e[x][2]*e[x][2]); 
	k=x+13*DIRECTION_NUM;
	M_h[k]=e[x][0]*e[x][1];
	k=x+14*DIRECTION_NUM;
	M_h[k]=e[x][1]*e[x][2];
	k=x+15*DIRECTION_NUM;
	M_h[k]=e[x][0]*e[x][2];
	k=x+16*DIRECTION_NUM;
	M_h[k]=(e[x][1]*e[k][1]-e[x][2]*e[x][2])*e[x][0];
	k=x+17*DIRECTION_NUM;
	M_h[k]=(e[x][2]*e[k][2]-e[x][0]*e[x][0])*e[x][1];
	k=x+18*DIRECTION_NUM;
	M_h[k]=(e[x][0]*e[k][0]-e[x][1]*e[x][1])*e[x][2];
 }
 for(x=0;x<DIRECTION_NUM;x++)
 {
	 for(y=0;y<DIRECTION_NUM;y++)
	 {
		 MI_h[x+y*DIRECTION_NUM]=M_h[y+x*DIRECTION_NUM];
	 }
 }
 float temp;
 
 for(x=0;x<DIRECTION_NUM;x++)
 {
	 temp=0.0;
	 for(y=0;y<DIRECTION_NUM;y++)
	 {
		 temp+=MI_h[x+y*DIRECTION_NUM];
	 }
	 for(y=0;y<DIRECTION_NUM;y++)
	 {
		 MI_h[x+y*DIRECTION_NUM]/=temp;
	 }
 }
} 

__host__ void init_geo()
{
	int i,j,k;	
	int k1;
	
	for(z = zstartoff; z < parallelnz - zstartoff; z++) // read data in three dimensions
	{
	  for(y = ystartoff; y < parallelny - ystartoff; y++)
	  {
		for( x = xstartoff; x < parallelnx - xstartoff; x++)
		{
			k1  = z*parallelnx*parallelny + y*parallelnx + x;
			i=x+baseLatticeNum[0]-xstartoff;
			j=y+baseLatticeNum[1]-ystartoff;
			k=z+baseLatticeNum[2]-zstartoff;
			
			if(i*(i-h_GlobalLatticeNum[0]+1)<0&&j*(j-h_GlobalLatticeNum[1]+1)<0&&k*(k-h_GlobalLatticeNum[2]+1)<0)
			{           
			    h_geoD[k1]=FLUID;
			}
			else
			{
				h_geoD[k1]=SOLID;

			}
		}
	  }
	}
	
	for(z = 0; z < parallelnz+2; z++) // read data in three dimensions
	{
	  for(y = 0; y < parallelny +2; y++)
	  {
		for( x = 0; x < parallelnx+2; x++)
		{
			k1  = z*(parallelnx+2)*(parallelny+2) + y*(parallelnx+2) + x;

			i=x+baseLatticeNum[0];
			j=y+baseLatticeNum[1];
			k=z+baseLatticeNum[2];

			if((i-2)*(i-2-h_GlobalLatticeNum[0]+1)<0&&(j-2)*(j-2-h_GlobalLatticeNum[1]+1)<0&&(k-2)*(k-2-h_GlobalLatticeNum[2]+1)<0)
			{
				h_geoDT[k1]=FLUID;
			}
			
			else
			{
				h_geoDT[k1]=SOLID;
			}
		}
	  }
	}
	
	for(z = zstartoff; z < parallelnz - zstartoff; z++) // read data in three dimensions
	{
	  for(y = ystartoff; y < parallelny - ystartoff; y++)
	  {
		for( x = xstartoff; x < parallelnx - xstartoff; x++)
		{
      
			k  = z*parallelnx*parallelny + y*parallelnx + x;
			
			int k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15,k16,k17,k18;
			k1=(z+1)*(parallelnx+2)*(parallelny+2) + (y+1)*(parallelnx+2) + (x+1)+1;
			k2=(z+1)*(parallelnx+2)*(parallelny+2) + (y+1)*(parallelnx+2) + (x+1)-1;
			
			k3=(z+1)*(parallelnx+2)*(parallelny+2) + (y+1+1)*(parallelnx+2) + (x+1);
			k4=(z+1)*(parallelnx+2)*(parallelny+2) + (y+1-1)*(parallelnx+2) + (x+1);
			
			k5=(z+1+1)*(parallelnx+2)*(parallelny+2) + (y+1)*(parallelnx+2) + (x+1);
			k6=(z+1-1)*(parallelnx+2)*(parallelny+2) + (y+1)*(parallelnx+2) + (x+1);
			
			k7=(z+1)*(parallelnx+2)*(parallelny+2) + (y+1+1)*(parallelnx+2) + x+1+1;
			k8=(z+1)*(parallelnx+2)*(parallelny+2) + (y+1+1)*(parallelnx+2) + x-1+1;
			k9=(z+1)*(parallelnx+2)*(parallelny+2) + (y-1+1)*(parallelnx+2) + x-1+1;
			k10=(z+1)*(parallelnx+2)*(parallelny+2) + (y-1+1)*(parallelnx+2) + x+1+1;
			
			k11=(z+1+1)*(parallelnx+2)*(parallelny+2) + (y+1+1)*(parallelnx+2) + x+1;
			k12=(z+1+1)*(parallelnx+2)*(parallelny+2) + (y+1-1)*(parallelnx+2) + x+1;
			k13=(z+1-1)*(parallelnx+2)*(parallelny+2) + (y+1-1)*(parallelnx+2) + x+1;
			k14=(z+1-1)*(parallelnx+2)*(parallelny+2) + (y+1+1)*(parallelnx+2) + x+1;
			
			k15=(z+1+1)*(parallelnx+2)*(parallelny+2) + (y+1)*(parallelnx+2) + x+1+1;
			k16=(z+1-1)*(parallelnx+2)*(parallelny+2) + (y+1)*(parallelnx+2) + x+1+1;
			k17=(z+1-1)*(parallelnx+2)*(parallelny+2) + (y+1)*(parallelnx+2) + x+1-1;
			k18=(z+1+1)*(parallelnx+2)*(parallelny+2) + (y+1)*(parallelnx+2) + x+1-1;
			
			if(h_geoD[k]==SOLID)
			{
			  if(h_geoDT[k1]*h_geoDT[k2]*h_geoDT[k3]*h_geoDT[k4]*h_geoDT[k5]*h_geoDT[k6]*h_geoDT[k7]*h_geoDT[k8]*h_geoDT[k9]*h_geoDT[k10]
			     *h_geoDT[k11]*h_geoDT[k12]*h_geoDT[k13]*h_geoDT[k14]*h_geoDT[k15]*h_geoDT[k16]*h_geoDT[k17]*h_geoDT[k18]==0)
			   h_geoD[k]=WALL;
			}
		}
	}
}
}

/********************The following is added for parallel version**************************/
// read inital parameters from parameter.ini
__host__ void readparameter()
{
	//read parameters from parameter.ini
	FILE *inpara;
	char dump[100];
	int i;
	float d,h;
	inpara = fopen(strcat(strcpy(paratmppath, ParaPath), "parameter.ini"), "r");	
	
	for(i=0; i<NDIM; i++)
	{
	  fscanf(inpara, "%s", dump);
	}

	fscanf(inpara, "%f", &d);
	fscanf(inpara, "%f", &h);
	
	fscanf(inpara, "%s", dump);
	fscanf(inpara, "%f", &meshnum);
	
	h_GlobalLatticeNum[0]=int(meshnum*d);
	h_GlobalLatticeNum[1]=int(meshnum*d);
	h_GlobalLatticeNum[2]=int(meshnum*h); 
	
	fscanf(inpara, "%s%s%s%s", dump,dump,dump,dump);
	fscanf(inpara, "%f%f%f%f", &inletsize,&inletp[0],&inletp[1],&inletp[2]);
	fscanf(inpara, "%s%s%s%s", dump,dump,dump,dump);
	fscanf(inpara, "%f%f%f%f", &outletsize,&outletp[0],&outletp[1],&outletp[2]);
	inletsize=inletsize*meshnum;
	outletsize=outletsize*meshnum;
	for(i=0;i<NDIM;i++)
	{
		inletp[i]=inletp[i]*meshnum;
		outletp[i]=outletp[i]*meshnum;
	}
	
	fscanf(inpara, "%s%s%s", dump,dump,dump);
	fscanf(inpara, "%d%d%d", &tbegin, &tend, &postproc_intervall);
	
	for(i=0; i<NDIM; i++)
	{
	  fscanf(inpara, "%s", dump);
	}
	
	for(i=0; i<NDIM; i++)
	{
	  fscanf(inpara, "%d", &h_procn[i]); //read number of process in each dimension
	}
	
	fscanf(inpara, "%s%s%s%s%s%s%s%s", dump,dump,dump,dump,dump,dump,dump,dump);
	fscanf(inpara, "%d%d%f%f%f%f%f%f", &gpu_bgnid, &gpu_num, &visocity, &h_bodyf,&U0_h,&rho_0,&conin_h,&sct_h);
	
        cout<<gpu_num<<endl;
	fscanf(inpara, "%s%s", dump,dump);
	fscanf(inpara, "%d%d",&NR,&NW);
	fscanf(inpara, "%s", dump);
	for(i=0;i<DIRECTION_NUM;i++)
	{
	  fscanf(inpara, "%f", &S_h[i]); 
	}  
	S_h[9]=3*visocity+0.5;
	S_h[11]=S_h[9];
	S_h[13]=S_h[9];
	S_h[14]=S_h[9];
	S_h[15]=S_h[9];
	
	fclose(inpara);
}
__host__ void readnodes()
{
	//read parameters from parameter.ini
	FILE *rotornodes,*wallnodes;
	rotornodes = fopen(strcat(strcpy(pararotornodepath,ParaPath), "rotornodes.ini"), "r");	
	wallnodes =  fopen(strcat(strcpy(parawallnodepath,ParaPath), "wallnodes.ini"), "r");	
	int temp;
	double Pi=3.1415926;
	for(int i=0;i<NW;i++)
	{
		fscanf(wallnodes, "%d%f%f%f", &temp,&nodeW_h[NDIM*i+0],&nodeW_h[NDIM*i+1],&nodeW_h[NDIM*i+2]);
		nodeW_h[NDIM*i+0]+=0.5*h_GlobalLatticeNum[0];
		nodeW_h[NDIM*i+1]+=0.5*h_GlobalLatticeNum[1];
		
	}

	for(int i=0;i<NR;i++)
	{
		fscanf(rotornodes, "%d%f%f%f", &temp,&nodeR_h[NDIM*i+0],&nodeR_h[NDIM*i+1],&nodeR_h[NDIM*i+2]);
	}
	
	for(int i=0;i<NR;i++)
	{
		double d=0.0,angle=0.0;
		if(nodeR_h[NDIM*i+1]>=0)
		{
			d=sqrt(nodeR_h[NDIM*i]*nodeR_h[NDIM*i]+nodeR_h[NDIM*i+1]*nodeR_h[NDIM*i+1]);
			angle=acos(nodeR_h[NDIM*i]/d);
		}
		else
		{
			d=sqrt(nodeR_h[NDIM*i]*nodeR_h[NDIM*i]+nodeR_h[NDIM*i+1]*nodeR_h[NDIM*i+1]);
			angle=2.*Pi-acos(nodeR_h[NDIM*i]/d);
		}
		nodeR_h[NDIM*i]=d;
		nodeR_h[NDIM*i+1]=angle;
	}
}
//Setup process position, borders and its neighbor processes.
__host__ void setup_parallel()
{
	int nbprocpos[NDIM];

	//process coordinate
	int myidtemp = h_procid;
	int i;
	
	for (i=0; i<NDIM; i++)
	{
		procpos[i] = myidtemp%h_procn[i];
		nbprocpos[i] = procpos[i];
		myidtemp /= h_procn[i];
	}

	//id of neighbor process for nonperiodic boundary conditions
	for (i=0; i<NDIM; i++)// non-period boundary conditions in the other two
	{
		if(procpos[i] > 0 )// left side
		{
		  nbprocpos[i] = (procpos[i]-1)%h_procn[i];
		  h_nbprocid[i][1] = nbprocpos[0] + nbprocpos[1]*h_procn[0]+ nbprocpos[2]*h_procn[0]*h_procn[1];
		}else
		{
		  h_nbprocid[i][1] = MPI_PROC_NULL; //
		}
		
		if(procpos[i] < h_procn[i]-1)
		{  
		  nbprocpos[i] = (procpos[i]+1)%h_procn[i];
		  h_nbprocid[i][0] = nbprocpos[0] + nbprocpos[1]*h_procn[0]+ nbprocpos[2]*h_procn[0]*h_procn[1];
		}else
		{
		  h_nbprocid[i][0] = MPI_PROC_NULL;
		}  	  

		nbprocpos[i] = procpos[i];
	}
}
/******************Velocity boundary and presure outlet of Guo*********************/
__global__ void disfBorders_inlet(int nx,int ny,int nz,float* fd,float* rho_d,float* ux_d,float *uy_d,float *uz_d,int pitch,unsigned int* geoD,int* baselatticenum,int* globallatticenum)
{
	//number of threads in this function
	int num_threads = blockDim.x;
	// Block index y
	int by = blockIdx.x;
	// Block index z
	int bz = blockIdx.y;
	// local thread index
	int tx = threadIdx.x;
	
	int ystart,xstart,zstart;
	int sendstart;
	int k,ktemp;
	int pitnum;
	float temp,square;
	float ux,uy,uz;
	
	float Bfeq,Ffeq;
	pitnum = pitch/sizeof(float);
	
	zstart = bz;
	ystart = by*num_threads + tx;
	xstart = 1; //dir=1, xstart= nx-1; dir=0,xstartx=0;
	
	k= zstart*nx*ny + ystart*nx + xstart;
	
	if(ystart<ny)
	{
		if(geoD[k]==SETU)
		{
		Bfeq=1.0/18.0*rho_d[k+1]*(1+3.0*U0+4.5*U0*U0-1.5*U0*U0);
		ux=ux_d[k+1];
		uy=uy_d[k+1];
		uz=uz_d[k+1];
		temp=ux;
		square=ux*ux+uy*uy+uz*uz;
                Ffeq=1.0/18.0*rho_d[k+1]*(1+3.0*temp+4.5*temp*temp-1.5*square);
		fd[k+pitnum*1]=fd[k+1+pitnum*1]+Bfeq-Ffeq;

		ktemp= zstart*nx*ny + (ystart+1)*nx + (xstart+1);
		Bfeq=1.0/36.0*rho_d[ktemp]*(1+3.0*U0+4.5*U0*U0-1.5*U0*U0);
		ux=ux_d[ktemp];
		uy=uy_d[ktemp];
		uz=uz_d[ktemp];
		temp=ux+uy;
		square=ux*ux+uy*uy+uz*uz;
                Ffeq=1.0/36.0*rho_d[ktemp]*(1+3.0*temp+4.5*temp*temp-1.5*square);
		fd[k+pitnum*7]=fd[ktemp+pitnum*7]+Bfeq-Ffeq;
		
		ktemp= zstart*nx*ny + (ystart-1)*nx + (xstart+1);
		Bfeq=1.0/36.0*rho_d[ktemp]*(1+3.0*U0+4.5*U0*U0-1.5*U0*U0);
		ux=ux_d[ktemp];
		uy=uy_d[ktemp];
		uz=uz_d[ktemp];
		temp=ux-uy;
		square=ux*ux+uy*uy+uz*uz;
                Ffeq=1.0/36.0*rho_d[ktemp]*(1+3.0*temp+4.5*temp*temp-1.5*square);
		fd[k+pitnum*10]=fd[ktemp+pitnum*10]+Bfeq-Ffeq;

		ktemp= (zstart+1)*nx*ny + ystart*nx + (xstart+1);
		Bfeq=1.0/36.0*rho_d[ktemp]*(1+3.0*U0+4.5*U0*U0-1.5*U0*U0);
		ux=ux_d[ktemp];
		uy=uy_d[ktemp];
		uz=uz_d[ktemp];
		temp=ux+uz;
		square=ux*ux+uy*uy+uz*uz;
                Ffeq=1.0/36.0*rho_d[ktemp]*(1+3.0*temp+4.5*temp*temp-1.5*square);
		fd[k+pitnum*15]=fd[ktemp+pitnum*15]+Bfeq-Ffeq;

		ktemp= (zstart-1)*nx*ny + ystart*nx + (xstart+1);
		Bfeq=1.0/36.0*rho_d[ktemp]*(1+3.0*U0+4.5*U0*U0-1.5*U0*U0);
		ux=ux_d[ktemp];
		uy=uy_d[ktemp];
		uz=uz_d[ktemp];
		temp=ux-uz;
		square=ux*ux+uy*uy+uz*uz;
                Ffeq=1.0/36.0*rho_d[ktemp]*(1+3.0*temp+4.5*temp*temp-1.5*square);
		fd[k+pitnum*16]=fd[ktemp+pitnum*16]+Bfeq-Ffeq;
	}
	}

}

__global__ void disfBorders_outlet(int nx,int ny,int nz,float* fd,float* rho_d,float* ux_d,float *uy_d,float *uz_d,int pitch,unsigned int* geoD,int* baselatticenum,int* globallatticenum)
{
	//number of threads in this function
	int num_threads = blockDim.x;
	// Block index y
	int by = blockIdx.x;
	// Block index z
	int bz = blockIdx.y;
	// local thread index
	int tx = threadIdx.x;
	
	int ystart,xstart,zstart;
	int sendstart;
	int k,ktemp;
	int pitnum;
	float temp,square;
	float ux,uy,uz;
	
	float Bfeq,Ffeq;
	pitnum = pitch/sizeof(float);
	
	zstart = bz;
	ystart = by*num_threads + tx;
	xstart = nx-1; //dir=1, xstart= nx-1; dir=0,xstartx=0;
	
	k= zstart*nx*ny + ystart*nx + xstart;
	if(ystart<ny)
	{
        	if(geoD[k]==ANOU)
		{
		ux=ux_d[k-1];
		uy=uy_d[k-1];
		uz=uz_d[k-1];
		temp=-ux;
		square=ux*ux+uy*uy+uz*uz;
                Ffeq=1.0/18.0*rho_d[k-1]*(1+3.0*temp+4.5*temp*temp-1.5*square);
		Bfeq=1.0/18.0*rho0_d*(1+3.0*temp+4.5*temp*temp-1.5*square);
		fd[k+pitnum*2]=fd[k-1+pitnum*2]+Bfeq-Ffeq;

		ktemp= zstart*nx*ny + (ystart+1)*nx + (xstart-1);
		ux=ux_d[ktemp];
		uy=uy_d[ktemp];
		uz=uz_d[ktemp];
		temp=-ux+uy;
		square=ux*ux+uy*uy+uz*uz;
                Ffeq=1.0/36.0*rho_d[ktemp]*(1+3.0*temp+4.5*temp*temp-1.5*square);
		Bfeq=1.0/36.0*rho0_d*(1+3.0*temp+4.5*temp*temp-1.5*square);
		fd[k+pitnum*8]=fd[ktemp+pitnum*8]+Bfeq-Ffeq;
		
		ktemp= zstart*nx*ny + (ystart-1)*nx + (xstart-1);
		ux=ux_d[ktemp];
		uy=uy_d[ktemp];
		uz=uz_d[ktemp];
		temp=-ux-uy;
		square=ux*ux+uy*uy+uz*uz;
                Ffeq=1.0/36.0*rho_d[ktemp]*(1+3.0*temp+4.5*temp*temp-1.5*square);
		Bfeq=1.0/36.0*rho0_d*(1+3.0*temp+4.5*temp*temp-1.5*square);
		fd[k+pitnum*9]=fd[ktemp+pitnum*9]+Bfeq-Ffeq;

		ktemp= (zstart-1)*nx*ny + ystart*nx + (xstart-1);
		ux=ux_d[ktemp];
		uy=uy_d[ktemp];
		uz=uz_d[ktemp];
		temp=-ux-uz;
		square=ux*ux+uy*uy+uz*uz;
                Ffeq=1.0/36.0*rho_d[ktemp]*(1+3.0*temp+4.5*temp*temp-1.5*square);
		Bfeq=1.0/36.0*rho0_d*(1+3.0*temp+4.5*temp*temp-1.5*square);
		fd[k+pitnum*17]=fd[ktemp+pitnum*17]+Bfeq-Ffeq;

		ktemp= (zstart+1)*nx*ny + ystart*nx + (xstart-1);
		ux=ux_d[ktemp];
		uy=uy_d[ktemp];
		uz=uz_d[ktemp];
		temp=-ux+uz;
		square=ux*ux+uy*uy+uz*uz;
                Ffeq=1.0/36.0*rho_d[ktemp]*(1+3.0*temp+4.5*temp*temp-1.5*square);
		Bfeq=1.0/36.0*rho0_d*(1+3.0*temp+4.5*temp*temp-1.5*square);
		fd[k+pitnum*18]=fd[ktemp+pitnum*18]+Bfeq-Ffeq;
		}
	}


}
/***************Transfer boundary lattice to neighbor process**********************/
__global__ void LBBorders_x(int nx, int ny, int nz, float* sendbuf, int dir, float* fd,int pitch)
{//grid1
	//number of threads in this function
	int num_threads = blockDim.x;
	// Block index y
	int by = blockIdx.x;
	// Block index z
	int bz = blockIdx.y;
	// local thread index
	int tx = threadIdx.x;
	
	int ystart,xstart,zstart;
	int sendstart;
	int k;
	int pitnum;
	
	pitnum = pitch/sizeof(float);
	
	zstart = bz;
	ystart = by*num_threads + tx;
	xstart = dir*(nx-1); //dir=1, xstart= nx-1; dir=0,xstartx=0;
	
	k= zstart*nx*ny + ystart*nx + xstart;
	sendstart = zstart* ny + ystart;
	
	if(ystart < ny)//the value of ystart should be less than ny
	{
	 if( dir ==1)
	 {
	  sendbuf[5*sendstart+0] = fd[k + pitnum*1]; //fe
	  sendbuf[5*sendstart+1] = fd[k + pitnum*7]; //fne
	  sendbuf[5*sendstart+2] = fd[k + pitnum*10]; //fse
	  sendbuf[5*sendstart+3] = fd[k + pitnum*15]; //fte
	  sendbuf[5*sendstart+4] = fd[k + pitnum*16]; //fb

	 }else
	 { 
	  sendbuf[5*sendstart+0] = fd[k + pitnum*2]; //fw
	  
	  sendbuf[5*sendstart+1] = fd[k + pitnum*8]; //fnw
	  sendbuf[5*sendstart+2] = fd[k + pitnum*9]; //fsw
	  sendbuf[5*sendstart+3] = fd[k + pitnum*17]; //fbw
	  sendbuf[5*sendstart+4] = fd[k + pitnum*18]; //ftw
	 }
	}	
}

__global__ void LBBorders_y(int nx, int ny, int nz, float* sendbuf, int dir, float* fd, int pitch)
{//gridx	
	//number of threads in this function
	int num_threads = blockDim.x;
	// Block index y
	int bx = blockIdx.x;
	int bz = blockIdx.y;
	// local thread index
	int tx = threadIdx.x;
	
	int ystart, xstart, zstart;
	int sendstart;
	
	int k;
	int pitnum;
	
	pitnum = pitch/sizeof(float);
	
	xstart = bx*num_threads + tx ;
	ystart = dir*(ny-1); //dir=1, ystart= ny-1; dir=0,ystart=0;
	zstart = bz;
	
	k= zstart*nx*ny + ystart*nx + xstart;
	sendstart = zstart*nx + xstart;
	
	if(xstart < nx) // xstart should be less than nx
	{
	 if( dir ==1)
	 {
	  sendbuf[5*sendstart+0] = fd[k + pitnum*3]; //fn
	 
	  sendbuf[5*sendstart+1] = fd[k + pitnum*7]; //fne
	  sendbuf[5*sendstart+2] = fd[k + pitnum*8]; //fnw
	  sendbuf[5*sendstart+3] = fd[k + pitnum*11]; //ftn
	  sendbuf[5*sendstart+4] = fd[k + pitnum*14]; //fbn
	 
	 }else
	 { 
	  sendbuf[5*sendstart+0] = fd[k + pitnum*4]; //fs
	 
	  sendbuf[5*sendstart+1] = fd[k + pitnum*9]; //fsw
	  sendbuf[5*sendstart+2] = fd[k + pitnum*10]; //fse
	  sendbuf[5*sendstart+3] = fd[k + pitnum*12]; //fts
	  sendbuf[5*sendstart+4] = fd[k + pitnum*13]; //fbs
	 }
	} 	
}

__global__ void LBBorders_z(int nx, int ny, int nz, float* sendbuf, int dir, float* fd,int pitch)
{//gridx	
	//number of threads in this function
	int num_threads = blockDim.x;
	// Block index y
	int bx = blockIdx.x;
	
	int by = blockIdx.y;
	// local thread index
	int tx = threadIdx.x;
	
	int ystart,xstart,zstart;
	int sendstart;
	
	int k;
	int pitnum;
	
	pitnum = pitch/sizeof(float);
	
	xstart = bx*num_threads + tx ;
	ystart = by; //dir=1, ystart= ny-1; dir=0,ystart=0;
	zstart = dir*(nz-1);
	
	k= zstart*nx*ny + ystart*nx + xstart;
	sendstart = ystart*nx + xstart;
	
	if(xstart < nx) // xstart should be less than nx
	{
	 if( dir == 1)
	 {
	  sendbuf[5*sendstart+0] = fd[k + pitnum*5]; //ft
	 
	  sendbuf[5*sendstart+1] = fd[k + pitnum*11]; //fnt
	  sendbuf[5*sendstart+2] = fd[k + pitnum*12]; //fst
	  sendbuf[5*sendstart+3] = fd[k + pitnum*15]; //fte
	  sendbuf[5*sendstart+4] = fd[k + pitnum*18]; //ftw
	 }else
	 { 
	  sendbuf[5*sendstart+0] = fd[k + pitnum*6]; //fb
	 
	  sendbuf[5*sendstart+1] = fd[k + pitnum*13]; //fsb
	  sendbuf[5*sendstart+2] = fd[k + pitnum*14]; //fnb
	  sendbuf[5*sendstart+3] = fd[k + pitnum*16]; //fbe
	  sendbuf[5*sendstart+4] = fd[k + pitnum*17]; //fbw

	 }
	} 	
}

__global__ void LBPostborders_x(int nx, int ny, int nz, int xstartoff, int ystartoff, int zstartoff, float* recvbuf, int dir, float* fd, int pitch, int* baselatticenum, int* globallatticenum)
{//grid1//need to be modified when there are more process in x direction
	//number of threads in this function
	int num_threads = blockDim.x;
	// Block index y
	int by = blockIdx.x;
	int bz = blockIdx.y;
	// local thread index
	int tx = threadIdx.x;
	
	int ystart,xstart,zstart;
	int sendstart;
	
	int k;
	int ylatticeId, zlatticeId;
	
	int pitnum;
	pitnum = pitch/sizeof(float);
	
	zstart = bz;
	ystart = by*num_threads + tx;
	
	sendstart = zstart* ny + ystart;
	ylatticeId = ystart + baselatticenum[1];
	zlatticeId = zstart + baselatticenum[2];
	
	if(ystart < ny) // ystart should be less than ny
	{
	 if(dir==1)	//dir=1, xstart= nx-1; dir=-1,xstartx=1;
	 {
	  xstart = 1 ;
	  
	  k= zstart*nx*ny + ystart*nx + xstart;
	  
	  fd[k+ pitnum*1] = recvbuf[5*sendstart + 0]; //fe
	  fd[k+ pitnum*7] = recvbuf[5*sendstart + 1]; //fne
	  fd[k+ pitnum*10] = recvbuf[5*sendstart + 2]; //fse
	  fd[k+ pitnum*15] = recvbuf[5*sendstart + 3]; //fte
	  fd[k+ pitnum*16] = recvbuf[5*sendstart + 4]; //fbe
	 }else
	 { 
	  xstart = nx -2 ;   
	  k= zstart*nx*ny + ystart*nx + xstart;
	  fd[k+ pitnum*2] = recvbuf[5*sendstart + 0]; //fw
	  fd[k+ pitnum*8] = recvbuf[5*sendstart + 1]; //fnw
	  fd[k+ pitnum*9] = recvbuf[5*sendstart + 2]; //fsw
	  fd[k+ pitnum*17] = recvbuf[5*sendstart + 3]; //ftw
	  fd[k+ pitnum*18] = recvbuf[5*sendstart + 4]; //fbw	  
	 }
	}  
}

__global__ void LBPostborders_y(int nx, int ny, int nz, int xstartoff, int ystartoff, int zstartoff, float* recvbuf, int dir, float* fd,int pitch, int* baselatticenum, int* globallatticenum)
{//gridx
	//number of threads in this function
	int num_threads = blockDim.x;
	// Block index y
	int bx = blockIdx.x;	
	int bz = blockIdx.y;
	// local thread index
	int tx = threadIdx.x;
	
	int ystart, xstart, zstart;
	int sendstart;
	
	int k;
	
	int pitnum;
	int xlatticeId, zlatticeId;
	
	pitnum = pitch/sizeof(float);
	
	xstart = bx*num_threads + tx ;
	zstart = bz;	
	
	sendstart = zstart*nx + xstart;
	
	xlatticeId = xstart + baselatticenum[0];
	zlatticeId = zstart + baselatticenum[2];
	
	if(xstart < nx) // xstart should be less than nx 
	{
	 if(dir==1)	//dir=1, ystart= 0; dir=0, ystartx= nx -1;
	 {// for the east and west distribute function has been changed before send out in the original process 
	  ystart = 1;
	  k= zstart*nx*ny + ystart*nx + xstart;
	  fd[k+ pitnum*3] = recvbuf[5*sendstart + 0]; //fn
	   fd[k+ pitnum*7] = recvbuf[5*sendstart + 1]; //fne
	   fd[k+ pitnum*8] = recvbuf[5*sendstart + 2]; //fnw
	   fd[k+ pitnum*11] = recvbuf[5*sendstart + 3]; //ftn
	   fd[k+ pitnum*14] = recvbuf[5*sendstart + 4]; //fbn  
	 }else
	 { 
	  ystart = ny -2 ; 
	  k= zstart*nx*ny + ystart*nx + xstart;
	  fd[k+ pitnum*4] = recvbuf[5*sendstart + 0]; //fs
	   fd[k+ pitnum*9] = recvbuf[5*sendstart + 1]; //fse
	   fd[k+ pitnum*10] = recvbuf[5*sendstart + 2]; //fsw
	   fd[k+ pitnum*12] = recvbuf[5*sendstart + 3]; //fts
	   fd[k+ pitnum*13] = recvbuf[5*sendstart + 4]; //fbs
	 }
	} 
}

__global__ void LBPostborders_z(int nx, int ny, int nz, int xstartoff, int ystartoff, int zstartoff, float* recvbuf, int dir, float* fd, int pitch, int* baselatticenum, int* globallatticenum)
{//gridx
	//number of threads in this function
	int num_threads = blockDim.x;
	// Block index y
	int bx = blockIdx.x;	
	int by = blockIdx.y;
	// local thread index
	int tx = threadIdx.x;
	
	int ystart, xstart, zstart;
	int sendstart;
	
	int k;
	
	int pitnum;
	pitnum = pitch/sizeof(float);
	int xlatticeId, ylatticeId;
	
	xstart = bx*num_threads + tx ;
	ystart = by;	
	sendstart = ystart*nx + xstart;
	
	xlatticeId = xstart + baselatticenum[0];
	ylatticeId = ystart + baselatticenum[1];
	
	if( xstart < nx ) // xstart should be less than nx
	{
	 if(dir==1)	//dir=1, ystart= 0; dir=0, ystartx= nx -1;
	 {// for the east and west distribute function has been changed before send out in the original process 
	  zstart = 1;
	  
	  k= zstart*nx*ny + ystart*nx + xstart;
	  
	  fd[k+ pitnum*5] = recvbuf[5*sendstart + 0]; //ft
	   fd[k+ pitnum*11] = recvbuf[5*sendstart + 1]; //fte
	   fd[k+ pitnum*12] = recvbuf[5*sendstart + 2]; //ftw
	   fd[k+ pitnum*15] = recvbuf[5*sendstart + 3]; //fnt
	   fd[k+ pitnum*18] = recvbuf[5*sendstart + 4]; //fst	 	 
	 }else
	 { 
	  zstart = nz -2 ; 
	  
	  k= zstart*nx*ny + ystart*nx + xstart;
	  
	  fd[k+ pitnum*6] = recvbuf[5*sendstart + 0]; //fb
	   fd[k+ pitnum*13] = recvbuf[5*sendstart + 1]; //fbe
	   fd[k+ pitnum*14] = recvbuf[5*sendstart + 2]; //fbw
	   fd[k+ pitnum*16] = recvbuf[5*sendstart + 3]; //fnb
	   fd[k+ pitnum*17] = recvbuf[5*sendstart + 4]; //fsb
	 }
	} 
}
__global__ void TransBorders_x(int nx, int ny, int nz, float* sendbuf, int dir, float* con,float* ux,float* uy,float* uz, int pitch)
{//grid1
	//number of threads in this function
	int num_threads = blockDim.x;
	// Block index y
	int by = blockIdx.x;
	// Block index z
	int bz = blockIdx.y;
	// local thread index
	int tx = threadIdx.x;
	
	int ystart,xstart,zstart;
	int sendstart;
	int k,k1,k2;

	zstart = bz;
	ystart = by*num_threads + tx;
	xstart = dir*(nx-1); //dir=1, xstart= nx-1; dir=0,xstartx=0;
	
	k= zstart*nx*ny + ystart*nx + xstart;
	k2= (zstart+1)*(nx+2)*(ny+2) + (ystart+1)*(nx+2) + (xstart+1);
	sendstart = zstart* ny + ystart;
	
	if(ystart < ny)//the value of ystart should be less than ny
	{
	 if( dir ==1)
	 {
	  k1=k-1;
	  k2=k2-2;
	  sendbuf[5*sendstart+0] = con[k2];
	  k2=k2+1;
	  sendbuf[5*sendstart+1] = con[k2];
	  sendbuf[5*sendstart+2] = ux[k1]; 
	  sendbuf[5*sendstart+3] = uy[k1]; 
	  sendbuf[5*sendstart+4] = uz[k1];  
	 }else
	 { 
	  k1=k+1;
	  k2=k2+1;
	  sendbuf[5*sendstart+0] = con[k2];
	  k2=k2+1;
	  sendbuf[5*sendstart+1] = con[k2];
	  sendbuf[5*sendstart+2] = ux[k1]; 
	  sendbuf[5*sendstart+3] = uy[k1]; 
	  sendbuf[5*sendstart+4] = uz[k1]; 
	 }
	}	
}

__global__ void TransBorders_y(int nx, int ny, int nz, float* sendbuf, int dir, float* con,float* ux,float* uy,float* uz,int pitch)
{//gridx	
	//number of threads in this function
	int num_threads = blockDim.x;
	// Block index y
	int bx = blockIdx.x;
	int bz = blockIdx.y;
	// local thread index
	int tx = threadIdx.x;
	int ystart, xstart, zstart;
	int sendstart;
	int k,k1,k2;
	
	xstart = bx*num_threads + tx ;
	ystart = dir*(ny-1); //dir=1, ystart= ny-1; dir=0,ystart=0;
	zstart = bz;
	
	k= zstart*nx*ny + ystart*nx + xstart;
	k2= (zstart+1)*(nx+2)*(ny+2) + (ystart+1)*(nx+2) + (xstart+1);
	sendstart = zstart*nx + xstart;
	
	if(xstart < nx) // xstart should be less than nx
	{
	 if( dir ==1)
	 {
	  k1=k-nx;
	  k2=k2-2*(nx+2);
	  sendbuf[5*sendstart+0] = con[k2];
	  k2=k2+(nx+2);
	  sendbuf[5*sendstart+1] = con[k2];
	  sendbuf[5*sendstart+2] = ux[k1]; 
	  sendbuf[5*sendstart+3] = uy[k1]; 
	  sendbuf[5*sendstart+4] = uz[k1]; 
	 }else
	 { 
	  k1=k+nx;
	  k2=k2+(nx+2);
	  sendbuf[5*sendstart+0] = con[k2];
	  k2=k2+(nx+2);
	  sendbuf[5*sendstart+1] = con[k2];
	  sendbuf[5*sendstart+2] = ux[k1]; 
	  sendbuf[5*sendstart+3] = uy[k1]; 
	  sendbuf[5*sendstart+4] = uz[k1]; 
	 }
	} 	
}

__global__ void TransBorders_z(int nx, int ny, int nz, float* sendbuf, int dir, float* con,float* ux,float* uy,float *uz, int pitch)
{//gridx	
	//number of threads in this function
	int num_threads = blockDim.x;
	// Block index y
	int bx = blockIdx.x;
	
	int by = blockIdx.y;
	// local thread index
	int tx = threadIdx.x;
	
	int ystart,xstart,zstart;
	int sendstart;
	
	int k,k1,k2;
	xstart = bx*num_threads + tx ;
	ystart = by; //dir=1, ystart= ny-1; dir=0,ystart=0;
	zstart = dir*(nz-1);
	
	k= zstart*nx*ny + ystart*nx + xstart;
	k2= (zstart+1)*(nx+2)*(ny+2) + (ystart+1)*(nx+2) + (xstart+1);
	
	sendstart = ystart*nx + xstart;
	
	if(xstart < nx) // xstart should be less than nx
	{
	 if( dir == 1)
	 {
	  k1=k-nx*ny;
	  k2=k2-2*(nx+2)*(ny+2);
	  sendbuf[5*sendstart+0] = con[k2];
	  k2=k2+(nx+2)*(ny+2);
	  sendbuf[5*sendstart+1] = con[k2];
	  sendbuf[5*sendstart+2] = ux[k1]; 
	  sendbuf[5*sendstart+3] = uy[k1]; 
	  sendbuf[5*sendstart+4] = uz[k1]; 

	 }else
	 { 
	  k1=k+nx*ny;
	  k2=k2+(nx+2)*(ny+2);
	  sendbuf[5*sendstart+0] = con[k2];
	  k2=k2+(nx+2)*(ny+2);
	  sendbuf[5*sendstart+1] = con[k2];
	  sendbuf[5*sendstart+2] = ux[k1]; 
	  sendbuf[5*sendstart+3] = uy[k1]; 
	  sendbuf[5*sendstart+4] = uz[k1];
	 }
	} 	
}

__global__ void TransPostborders_x(int nx, int ny, int nz, int xstartoff, int ystartoff, int zstartoff, float* recvbuf, int dir, float* con,float* ux,float* uy,float* uz, int pitch, int* baselatticenum, int* globallatticenum)
{//grid1//need to be modified when there are more process in x direction
	//number of threads in this function
	int num_threads = blockDim.x;
	// Block index y
	int by = blockIdx.x;
	int bz = blockIdx.y;
	// local thread index
	int tx = threadIdx.x;
	
	int ystart,xstart,zstart;
	int sendstart;
	
	int k1,k2;
	
	zstart = bz;
	ystart = by*num_threads + tx;
	
	sendstart = zstart* ny + ystart;
	
	if(ystart < ny) // ystart should be less than ny
	{
	 if(dir==1)	//dir=1, xstart= nx-1; dir=-1,xstartx=1;
	 {
	  xstart = 1 ;
	  k1= zstart*nx*ny + ystart*nx + xstart-1;
	  k2=(zstart+1)*(nx+2)*(ny+2) + (ystart+1)*(nx+2) + (xstart+1)-2;
	  con[k2] = recvbuf[5*sendstart + 0];
	  k2=(zstart+1)*(nx+2)*(ny+2) + (ystart+1)*(nx+2) + (xstart+1)-1;
	  con[k2] = recvbuf[5*sendstart + 1];
	  ux[k1] = recvbuf[5*sendstart + 2]; 	 	 
	  uy[k1] = recvbuf[5*sendstart + 3]; 	 	 
	  uz[k1] = recvbuf[5*sendstart + 4]; 
	 }else
	 { 
	  xstart = nx -2 ;   
	  k1= zstart*nx*ny + ystart*nx + xstart+1;
	  k2=(zstart+1)*(nx+2)*(ny+2) + (ystart+1)*(nx+2) + (xstart+1)+1;
	  con[k2] = recvbuf[5*sendstart + 0];
	  k2=(zstart+1)*(nx+2)*(ny+2) + (ystart+1)*(nx+2) + (xstart+1)+2;
	  con[k2] = recvbuf[5*sendstart + 1];
	  ux[k1] = recvbuf[5*sendstart + 2]; 	 	 
	  uy[k1] = recvbuf[5*sendstart + 3]; 	 	 
	  uz[k1] = recvbuf[5*sendstart + 4]; 
	 }
	}  	
	
}

__global__ void TransPostborders_y(int nx, int ny, int nz, int xstartoff, int ystartoff, int zstartoff, float* recvbuf, int dir,float* con,float* ux,float* uy,float* uz, int pitch, int* baselatticenum, int* globallatticenum)
{//gridx
	//number of threads in this function
	int num_threads = blockDim.x;
	// Block index y
	int bx = blockIdx.x;	
	int bz = blockIdx.y;
	// local thread index
	int tx = threadIdx.x;
	
	int ystart, xstart, zstart;
	int sendstart;
	
	int k1,k2;
	
	xstart = bx*num_threads + tx ;
	zstart = bz;	
	
	sendstart = zstart*nx + xstart;
	
	if(xstart < nx) // xstart should be less than nx 
	{
	 if(dir==1)	//dir=1, ystart= 0; dir=0, ystartx= nx -1;
	 {// for the east and west distribute function has been changed before send out in the original process 
	  ystart = 1;
	  k1= zstart*nx*ny + (ystart-1)*nx + xstart;
	  k2=(zstart+1)*(nx+2)*(ny+2) + (ystart-1)*(nx+2) + (xstart+1);
	  con[k2] = recvbuf[5*sendstart + 0];
	  k2=(zstart+1)*(nx+2)*(ny+2) + (ystart)*(nx+2) + (xstart+1);
	  con[k2] = recvbuf[5*sendstart + 1];
	  ux[k1] = recvbuf[5*sendstart + 2]; 	 	 
	  uy[k1] = recvbuf[5*sendstart + 3]; 	 	 
	  uz[k1] = recvbuf[5*sendstart + 4]; 	  
	 }else
	 { 
	  ystart = ny -2 ; 
	  k1= zstart*nx*ny + (ystart+1)*nx + xstart;
	  k2=(zstart+1)*(nx+2)*(ny+2) + (ystart+2)*(nx+2) + (xstart+1);
	  con[k2] = recvbuf[5*sendstart + 0];
	  k2=(zstart+1)*(nx+2)*(ny+2) + (ystart+3)*(nx+2) + (xstart+1);
	  con[k2] = recvbuf[5*sendstart + 1];
	  ux[k1] = recvbuf[5*sendstart + 2]; 	 	 
	  uy[k1] = recvbuf[5*sendstart + 3]; 	 	 
	  uz[k1] = recvbuf[5*sendstart + 4]; 
	 }
	} 
	
}

__global__ void TransPostborders_z(int nx, int ny, int nz, int xstartoff, int ystartoff, int zstartoff, float* recvbuf, int dir,float* con,float* ux,float* uy,float* uz, int pitch, int* baselatticenum, int* globallatticenum)
{//gridx
	//number of threads in this function
	int num_threads = blockDim.x;
	// Block index y
	int bx = blockIdx.x;	
	int by = blockIdx.y;
	// local thread index
	int tx = threadIdx.x;
	int ystart, xstart, zstart;
	int sendstart;
	
	int k1,k2;
	
	xstart = bx*num_threads + tx ;
	ystart = by;	
	sendstart = ystart*nx + xstart;
	
	if( xstart < nx ) // xstart should be less than nx
	{
	 if(dir==1)	//dir=1, ystart= 0; dir=0, ystartx= nx -1;
	 {// for the east and west distribute function has been changed before send out in the original process 
	  zstart = 1;
	  k1= (zstart-1)*nx*ny + ystart*nx + xstart;
	  k2= (zstart+1-2)*(nx+2)*(ny+2) + (ystart+1)*(nx+2) + (xstart+1);
	  con[k2] = recvbuf[5*sendstart + 0]; 
	  k2= (zstart+1-1)*(nx+2)*(ny+2) + (ystart+1)*(nx+2) + (xstart+1);
	  con[k2] = recvbuf[5*sendstart + 1]; 	 	 
	  ux[k1] = recvbuf[5*sendstart + 2]; 	 	 
	  uy[k1] = recvbuf[5*sendstart + 3]; 	 	 
	  uz[k1] = recvbuf[5*sendstart + 4]; 	 	 
	 }else
	 { 
	  zstart = nz -2 ; 
	  k1= (zstart+1)*nx*ny + ystart*nx + xstart;
	  k2= (zstart+1+1)*(nx+2)*(ny+2) + (ystart+1)*(nx+2) + (xstart+1);
	  con[k2] = recvbuf[5*sendstart + 0]; 
	  k2= (zstart+1+2)*(nx+2)*(ny+2) + (ystart+1)*(nx+2) + (xstart+1);
	  con[k2] = recvbuf[5*sendstart + 1]; 	 	 
	  ux[k1] = recvbuf[5*sendstart + 2]; 	 	 
	  uy[k1] = recvbuf[5*sendstart + 3]; 	 	 
	  uz[k1] = recvbuf[5*sendstart + 4]; 
	 }
	} 
}


// write the calculated lattice data to lattice can be read by TECPlot
__host__ void write_TEC1_file(int t1, int postinterval)
{
	float *rho,*ux,*uy,*uz,*con,*geo;
	float *rho1,*ux1,*uy1,*uz1,*con1,*geo1;
	float *rhoo,*uxo,*uyo,*uzo,*cono,*geoo;
	int NX,NY,NZ;
	int i,j,k,procsid;
	int il,jl,kl,t,tl,tc,tg;
	int procp[3];
	
	procsid=h_procid;

	NX=h_GlobalLatticeNum[0];
	NY=h_GlobalLatticeNum[1];
	NZ=h_GlobalLatticeNum[2];
	
	con=(float *)malloc(sizeof(float)*nx*ny*nz);
	geo=(float *)malloc(sizeof(float)*nx*ny*nz);
	rho=(float *)malloc(sizeof(float)*nx*ny*nz);
	ux=(float *)malloc(sizeof(float)*nx*ny*nz);
	uy=(float *)malloc(sizeof(float)*nx*ny*nz);
	uz=(float *)malloc(sizeof(float)*nx*ny*nz);
	for(z=0;z<nz;z++)
	{
	  for(y=0;y<ny;y++)
	  {
	    for(x=0;x<nx;x++)
	    {
		tl=x+nx*(y+ny*z);
		t=(x+1)+(nx+2)*(y+1+(ny+2)*(z+1));
		tc=(x+2)+(nx+4)*(y+2+(ny+4)*(z+2));
				
		con[tl]=con_h[tc];
		geo[tl]=h_geoD[t];
		rho[tl]=rho_h[t];
		ux[tl]=ux_h[t];
		uy[tl]=uy_h[t];
		uz[tl]=uz_h[t];
	    }
	  }
	}
	
	con1=(float *)malloc(sizeof(float)*NX*NY*NZ);
	geo1=(float *)malloc(sizeof(float)*NX*NY*NZ);
	rho1=(float *)malloc(sizeof(float)*NX*NY*NZ);
	ux1=(float *)malloc(sizeof(float)*NX*NY*NZ);
	uy1=(float *)malloc(sizeof(float)*NX*NY*NZ);
	uz1=(float *)malloc(sizeof(float)*NX*NY*NZ);

	cono=(float *)malloc(sizeof(float)*NX*NY*NZ);
	geoo=(float *)malloc(sizeof(float)*NX*NY*NZ);
	rhoo=(float *)malloc(sizeof(float)*NX*NY*NZ);
	uxo=(float *)malloc(sizeof(float)*NX*NY*NZ);
	uyo=(float *)malloc(sizeof(float)*NX*NY*NZ);
	uzo=(float *)malloc(sizeof(float)*NX*NY*NZ);

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Gather(con,nx*ny*nz,MPI_FLOAT,con1,nx*ny*nz,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Gather(geo,nx*ny*nz,MPI_FLOAT,geo1,nx*ny*nz,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Gather(rho,nx*ny*nz,MPI_FLOAT,rho1,nx*ny*nz,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Gather(ux,nx*ny*nz,MPI_FLOAT,ux1,nx*ny*nz,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Gather(uy,nx*ny*nz,MPI_FLOAT,uy1,nx*ny*nz,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Gather(uz,nx*ny*nz,MPI_FLOAT,uz1,nx*ny*nz,MPI_FLOAT,0,MPI_COMM_WORLD);

	if(procsid==0)
	{
	   for(z=0;z<NZ;z++)
	   {
	      for(y=0;y<NY;y++)
	      {
		  for(x=0;x<NX;x++)
		  {
			t=x+NX*(y+NY*z);//position in gathered memory
			procsid=t/(nx*ny*nz);
			procp[0]=procsid%h_procn[0];
			procsid=procsid/h_procn[0];
			procp[1]=procsid%h_procn[1];
			procsid=procsid/h_procn[1];
			procp[2]=procsid%h_procn[2];

			tl=t-procsid*nx*ny*nz;//postion in task memory

			il=tl%nx;
			tl=tl/nx;
			jl=tl%ny;
			tl=tl/ny;
			kl=tl%nz;

			i=il+procp[0]*nx;
			j=jl+procp[1]*ny;
			k=kl+procp[2]*nz;

			tg=i+NX*(j+NY*k);
			
			cono[tg]=con1[t];
			geoo[tg]=geo1[t];
			rhoo[tg]=rho1[t];
			uxo[tg]=ux1[t];
	  		uyo[tg]=uy1[t];
			uzo[tg]=uz1[t];
		  }
	      }
	    }
	    
	    //writing data for tecplot reading
	    ostringstream name;
	    name<<"BP_Mixing_Tank"<<t1<<".plt";
	    ofstream out(name.str().c_str());
	    out<<"Title=\"BP_Mixing_Tank\"\n"<<"Variables =\"X\",\"Y\",\"Z\",\"geo\",\"rho\",\"con\",\"U\",\"V\",\"W\"\n"<<"Zone T=\"BOX\",I="
	       <<" "<<NX<<",J="<<" "<<NY<<",K="<<" "<<NZ<<" "<<",F= POINT"<<endl;
	
	   for(k=0;k<NZ;k++)
	   {
		for(j=0;j<NY;j++)
		{
		    for(i=0;i<NX;i++)
		    {
			t=i+NX*(j+NY*k);
			out<<double(i)/NX<<" "<<double(j)/NX<<" "<<double(k)/NX<<" "<<geoo[t]<<" "<<rhoo[t]<<" "<<cono[t]<<" "<<uxo[t]<<" "<<uyo[t]<<" "<<uzo[t]<<'\n';
		    }
		}
	    }
	}
	
	free(con);
	free(geo);
	free(rho);
	free(ux);
	free(uy);
	free(uz);

	free(con1);
	free(geo1);
	free(rho1);
	free(ux1);
	free(uy1);
	free(uz1);
	
	free(cono);
	free(geoo);
	free(rhoo);
	free(uxo);
	free(uyo);
	free(uzo);

	MPI_Barrier(MPI_COMM_WORLD);
	
}// end of write_TEC1_file
__global__ void MacroCal(int nx, int ny, int nz, int xoff, int yoff, int zoff, unsigned int* geoD,
int pitch, float* f0_c1,float* rho_d,float* ux_d,float* uy_d,float* uz_d)
{
	int num_threads = blockDim.x;
	// local thread index
	int tx = threadIdx.x;
	// Block index in x
	int bx = blockIdx.x;
	// Block index in y
	int by = blockIdx.y;
	// Global x-Index
	int xStart = tx+ xoff;
	// Global y-Index
	int yStart = bx+ yoff;
	int zStart = by + zoff;
	int pitnum;
	pitnum = pitch/sizeof(float);
	
	//add this for thread id great than lattice number
	if(tx >= nx-2*xoff ) return;
	// Index k in 1D-arrays
	int k = nx*(ny*zStart+ yStart)+ xStart;
	float mybodyf;
	float rho,rho1, vx1, vy1, vz1;
	float F_IN_R1, F_IN_E1, F_IN_N1, F_IN_W1, F_IN_S1, F_IN_T1, F_IN_B1, F_IN_NE1, F_IN_NW1, F_IN_SW1, F_IN_SE1,
	   	F_IN_NT1, F_IN_ST1, F_IN_SB1,F_IN_NB1, F_IN_TE1, F_IN_BE1, F_IN_BW1, F_IN_TW1;
	mybodyf=bodyf;
	F_IN_R1 = f0_c1[k];
	
	F_IN_E1 = f0_c1[k+1*pitnum];	
	F_IN_W1 = f0_c1[k+2*pitnum];
	F_IN_N1 = f0_c1[k+3*pitnum];
	F_IN_S1 = f0_c1[k+4*pitnum];
	F_IN_T1 = f0_c1[k+5*pitnum];
	F_IN_B1 = f0_c1[k+6*pitnum];
	
	F_IN_NE1 = f0_c1[k+7*pitnum];
	F_IN_NW1 = f0_c1[k+8*pitnum];
	F_IN_SW1 = f0_c1[k+9*pitnum];
	F_IN_SE1 = f0_c1[k+10*pitnum];
	
	F_IN_NT1 = f0_c1[k+11*pitnum];
	F_IN_ST1 = f0_c1[k+12*pitnum];
	F_IN_SB1 = f0_c1[k+13*pitnum];
	F_IN_NB1 = f0_c1[k+14*pitnum];
	
	F_IN_TE1 = f0_c1[k+15*pitnum];
	F_IN_BE1 = f0_c1[k+16*pitnum];
	F_IN_BW1 = f0_c1[k+17*pitnum];
	F_IN_TW1 = f0_c1[k+18*pitnum];
	
	rho1 = F_IN_R1 + F_IN_E1 + F_IN_N1 + F_IN_W1 + F_IN_S1 +F_IN_T1 + F_IN_B1+ F_IN_NE1+ F_IN_NW1+ F_IN_SW1+ F_IN_SE1+
	   	F_IN_NT1+ F_IN_ST1 + F_IN_SB1 + F_IN_NB1+ F_IN_TE1+ F_IN_BE1+ F_IN_BW1+ F_IN_TW1;
	rho=1.0/rho1;
	vx1 = (F_IN_E1 - F_IN_W1 + F_IN_NE1 - F_IN_NW1 + F_IN_SE1 - F_IN_SW1 + F_IN_TE1 + F_IN_BE1 - F_IN_TW1 - F_IN_BW1)*rho+0.5*mybodyf*rho;
	vy1 = (F_IN_N1 - F_IN_S1 + F_IN_NE1 + F_IN_NW1 - F_IN_SE1 - F_IN_SW1 + F_IN_NT1 - F_IN_ST1 + F_IN_NB1 - F_IN_SB1)*rho+0.5*mybodyf*rho;
	vz1 = (F_IN_T1 - F_IN_B1 + F_IN_NT1 + F_IN_ST1 - F_IN_SB1 - F_IN_NB1 + F_IN_TE1 - F_IN_BE1 + F_IN_TW1 - F_IN_BW1)*rho+0.5*mybodyf*rho;
	
	 rho_d[k] = rho1;
	 ux_d[k] = vx1;
	 uy_d[k] = vy1;
	 uz_d[k] = vz1;
}

//
__device__ float D(float a,int b)
{
	double temp=0.0;
	temp=abs(a-b);
	if(temp<=1)
		temp=1-temp;
	else
		temp=0;
	return temp;
}
//interpolation fluid velocity to boundary nodes and calculate force exerted by nodes;
__global__ void	 RotorForce (int timestep,int N,float* node,float* F,float* ux,float* uy,float* uz,int nx,int ny,int nz,int* d_baseLatticeNum,int* d_GlobalLatticeNum)
{
	int num_threads = blockDim.x;
	// local thread index
	int tx = threadIdx.x;
	// Block index in x
	int bx = blockIdx.x;

	int xStart = tx;
	
	int yStart = bx;

	int k = yStart*num_threads+ xStart;

	if(k>=N)
	return;
	
	float Pi=3.1415926;
	
	float uxnoF,uynoF,uznoF;
	float uxd,uyd,uzd;
	float nodex_update,nodey_update,nodez_update;
	int i_m,i_p,j_m,j_p,k_m,k_p;
	int DIM=3;

	nodex_update=node[k*DIM+0]*cos(U0*timestep+node[k*DIM+1])+0.5*d_GlobalLatticeNum[0];
	nodey_update=node[k*DIM+0]*sin(U0*timestep+node[k*DIM+1])+0.5*d_GlobalLatticeNum[1];
	nodez_update=node[k*DIM+2];
	
	i_m=int(nodex_update);
	i_p=i_m+1;
	j_m=int(nodey_update);
	j_p=j_m+1;
	k_m=int(nodez_update);
	k_p=k_m+1;
	
	if(i_m>=d_baseLatticeNum[0]-1&&j_m>=d_baseLatticeNum[1]-1&&k_m>=d_baseLatticeNum[2]-1
	&&i_p<=d_baseLatticeNum[0]+nx-1&&j_p<=d_baseLatticeNum[1]+ny-1&&k_p<=d_baseLatticeNum[2]+nz-1)
	{
		int I_m,J_m,K_m,I_p,J_p,K_p;
		I_m=i_m-d_baseLatticeNum[0]+1;
		J_m=j_m-d_baseLatticeNum[1]+1;
		K_m=k_m-d_baseLatticeNum[2]+1;

		I_p=i_p-d_baseLatticeNum[0]+1;
		J_p=j_p-d_baseLatticeNum[1]+1;
		K_p=k_p-d_baseLatticeNum[2]+1;

	uxnoF=ux[nx*(K_m*ny+J_m)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_m)
	     +ux[nx*(K_m*ny+J_m)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_m)
	     +ux[nx*(K_m*ny+J_p)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_m)
	     +ux[nx*(K_m*ny+J_p)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_m)
	     +ux[nx*(K_p*ny+J_m)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_p)
	     +ux[nx*(K_p*ny+J_m)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_p)
	     +ux[nx*(K_p*ny+J_p)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_p)
	     +ux[nx*(K_p*ny+J_p)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_p);

	uynoF=uy[nx*(K_m*ny+J_m)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_m)
	     +uy[nx*(K_m*ny+J_m)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_m)
	     +uy[nx*(K_m*ny+J_p)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_m)
	     +uy[nx*(K_m*ny+J_p)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_m)
	     +uy[nx*(K_p*ny+J_m)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_p)
	     +uy[nx*(K_p*ny+J_m)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_p)
	     +uy[nx*(K_p*ny+J_p)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_p)
	     +uy[nx*(K_p*ny+J_p)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_p);

	uznoF=uz[nx*(K_m*ny+J_m)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_m)
	     +uz[nx*(K_m*ny+J_m)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_m)
	     +uz[nx*(K_m*ny+J_p)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_m)
	     +uz[nx*(K_m*ny+J_p)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_m)
	     +uz[nx*(K_p*ny+J_m)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_p)
	     +uz[nx*(K_p*ny+J_m)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_p)
	     +uz[nx*(K_p*ny+J_p)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_p)
	     +uz[nx*(K_p*ny+J_p)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_p);

	uxd=U0*node[k*DIM+0]*cos(U0*timestep+node[k*DIM+1]+0.5*Pi);
	uyd=U0*node[k*DIM+0]*sin(U0*timestep+node[k*DIM+1]+0.5*Pi);
	uzd=0.0;
	
	F[0+k*DIM]=uxd-uxnoF;
	F[1+k*DIM]=uyd-uynoF;
	F[2+k*DIM]=uzd-uznoF;
	}
	else
	{
		F[0+k*DIM]=0;
		F[1+k*DIM]=0;
		F[2+k*DIM]=0;
	}
	
							
}
__global__ void	 WallForce (int N,float* node,float* F,float* ux,float* uy,float* uz,int nx,int ny,int nz,int* d_baseLatticeNum,int* d_GlobalLatticeNum)
{
	int num_threads = blockDim.x;
	// local thread index
	int tx = threadIdx.x;
	// Block index in x
	int bx = blockIdx.x;

	int xStart = tx;
	
	int yStart = bx;

	int xlatticeId, ylatticeId, zlatticeId;
	int k = yStart*num_threads+ xStart;
	if(k>=N)
	return;
	
	float Pi=3.1415926;
	
	float uxnoF,uynoF,uznoF;
	float uxd,uyd,uzd;
	float nodex_update,nodey_update,nodez_update;
	int i_m,i_p,j_m,j_p,k_m,k_p;
	int DIM=3;
	nodex_update=node[k*DIM+0];
	nodey_update=node[k*DIM+1];
	nodez_update=node[k*DIM+2];
	
	i_m=int(nodex_update);
	i_p=i_m+1;
	j_m=int(nodey_update);
	j_p=j_m+1;
	k_m=int(nodez_update);
	k_p=k_m+1;
	
	if(i_m>=d_baseLatticeNum[0]-1&&j_m>=d_baseLatticeNum[1]-1&&k_m>=d_baseLatticeNum[2]-1
	&&i_p<=d_baseLatticeNum[0]+nx-1&&j_p<=d_baseLatticeNum[1]+ny-1&&k_p<=d_baseLatticeNum[2]+nz-1)
	{
		int I_m,J_m,K_m,I_p,J_p,K_p;
		I_m=i_m-d_baseLatticeNum[0]+1;
		J_m=j_m-d_baseLatticeNum[1]+1;
		K_m=k_m-d_baseLatticeNum[2]+1;

		I_p=i_p-d_baseLatticeNum[0]+1;
		J_p=j_p-d_baseLatticeNum[1]+1;
		K_p=k_p-d_baseLatticeNum[2]+1;

	uxnoF=ux[nx*(K_m*ny+J_m)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_m)
	     +ux[nx*(K_m*ny+J_m)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_m)
	     +ux[nx*(K_m*ny+J_p)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_m)
	     +ux[nx*(K_m*ny+J_p)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_m)
	     +ux[nx*(K_p*ny+J_m)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_p)
	     +ux[nx*(K_p*ny+J_m)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_p)
	     +ux[nx*(K_p*ny+J_p)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_p)
	     +ux[nx*(K_p*ny+J_p)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_p);

	uynoF=uy[nx*(K_m*ny+J_m)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_m)
	     +uy[nx*(K_m*ny+J_m)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_m)
	     +uy[nx*(K_m*ny+J_p)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_m)
	     +uy[nx*(K_m*ny+J_p)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_m)
	     +uy[nx*(K_p*ny+J_m)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_p)
	     +uy[nx*(K_p*ny+J_m)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_p)
	     +uy[nx*(K_p*ny+J_p)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_p)
	     +uy[nx*(K_p*ny+J_p)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_p);

	uznoF=uz[nx*(K_m*ny+J_m)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_m)
	     +uz[nx*(K_m*ny+J_m)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_m)
	     +uz[nx*(K_m*ny+J_p)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_m)
	     +uz[nx*(K_m*ny+J_p)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_m)
	     +uz[nx*(K_p*ny+J_m)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_p)
	     +uz[nx*(K_p*ny+J_m)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_p)
	     +uz[nx*(K_p*ny+J_p)+I_p]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_p)
	     +uz[nx*(K_p*ny+J_p)+I_m]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_p);

	uxd=0.0;
	uyd=0.0;
	uzd=0.0;
	
	F[0+k*DIM]=uxd-uxnoF;
	F[1+k*DIM]=uyd-uynoF;
	F[2+k*DIM]=uzd-uznoF;
	}
	else
	{
		F[0+k*DIM]=0;
		F[1+k*DIM]=0;
		F[2+k*DIM]=0;
	}
	
		

}
	 //caculate force exerted on the fluids
	 //step 1 initilize Force exerted by fluids
__global__ void	 Forceinitial (int nx,int ny,int nz,float*Fsf)
{
	int num_threads = blockDim.x;
	// local thread index
	int tx = threadIdx.x;
	// Block index in x
	int bx = blockIdx.x;
	// Block index in y
	int by = blockIdx.y;
	int xoff=1,yoff=1,zoff=1;
	// Global x-Index
	int xStart = tx+ xoff;
	// Global y-Index
	int yStart = bx+ yoff;
	int zStart = by + zoff;
	int xlatticeId, ylatticeId, zlatticeId;
	int N=nx*ny*nz;
	
	//add this for thread id great than lattice number
	if(tx >= nx-2*xoff ) return;
		
	// Index k in 1D-arrays
	int k = nx*(ny*zStart+ yStart)+ xStart;
	Fsf[k]=0.0;
	Fsf[k+N]=0.0;
	Fsf[k+2*N]=0.0;
}
	 //step 2 caculate Forceexerted by fluids
__global__ void	 RotorForceonFluid (int timestep,int NR,int nx,int ny,int nz,float* node,float* F,int* d_baseLatticeNum,int* d_GlobalLatticeNum,float* Fsf)
{
	int num_threads = blockDim.x;
	// local thread index
	int tx = threadIdx.x;
	// Block index in x
	int bx = blockIdx.x;

	int xStart = tx;
	
	int yStart = bx;

	int k = yStart*num_threads+ xStart;

	int N=nx*ny*nz;
	if(k>=NR)
	return;
	
	float Pi=3.1415926;
	
	float nodex_update,nodey_update,nodez_update;
	int i_m,i_p,j_m,j_p,k_m,k_p;
	int DIM=3;
	
	nodex_update=node[k*DIM+0]*cos(U0*timestep+node[k*DIM+1])+0.5*d_GlobalLatticeNum[0];
	nodey_update=node[k*DIM+0]*sin(U0*timestep+node[k*DIM+1])+0.5*d_GlobalLatticeNum[1];
	nodez_update=node[k*DIM+2];
	
	i_m=int(nodex_update);
	i_p=i_m+1;
	j_m=int(nodey_update);
	j_p=j_m+1;
	k_m=int(nodez_update);
	k_p=k_m+1;
	
	if(i_m>d_baseLatticeNum[0]-1&&j_m>=d_baseLatticeNum[1]-1&&k_m>=d_baseLatticeNum[2]-1
	&&i_p<=d_baseLatticeNum[0]+nx-1&&j_p<=d_baseLatticeNum[1]+ny-1&&k_p<=d_baseLatticeNum[2]+nz-1)
	{
		int I_m,J_m,K_m,I_p,J_p,K_p;
		I_m=i_m-d_baseLatticeNum[0]+1;
		J_m=j_m-d_baseLatticeNum[1]+1;
		K_m=k_m-d_baseLatticeNum[2]+1;

		I_p=i_p-d_baseLatticeNum[0]+1;
		J_p=j_p-d_baseLatticeNum[1]+1;
		K_p=k_p-d_baseLatticeNum[2]+1;


		Fsf[K_m*nx*ny+J_m*nx+I_m]+=F[NDIM*k]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_m);
		Fsf[K_m*nx*ny+J_m*nx+I_p]+=F[NDIM*k]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_m);
		Fsf[K_m*nx*ny+J_p*nx+I_m]+=F[NDIM*k]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_m);
		Fsf[K_m*nx*ny+J_p*nx+I_p]+=F[NDIM*k]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_m);
		Fsf[K_p*nx*ny+J_m*nx+I_m]+=F[NDIM*k]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_p);
		Fsf[K_p*nx*ny+J_m*nx+I_p]+=F[NDIM*k]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_p);
		Fsf[K_p*nx*ny+J_p*nx+I_m]+=F[NDIM*k]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_p);
		Fsf[K_p*nx*ny+J_p*nx+I_p]+=F[NDIM*k]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_p);

		Fsf[K_m*nx*ny+J_m*nx+I_m+N]+=F[NDIM*k+1]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_m);
		Fsf[K_m*nx*ny+J_m*nx+I_p+N]+=F[NDIM*k+1]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_m);
		Fsf[K_m*nx*ny+J_p*nx+I_m+N]+=F[NDIM*k+1]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_m);
		Fsf[K_m*nx*ny+J_p*nx+I_p+N]+=F[NDIM*k+1]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_m);
		Fsf[K_p*nx*ny+J_m*nx+I_m+N]+=F[NDIM*k+1]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_p);
		Fsf[K_p*nx*ny+J_m*nx+I_p+N]+=F[NDIM*k+1]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_p);
		Fsf[K_p*nx*ny+J_p*nx+I_m+N]+=F[NDIM*k+1]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_p);
		Fsf[K_p*nx*ny+J_p*nx+I_p+N]+=F[NDIM*k+1]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_p);

		Fsf[K_m*nx*ny+J_m*nx+I_m+2*N]+=F[NDIM*k+2]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_m);
		Fsf[K_m*nx*ny+J_m*nx+I_p+2*N]+=F[NDIM*k+2]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_m);
		Fsf[K_m*nx*ny+J_p*nx+I_m+2*N]+=F[NDIM*k+2]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_m);
		Fsf[K_m*nx*ny+J_p*nx+I_p+2*N]+=F[NDIM*k+2]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_m);
		Fsf[K_p*nx*ny+J_m*nx+I_m+2*N]+=F[NDIM*k+2]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_p);
		Fsf[K_p*nx*ny+J_m*nx+I_p+2*N]+=F[NDIM*k+2]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_p);
		Fsf[K_p*nx*ny+J_p*nx+I_m+2*N]+=F[NDIM*k+2]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_p);
		Fsf[K_p*nx*ny+J_p*nx+I_p+2*N]+=F[NDIM*k+2]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_p);
	}
	
	
}
__global__ void	 WallForceonFluid (int NW,int nx,int ny,int nz,float* node,float* F,int* d_baseLatticeNum,int* d_GlobalLatticeNum,float* Fsf)
{
	int num_threads = blockDim.x;
	// local thread index
	int tx = threadIdx.x;
	// Block index in x
	int bx = blockIdx.x;

	int xStart = tx;
	
	int yStart = bx;

	int xlatticeId, ylatticeId, zlatticeId;
	int k = yStart*num_threads+ xStart;

	int N=nx*ny*nz;
	if(k>=NW)
	return;
	
	float nodex_update,nodey_update,nodez_update;
	int i_m,i_p,j_m,j_p,k_m,k_p;
	int DIM=3;
	
	nodex_update=node[k*DIM+0];
	nodey_update=node[k*DIM+1];
	nodez_update=node[k*DIM+2];
	
	i_m=int(nodex_update);
	i_p=i_m+1;
	j_m=int(nodey_update);
	j_p=j_m+1;
	k_m=int(nodez_update);
	k_p=k_m+1;
	
	if(i_m>=d_baseLatticeNum[0]-1&&j_m>=d_baseLatticeNum[1]-1&&k_m>=d_baseLatticeNum[2]-1
	&&i_p<=d_baseLatticeNum[0]+nx-1&&j_p<=d_baseLatticeNum[1]+ny-1&&k_p<=d_baseLatticeNum[2]+nz-1)
	{
		int I_m,J_m,K_m,I_p,J_p,K_p;
		I_m=i_m-d_baseLatticeNum[0]+1;
		J_m=j_m-d_baseLatticeNum[1]+1;
		K_m=k_m-d_baseLatticeNum[2]+1;

		I_p=i_p-d_baseLatticeNum[0]+1;
		J_p=j_p-d_baseLatticeNum[1]+1;
		K_p=k_p-d_baseLatticeNum[2]+1;


		Fsf[K_m*nx*ny+J_m*nx+I_m]+=F[NDIM*k]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_m);
		Fsf[K_m*nx*ny+J_m*nx+I_p]+=F[NDIM*k]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_m);
		Fsf[K_m*nx*ny+J_p*nx+I_m]+=F[NDIM*k]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_m);
		Fsf[K_m*nx*ny+J_p*nx+I_p]+=F[NDIM*k]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_m);
		Fsf[K_p*nx*ny+J_m*nx+I_m]+=F[NDIM*k]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_p);
		Fsf[K_p*nx*ny+J_m*nx+I_p]+=F[NDIM*k]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_p);
		Fsf[K_p*nx*ny+J_p*nx+I_m]+=F[NDIM*k]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_p);
		Fsf[K_p*nx*ny+J_p*nx+I_p]+=F[NDIM*k]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_p);

		Fsf[K_m*nx*ny+J_m*nx+I_m+N]+=F[NDIM*k+1]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_m);
		Fsf[K_m*nx*ny+J_m*nx+I_p+N]+=F[NDIM*k+1]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_m);
		Fsf[K_m*nx*ny+J_p*nx+I_m+N]+=F[NDIM*k+1]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_m);
		Fsf[K_m*nx*ny+J_p*nx+I_p+N]+=F[NDIM*k+1]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_m);
		Fsf[K_p*nx*ny+J_m*nx+I_m+N]+=F[NDIM*k+1]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_p);
		Fsf[K_p*nx*ny+J_m*nx+I_p+N]+=F[NDIM*k+1]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_p);
		Fsf[K_p*nx*ny+J_p*nx+I_m+N]+=F[NDIM*k+1]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_p);
		Fsf[K_p*nx*ny+J_p*nx+I_p+N]+=F[NDIM*k+1]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_p);

		Fsf[K_m*nx*ny+J_m*nx+I_m+2*N]+=F[NDIM*k+2]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_m);
		Fsf[K_m*nx*ny+J_m*nx+I_p+2*N]+=F[NDIM*k+2]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_m);
		Fsf[K_m*nx*ny+J_p*nx+I_m+2*N]+=F[NDIM*k+2]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_m);
		Fsf[K_m*nx*ny+J_p*nx+I_p+2*N]+=F[NDIM*k+2]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_m);
		Fsf[K_p*nx*ny+J_m*nx+I_m+2*N]+=F[NDIM*k+2]*D(nodex_update,i_m)*D(nodey_update,j_m)*D(nodez_update,k_p);
		Fsf[K_p*nx*ny+J_m*nx+I_p+2*N]+=F[NDIM*k+2]*D(nodex_update,i_p)*D(nodey_update,j_m)*D(nodez_update,k_p);
		Fsf[K_p*nx*ny+J_p*nx+I_m+2*N]+=F[NDIM*k+2]*D(nodex_update,i_m)*D(nodey_update,j_p)*D(nodez_update,k_p);
		Fsf[K_p*nx*ny+J_p*nx+I_p+2*N]+=F[NDIM*k+2]*D(nodex_update,i_p)*D(nodey_update,j_p)*D(nodez_update,k_p);
	}
}
	 //update fluid velocity
__global__ void	 velocity_update (int nx,int ny,int nz,float* ux,float* uy,float* uz,float* Fsf)
{
	int num_threads = blockDim.x;
	// local thread index
	int tx = threadIdx.x;
	// Block index in x
	int bx = blockIdx.x;
	// Block index in y
	int by = blockIdx.y;
	int xoff=1,yoff=1,zoff=1;
	// Global x-Index
	int xStart = tx+ xoff;
	// Global y-Index
	int yStart = bx+ yoff;
	int zStart = by + zoff;
	int N=nx*ny*nz;
	
	//add this for thread id great than lattice number
	if(tx >= nx-2*xoff ) return;
		
	// Index k in 1D-arrays
	int k = nx*(ny*zStart+ yStart)+ xStart;

	ux[k]+=0.5*Fsf[k];
	uy[k]+=0.5*Fsf[k+N];
	uz[k]+=0.5*Fsf[k+2*N];
	
}
__device__ float minmod(float a)
{
	float temp=0,temp0,temp1; 
	
	//superBee
	if(a<=0)
	{
		temp=0;
	}
	else
	{
		if(2*a<1)
		temp0=2*a;
		else
		temp0=1;
		if(a<2)
		temp1=a;
		else
		temp1=2;
		if(temp0>temp1)
		temp=temp0;
		else
		temp=temp1;
	}
	return temp;
}
__device__ float TVD(float con, float conE,float conEE,float conW,float conWW,float conN,float conNN,float conS,float conSS,
float conT,float conTT,float conB,float conBB,float ux,float uy,float uz,float uxE,float uxW,float uyN,float uyS,float uzT,float uzB,float visco)
{
	float fmE,fmW,fmN,fmS,fmT,fmB;
	float temp;
	//TVD
	//E
	if(ux+uxE>=0)
	{
		if(con!=conE)
		fmE=0.5*(ux+uxE)*(con+0.5*minmod((con-conW)/(conE-con))*(conE-con));
		else
		fmE=0.5*(ux+uxE)*con;
	}
	else
	{
		if(con!=conE)
		fmE=0.5*(ux+uxE)*(conE+0.5*minmod((conEE-conE)/(conE-con))*(conE-con));
		else
		fmE=0.5*(ux+uxE)*conE;
	}
	//W		
	if(uxW+ux>=0)
	{
		if(con!=conW)
		fmW=0.5*(uxW+ux)*(conW+0.5*minmod((conW-conWW)/(con-conW))*(con-conW));
		else
		fmW=0.5*(uxW+ux)*conW;
	}
	else
	{
		if(con!=conW)
		fmW=0.5*(uxW+ux)*(con+0.5*minmod((conE-con)/(con-conW))*(con-conW));
		else
		fmW=0.5*(uxW+ux)*con;
	}
				
	//N
	if(uy+uyN>=0)
	{
		if(con!=conN)
		fmN=0.5*(uy+uyN)*(con+0.5*minmod((con-conS)/(conN-con))*(conN-con));
		else
		fmN=0.5*(uy+uyN)*con;
	}
	else
	{
		if(con!=conN)
		fmN=0.5*(uy+uyN)*(conN+0.5*minmod((conNN-conN)/(conN-con))*(conN-con));
		else
		fmN=0.5*(uy+uyN)*conN;
	}
	//S		
	if(uyS+uy>=0)
	{
		if(con!=conS)
		fmS=0.5*(uyS+uy)*(conS+0.5*minmod((conS-conSS)/(con-conS))*(con-conS));
		else
		fmS=0.5*(uyS+uy)*conS;
	}
	else
	{
		if(con!=conS)
		fmS=0.5*(uyS+uy)*(con+0.5*minmod((conN-con)/(con-conS))*(con-conS));
		else
		fmS=0.5*(uyS+uy)*con;
	}
		
	//T
	if(uz+uzT>=0)
	{
		if(con!=conT)
		fmT=0.5*(uz+uzT)*(con+0.5*minmod((con-conB)/(conT-con))*(conT-con));
		else
		fmT=0.5*(uz+uzT)*con;
	}
	else
	{
		if(con!=conT)
		fmT=0.5*(uz+uzT)*(conT+0.5*minmod((conTT-conT)/(conT-con))*(conT-con));
		else
		fmT=0.5*(uz+uzT)*conT;
	}
	//B		
	if(uzB+uz>=0)
	{
		if(con!=conB)
		fmB=0.5*(uzB+uz)*(conB+0.5*minmod((conB-conBB)/(con-conB))*(con-conB));
		else
		fmB=0.5*(uzB+uz)*conB;
	}
	else
	{
		if(con!=conB)
		fmB=0.5*(uzB+uz)*(con+0.5*minmod((conT-con)/(con-conB))*(con-conB));
		else
		fmB=0.5*(uzB+uz)*con;
	}
		
	temp=-(fmE-fmW+fmN-fmS+fmT-fmB)+sct*visco*(conE+conW+conN+conS+conT+conB-6.0*con);
	return temp;
}
__global__ void TransCal(int nx, int ny, int nz, int xoff, int yoff, int zoff, unsigned int* geoD,unsigned int* geoDT,
int pitch,float* con_d,float* ux_d,float* uy_d,float* uz_d,float* vt,int* baselatticenum, int* globallatticenum)
{
	int num_threads = blockDim.x;
	// local thread index
	int tx = threadIdx.x;
	// Block index in x
	int bx = blockIdx.x;
	// Block index in y
	int by = blockIdx.y;
	// Global x-Index
	int xStart = tx+ xoff;
	// Global y-Index
	int yStart = bx+ yoff;
	int zStart = by + zoff;
	int xlatticeId, ylatticeId, zlatticeId;
	
	//add this for thread id great than lattice number
	if(tx >= nx-2*xoff ) return;
	// Index k in 1D-arrays
	int k = nx*(ny*zStart+ yStart)+ xStart;
	int k1 = (nx+2)*((ny+2)*(zStart+1)+ (yStart+1))+ (xStart+1);
	
	int pitnum;
	pitnum = pitch/sizeof(float);
	xlatticeId = xStart + baselatticenum[0];
	ylatticeId = yStart + baselatticenum[1];
	zlatticeId = zStart + baselatticenum[2];
	
	int ke,kw,kn,ks,kt,kb;
	int ke1,kw1,kn1,ks1,kt1,kb1;
	int kee1,kww1,knn1,kss1,ktt1,kbb1;

	float ux,uy,uz;
	float uxE,uxW,uxN,uxS,uxT,uxB;
	float uyE,uyW,uyN,uyS,uyT,uyB;
	float uzE,uzW,uzN,uzS,uzT,uzB;

	float conW,conE,conN,conS,conT,conB;
	float conWW,conEE,conNN,conSS,conTT,conBB;
	float con;
	float visco;
	float s00,s01,s02,s12,s11,s22;

	ke=k+1;
	kw=k-1;
	kn=k+nx;
	ks=k-nx;
	kt=k+nx*ny;
	kb=k-nx*ny;
	
	ke1=k1+1;
	kw1=k1-1;
	kn1=k1+(nx+2);
	ks1=k1-(nx+2);
	kt1=k1+(nx+2)*(ny+2);
	kb1=k1-(nx+2)*(ny+2);
	
	kee1=k1+2;
	kww1=k1-2;
	knn1=k1+2*(nx+2);
	kss1=k1-2*(nx+2);
	ktt1=k1+2*(nx+2)*(ny+2);
	kbb1=k1-2*(nx+2)*(ny+2);
	
	if(geoD[k]==FLUID)
	{
	 ux=ux_d[k];
	 uy=uy_d[k];
	 uz=uz_d[k];
		
	 if(geoD[ke]==FLUID)
	 {
	  uxE=ux_d[ke];
	  uyE=uy_d[ke];
	  uzE=uz_d[ke];
	 }
	 else if(geoD[ke]== SOLID||geoD[ke]==WALL)
	 {
	  uxE=0;
	  uyE=0;
	  uzE=0;
	 }
	 else if (geoD[ke]== SETU)
	 {
	  uxE=U0;
	  uyE=0;
	  uzE=0;
	 }
	 else
	 {
	  uxE=ux;
	  uyE=uy;
	  uzE=uz;
	 }
		
	 if(geoD[kw]==FLUID)
	 {
	  uxW=ux_d[kw];
	  uyW=uy_d[kw];
	  uzW=uz_d[kw];
	 }
	 else if(geoD[kw]== SOLID||geoD[kw]==WALL)
	 {
	  uxW=0;
	  uyW=0;
	  uzW=0;
	 }
	 else if (geoD[kw]== SETU)
	 {
	  uxW=U0;
	  uyW=0;
	  uzW=0;
	 }
	 else
	 {
	  uxW=ux;
	  uyW=uy;
	  uzW=uz;
	 }
		
	 if(geoD[kn]==FLUID)
	 {
	  uxN=ux_d[kn];
	  uyN=uy_d[kn];
	  uzN=uz_d[kn];
	 }
	 else if(geoD[kn]== SOLID||geoD[kn]==WALL)
	 {
	  uxN=0;
	  uyN=0;
	  uzN=0;
	 }
	 else if (geoD[kn]== SETU)
	 {
	  uxN=U0;
	  uyN=0;
	  uzN=0;
	 }
	 else
	 {
	  uxN=ux;
	  uyN=uy;
	  uzN=uz;
	 }
		
	 if(geoD[ks]==FLUID)
	 {
	  uxS=ux_d[ks];
	  uyS=uy_d[ks];
	  uzS=uz_d[ks];
         }
	 else if(geoD[ks]== SOLID||geoD[ks]==WALL)
	 {
	  uxS=0;
	  uyS=0;
	  uzS=0;
	 }
	 else if (geoD[ks]== SETU)
	 {
	  uxS=U0;
	  uyS=0;
	  uzS=0;
	 }
	 else
	{
	  uxS=ux;
	  uyS=uy;
	  uzS=uz;
	}
		
	if(geoD[kt]==FLUID)
	{
	   uxT=ux_d[kt];
	   uyT=uy_d[kt];
	   uzT=uz_d[kt];
	}
	else if(geoD[kt]== SOLID||geoD[kt]==WALL)
	{
	   uxT=0;
	   uyT=0;
	   uzT=0;
	}
	else if (geoD[kt]== SETU)
	{
	   uxT=U0;
	   uyT=0;
	   uzT=0;
	}
	else
	{
	   uxT=ux;
	   uyT=uy;
	   uzT=uz;
	}
		
	if(geoD[kb]==FLUID)
	{
	   uxB=ux_d[kb];
	   uyB=uy_d[kb];
	   uzB=uz_d[kb];
	}
	else if(geoD[kb]== SOLID||geoD[kb]==WALL)
	{
	   uxB=0;
	   uyB=0;
	   uzB=0;
	}
	else if (geoD[kb]== SETU)
	{
	   uxB=U0;
	   uyB=0;
	   uzB=0;
	}
	else
	{
	   uxB=ux;
	   uyB=uy;
	   uzB=uz;
	}
		
	con=con_d[k1];
		
	if(geoDT[ke1]==FLUID)
	{
	  conE=con_d[ke1];
	}
	
	else if (geoDT[ke1]== SETU)
	{
	  conE=con_in;
	}
	else
	{
	  conE=con;
	}
		
	if(geoDT[kw1]==FLUID)
	{
	  conW=con_d[kw1];
	}
	else if (geoDT[kw1]== SETU)
	{
	  conW=con_in;
	}
	else
	{
	  conW=con;
	}
		
	if(geoDT[kn1]==FLUID)
	{
	  conN=con_d[kn1];
	}
	else if (geoDT[kn1]== SETU)
	{
	  conN=con_in;
	}
	else
	{
	  conN=con;
	}
		
	if(geoDT[ks1]==FLUID)
	{
	  conS=con_d[ks1];
	}
	else if (geoDT[ks1]== SETU)
	{
	  conS=con_in;
	}
	else
	{
	  conS=con;
	}
		
	if(geoDT[kt1]==FLUID)
	{
	  conT=con_d[kt1];
	}
	else if (geoDT[kt1]== SETU)
	{
	  conT=con_in;
	}
	else
	{
	  conT=con;
	}
		
	if(geoDT[kb1]==FLUID)
	{
	  conB=con_d[kb1];
	}
	else if (geoDT[kb1]== SETU)
	{
	  conB=con_in;
	}
	else
	{
	  conB=con;
	}
		
	if(geoDT[kee1]==FLUID)
	{
	  conEE=con_d[kee1];
	}
	else if (geoDT[kee1]== SETU)
	{
	  conEE=con_in;
	}
	else
	{
	  conEE=conE;
	}
		
	if(geoDT[kww1]==FLUID)
	{
	  conWW=con_d[kww1];
	}
	else if (geoDT[kww1]== SETU)
	{
	  conWW=con_in;
	}
	else
	{
	  conWW=conW;
	}
		
	if(geoDT[knn1]==FLUID)
	{
	  conNN=con_d[knn1];
	}
	else if (geoDT[knn1]== SETU)
	{
	  conNN=con_in;
	}
	else
	{
	  conNN=conN;
	}
		
	if(geoDT[kss1]==FLUID)
	{
	  conSS=con_d[kss1];
	}
	else if (geoDT[kss1]== SETU)
	{
	  conSS=con_in;
	}
	else
	{
	  conSS=conS;
	}
		
	if(geoDT[ktt1]==FLUID)
	{
	  conTT=con_d[ktt1];
	}
	else if (geoDT[ktt1]== SETU)
	{
	  conTT=con_in;
	}
	else
	{
	  conTT=conT;
	}
		
	if(geoDT[kbb1]==FLUID)
	{
	  conBB=con_d[kbb1];
	}
	else if (geoDT[kbb1]== SETU)
	{
	  conBB=con_in;
	}
	else
	{
	  conBB=conB;
	}
		
	s00=0.5*(uxE-uxW);
	s11=0.5*(uyN-uyS);
	s22=0.5*(uzT-uzB);
	s01=0.25*(uxN-uxS+uyE-uyW);
	s02=0.25*(uxT-uxB+uzE-uzW);
	s12=0.25*(uyT-uyB+uzN-uzS);
		
	visco=sqrt(2*(s00*s00+s11*s11+s22*s22+2.0*s01*s01+2.0*s02*s02+2.0*s12*s12));
		
	con=con+TVD(con,conE,conEE,conW,conWW,conN,conNN,conS,conSS,
			conT,conTT,conB,conBB,ux,uy,uz,uxE,uxW,uyN,uyS,uzT,uzB,visco);
         vt[k]=visco;
	 con_d[k1]=con;	
	}
	
}

////////////////Kernel functi(n for collosion and propogation for 3d case of parallel version  ///////////////////////////
__global__ void LBCollProp(int nx, int ny, int nz, int xoff, int yoff, int zoff, unsigned int* geoD, 
			int pitch, float* f0_c1, float* f1_c1 ,float* rho_d,float* con_d,float* ux_d,float* uy_d,float* uz_d,float *Fsf,float *vt, int* baselatticenum, int* globallatticenum)
{
	int num_threads = blockDim.x;
	// local thread index
	int tx = threadIdx.x;
	// Block index in x
	int bx = blockIdx.x;
	// Block index in y
	int by = blockIdx.y;
	// Global x-Index
	int xStart = tx+ xoff;
	// Global y-Index
	int yStart = bx+ yoff;
	int zStart = by + zoff;
	int xlatticeId, ylatticeId, zlatticeId;
	int NodeNum=nx*ny*nz; 
	
	//add this for thread id great than lattice number
	if(tx >= nx-2*xoff ) return;
		
	// Index k in 1D-arrays
	int k = nx*(ny*zStart+ yStart)+ xStart;
	float rho1, vx1, vy1, vz1, visco,square, dummy;
	float m_eq0, m_eq1, m_eq2, m_eq3, m_eq4, m_eq5, m_eq6, m_eq7, m_eq8, m_eq9,
	      m_eq10, m_eq11, m_eq12, m_eq13, m_eq14, m_eq15, m_eq16, m_eq17, m_eq18;
	float m_0, m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8, m_9,
	      m_10, m_11, m_12, m_13, m_14, m_15, m_16, m_17, m_18;
	float F_0, F_1, F_2, F_3, F_4, F_5, F_6, F_7, F_8, F_9,
	      F_10, F_11, F_12, F_13, F_14, F_15, F_16, F_17, F_18;
	float rho;
	float Fx,Fy,Fz;
	
	float d19,d2394,d252,d36,d72,d12,d24;
        float mybodyf=bodyf;
	//Shared memory for propagation
	__shared__ float F_OUT_E1[THREAD_NUM+1];
	__shared__ float F_OUT_W1[THREAD_NUM+1];
	
	__shared__ float F_OUT_NE1[THREAD_NUM+1];
	__shared__ float F_OUT_NW1[THREAD_NUM+1];
	__shared__ float F_OUT_SW1[THREAD_NUM+1];
	__shared__ float F_OUT_SE1[THREAD_NUM+1];
	
	__shared__ float F_OUT_TE1[THREAD_NUM+1];
	__shared__ float F_OUT_BE1[THREAD_NUM+1];
	__shared__ float F_OUT_BW1[THREAD_NUM+1];
	__shared__ float F_OUT_TW1[THREAD_NUM+1];
	
	//load fr0[k],fe0[k],fn0[k],fw0[k],fs0[k],fne0[k],fnw0[k],fsw0[k],fse0[k] to local variables 
	float F_IN_R1, F_IN_E1, F_IN_N1, F_IN_W1, F_IN_S1, F_IN_T1, F_IN_B1, F_IN_NE1, F_IN_NW1, F_IN_SW1, F_IN_SE1,
	   	F_IN_NT1, F_IN_ST1, F_IN_SB1,F_IN_NB1, F_IN_TE1, F_IN_BE1, F_IN_BW1, F_IN_TW1;	 

	int nxny;
	int kn, ks, kt, kb, knt, kst, ksb, knb;
	int kw, ke, kse, ksw, knw, kne, ktw,kte, kbw, kbe; // only used when there are more than 1 process in X direction
	int pitnum;
	
	pitnum = pitch/sizeof(float);
	
	xlatticeId = xStart + baselatticenum[0];
	ylatticeId = yStart + baselatticenum[1];
	zlatticeId = zStart + baselatticenum[2];
	
	F_IN_R1 = f0_c1[k];
	
	F_IN_E1 = f0_c1[k+1*pitnum];	
	F_IN_W1 = f0_c1[k+2*pitnum];
	F_IN_N1 = f0_c1[k+3*pitnum];
	F_IN_S1 = f0_c1[k+4*pitnum];
	F_IN_T1 = f0_c1[k+5*pitnum];
	F_IN_B1 = f0_c1[k+6*pitnum];
	
	F_IN_NE1 = f0_c1[k+7*pitnum];
	F_IN_NW1 = f0_c1[k+8*pitnum];
	F_IN_SW1 = f0_c1[k+9*pitnum];
	F_IN_SE1 = f0_c1[k+10*pitnum];
	
	F_IN_NT1 = f0_c1[k+11*pitnum];
	F_IN_ST1 = f0_c1[k+12*pitnum];
	F_IN_SB1 = f0_c1[k+13*pitnum];
	F_IN_NB1 = f0_c1[k+14*pitnum];
	
	F_IN_TE1 = f0_c1[k+15*pitnum];
	F_IN_BE1 = f0_c1[k+16*pitnum];
	F_IN_BW1 = f0_c1[k+17*pitnum];
	F_IN_TW1 = f0_c1[k+18*pitnum];

	rho1 = rho_d[k];
	rho = rho1;
	
	vx1 = ux_d[k];
	vy1 = uy_d[k];
	vz1 = uz_d[k];
	visco = vt[k];
      
	visco =1.0/(1.0/S_d[9]+3.0*0);
	
	d19=1.0/19;
	d2394=1.0/2394;
	d252=1.0/252;
	d36=1.0/36;
	d72=1.0/72;
	d12=1.0/12;
	d24=0.5*d12;
	
	Fx=0.5*Fsf[k];
	Fy=0.5*Fsf[k+NodeNum];
	Fz=0.5*Fsf[k+2*NodeNum];

	
	if(geoD[k] == FLUID || geoD[k]==WALL)
	{
	//collision:modify F_IN_R,F_IN_E,...,F_IN_SE
	// fluid node: compute Omega_i (\rho,\vec v) Omega_i depends on f_i^eq and f_i, to compute f_i^eq you need density and velocities 		
		// ***** compute all f_i^eq from rho and v *****
	 	
		m_0=F_IN_R1 + F_IN_E1 + F_IN_W1 + F_IN_N1 + F_IN_S1 +F_IN_T1 + F_IN_B1+ F_IN_NE1+ F_IN_NW1+ F_IN_SW1+ F_IN_SE1+
	   	F_IN_NT1+ F_IN_ST1 + F_IN_SB1 + F_IN_NB1+ F_IN_TE1+ F_IN_BE1+ F_IN_BW1+ F_IN_TW1;

		m_1=-30.*F_IN_R1 -11.*(F_IN_E1 + F_IN_N1 + F_IN_W1 + F_IN_S1 +F_IN_T1 + F_IN_B1)+ 8.*(F_IN_NE1+ F_IN_NW1+ F_IN_SW1+ F_IN_SE1+
	   	F_IN_NT1+ F_IN_ST1 + F_IN_SB1 + F_IN_NB1+ F_IN_TE1+ F_IN_BE1+ F_IN_BW1+ F_IN_TW1);

		m_2=12.*F_IN_R1 -4.*(F_IN_E1 + F_IN_N1 + F_IN_W1 + F_IN_S1 +F_IN_T1 + F_IN_B1)+ F_IN_NE1+ F_IN_NW1+ F_IN_SW1+ F_IN_SE1+
	   	F_IN_NT1+ F_IN_ST1 + F_IN_SB1 + F_IN_NB1+ F_IN_TE1+ F_IN_BE1+ F_IN_BW1+ F_IN_TW1;

		m_3=F_IN_E1 - F_IN_W1 + F_IN_NE1- F_IN_NW1- F_IN_SW1+ F_IN_SE1+F_IN_TE1+ F_IN_BE1- F_IN_BW1- F_IN_TW1;

	        m_4=-4.*(F_IN_E1 - F_IN_W1) + F_IN_NE1- F_IN_NW1- F_IN_SW1+ F_IN_SE1+F_IN_TE1+ F_IN_BE1- F_IN_BW1- F_IN_TW1;

		m_5=F_IN_N1 - F_IN_S1 + F_IN_NE1+ F_IN_NW1- F_IN_SW1- F_IN_SE1+F_IN_NT1- F_IN_ST1- F_IN_SB1 + F_IN_NB1;

		m_6=-4*(F_IN_N1 - F_IN_S1) + F_IN_NE1+ F_IN_NW1- F_IN_SW1- F_IN_SE1+F_IN_NT1- F_IN_ST1- F_IN_SB1 + F_IN_NB1;

		m_7=F_IN_T1 - F_IN_B1+ F_IN_NT1+ F_IN_ST1 - F_IN_SB1 - F_IN_NB1+ F_IN_TE1- F_IN_BE1- F_IN_BW1+ F_IN_TW1;

		m_8=-4*(F_IN_T1 - F_IN_B1)+ F_IN_NT1+ F_IN_ST1 - F_IN_SB1 - F_IN_NB1+ F_IN_TE1- F_IN_BE1- F_IN_BW1+ F_IN_TW1;

		m_9=2.0*(F_IN_E1 + F_IN_W1) - (F_IN_N1 + F_IN_S1 +F_IN_T1 + F_IN_B1)+ F_IN_NE1+ F_IN_NW1+ F_IN_SW1+ F_IN_SE1-
	   	2.0*(F_IN_NT1+ F_IN_ST1 + F_IN_SB1 + F_IN_NB1)+ F_IN_TE1+ F_IN_BE1+ F_IN_BW1+ F_IN_TW1;

		m_10=-4.*(F_IN_E1 + F_IN_W1) +2.0* (F_IN_N1 + F_IN_S1 +F_IN_T1 + F_IN_B1)+ F_IN_NE1+ F_IN_NW1+ F_IN_SW1+ F_IN_SE1-
	   	2.0*(F_IN_NT1+ F_IN_ST1 + F_IN_SB1 + F_IN_NB1)+ F_IN_TE1+ F_IN_BE1+ F_IN_BW1+ F_IN_TW1;

		m_11= F_IN_N1 + F_IN_S1 -F_IN_T1 - F_IN_B1+ F_IN_NE1+ F_IN_NW1+ F_IN_SW1+ F_IN_SE1-( F_IN_TE1+ F_IN_BE1+ F_IN_BW1+ F_IN_TW1);

		m_12= -2.*(F_IN_N1 + F_IN_S1) +2.0*(F_IN_T1 + F_IN_B1)+ F_IN_NE1+ F_IN_NW1+ F_IN_SW1+ F_IN_SE1-( F_IN_TE1+ F_IN_BE1+ F_IN_BW1+ F_IN_TW1);

		m_13= F_IN_NE1- F_IN_NW1+ F_IN_SW1- F_IN_SE1;
		m_14=F_IN_NT1- F_IN_ST1 + F_IN_SB1 - F_IN_NB1;
		m_15=F_IN_TE1- F_IN_BE1+ F_IN_BW1- F_IN_TW1;
		m_16= F_IN_NE1- F_IN_NW1- F_IN_SW1+ F_IN_SE1- F_IN_TE1- F_IN_BE1+ F_IN_BW1+ F_IN_TW1;
		m_17= -F_IN_NE1- F_IN_NW1+ F_IN_SW1+ F_IN_SE1+F_IN_NT1- F_IN_ST1 - F_IN_SB1 + F_IN_NB1;
		m_18=-F_IN_NT1- F_IN_ST1 + F_IN_SB1 + F_IN_NB1+ F_IN_TE1- F_IN_BE1- F_IN_BW1+ F_IN_TW1;

		square = vx1*vx1 + vy1*vy1 + vz1*vz1;
		m_eq0 = rho1;
		
		m_eq1 = rho1*(-11.+19.*square);
		m_eq2 = rho1*(3-5.5*square);
		m_eq3 = rho1*vx1;
		m_eq4 = -2./3.*m_eq3;
		m_eq5 = rho1*vy1;
		m_eq6 = -2./3.*m_eq5;
		m_eq7 = rho1*vz1;
		m_eq8 = -2./3.*m_eq7;
		m_eq9 = rho1*(3.*vx1*vx1-square);
		m_eq10 = -0.5*m_eq9;
		m_eq11 = rho1*(vy1*vy1-vz1*vz1);
		m_eq12 = -0.5*m_eq11;
		m_eq13 = rho1* (vx1*vy1);
		m_eq14 = rho1* (vy1*vz1);
		m_eq15 = rho1* (vz1*vx1);
		m_eq16 = 0;
		m_eq17 = 0;
		m_eq18 = 0;

		F_0=0.;
		F_1=38.*mybodyf*vz1+38.*(Fx*vx1+Fy*vy1+Fz*vz1);		
		F_2=-11.*mybodyf*vz1-11.*(Fx*vx1+Fy*vy1+Fz*vz1);		
		F_3=0+Fx;		
		F_4=0-2./3.*Fx;		
		F_5=0+Fy;		
		F_6=0-2./3.*Fy;		
		F_7=mybodyf+Fz;		
		F_8=-2./3.*(mybodyf+Fz);
		F_9=-2.0*mybodyf*vz1+6*Fx*vx1-2.*Fx*vx1-2.*Fy*vy1-2.*Fz*vz1;
		F_10=-0.5*F_9;
		F_11=-2.0*(mybodyf+Fz)*vz1+2.0*Fy*vy1;
		F_12=-0.5*F_11;
		F_13=Fx*vy1+Fy*vx1;
		F_14=(mybodyf+Fz)*vy1+Fy*vz1;
		F_15=(mybodyf+Fz)*vx1+Fx*vz1;
		F_16=0;
		F_17=0;
		F_18=0;

		m_0=m_0+S_d[0]*(m_eq0-m_0)+(1-0.5*S_d[0])*F_0;
		m_1=m_1+S_d[1]*(m_eq1-m_1)+(1-0.5*S_d[1])*F_1;
		m_2=m_2+S_d[2]*(m_eq2-m_2)+(1-0.5*S_d[2])*F_2;
		m_3=m_3+S_d[3]*(m_eq3-m_3)+(1-0.5*S_d[3])*F_3;
		m_4=m_4+S_d[4]*(m_eq4-m_4)+(1-0.5*S_d[4])*F_4;
		m_5=m_5+S_d[5]*(m_eq5-m_5)+(1-0.5*S_d[5])*F_5;
		m_6=m_6+S_d[6]*(m_eq6-m_6)+(1-0.5*S_d[6])*F_6;
		m_7=m_7+S_d[7]*(m_eq7-m_7)+(1-0.5*S_d[7])*F_7;
		m_8=m_8+S_d[8]*(m_eq8-m_8)+(1-0.5*S_d[8])*F_8;
		m_9=m_9+visco*(m_eq9-m_9)+(1-0.5*visco)*F_9;

		m_10=m_10+S_d[10]*(m_eq10-m_10)+(1-0.5*S_d[10])*F_10;
		m_11=m_11+visco*(m_eq11-m_11)+(1-0.5*visco)*F_11;
		m_12=m_12+S_d[12]*(m_eq12-m_12)+(1-0.5*S_d[12])*F_12;
		m_13=m_13+visco*(m_eq13-m_13)+(1-0.5*visco)*F_13;
		m_14=m_14+visco*(m_eq14-m_14)+(1-0.5*visco)*F_14;
		m_15=m_15+visco*(m_eq15-m_15)+(1-0.5*visco)*F_15;
		m_16=m_16+S_d[16]*(m_eq16-m_16)+(1-0.5*S_d[16])*F_16;
		m_17=m_17+S_d[17]*(m_eq17-m_17)+(1-0.5*S_d[17])*F_17;
		m_18=m_18+S_d[18]*(m_eq18-m_18)+(1-0.5*S_d[18])*F_18;


		// modify distributions according to collision contribution the term after +=is \Omega_i	
		F_IN_R1 = d19*m_0-30.*d2394*m_1+12.*d252*m_2;
		
		F_IN_E1 = d19*m_0-11.*d2394*m_1-4.*d252*m_2+0.1*m_3-0.1*m_4+2.*d36*m_9-4.*d72*m_10;
		F_IN_W1 = d19*m_0-11.*d2394*m_1-4.*d252*m_2-0.1*m_3+0.1*m_4+2.*d36*m_9-4.*d72*m_10;
		F_IN_N1 = d19*m_0-11.*d2394*m_1-4.*d252*m_2+0.1*m_5-0.1*m_6-d36*m_9+2.*d72*m_10+d12*m_11-2.*d24*m_12;
		F_IN_S1 = d19*m_0-11.*d2394*m_1-4.*d252*m_2-0.1*m_5+0.1*m_6-d36*m_9+2.*d72*m_10+d12*m_11-2.*d24*m_12;
		F_IN_T1 = d19*m_0-11.*d2394*m_1-4.*d252*m_2+0.1*m_7-0.1*m_8-d36*m_9+2.*d72*m_10-d12*m_11+2.*d24*m_12;
		F_IN_B1 = d19*m_0-11.*d2394*m_1-4.*d252*m_2-0.1*m_7+0.1*m_8-d36*m_9+2.*d72*m_10-d12*m_11+2.*d24*m_12;
		
		F_IN_NE1 = d19*m_0+8.*d2394*m_1+d252*m_2+0.1*m_3+0.025*m_4+0.1*m_5+0.025*m_6+d36*m_9+d72*m_10+d12*m_11+d24*m_12+0.25*m_13+0.125*m_16-0.125*m_17;
		F_IN_NW1 = d19*m_0+8.*d2394*m_1+d252*m_2-0.1*m_3-0.025*m_4+0.1*m_5+0.025*m_6+d36*m_9+d72*m_10+d12*m_11+d24*m_12-0.25*m_13-0.125*m_16-0.125*m_17;
		F_IN_SW1 = d19*m_0+8.*d2394*m_1+d252*m_2-0.1*m_3-0.025*m_4-0.1*m_5-0.025*m_6+d36*m_9+d72*m_10+d12*m_11+d24*m_12+0.25*m_13-0.125*m_16+0.125*m_17;
		F_IN_SE1 = d19*m_0+8.*d2394*m_1+d252*m_2+0.1*m_3+0.025*m_4-0.1*m_5-0.025*m_6+d36*m_9+d72*m_10+d12*m_11+d24*m_12-0.25*m_13+0.125*m_16+0.125*m_17;
		
		F_IN_NT1 = d19*m_0+8.*d2394*m_1+d252*m_2+0.1*m_5+0.025*m_6+0.1*m_7+0.025*m_8-2.0*d36*m_9-2.0*d72*m_10+0.25*m_14+0.125*m_17-0.125*m_18;
		F_IN_ST1 = d19*m_0+8.*d2394*m_1+d252*m_2-0.1*m_5-0.025*m_6+0.1*m_7+0.025*m_8-2.0*d36*m_9-2.0*d72*m_10-0.25*m_14-0.125*m_17-0.125*m_18;
		F_IN_SB1 = d19*m_0+8.*d2394*m_1+d252*m_2-0.1*m_5-0.025*m_6-0.1*m_7-0.025*m_8-2.0*d36*m_9-2.0*d72*m_10+0.25*m_14-0.125*m_17+0.125*m_18;
		F_IN_NB1 = d19*m_0+8.*d2394*m_1+d252*m_2+0.1*m_5+0.025*m_6-0.1*m_7-0.025*m_8-2.0*d36*m_9-2.0*d72*m_10-0.25*m_14+0.125*m_17+0.125*m_18;
		
		F_IN_TE1 = d19*m_0+8.*d2394*m_1+d252*m_2+0.1*m_3+0.025*m_4+0.1*m_7+0.025*m_8+d36*m_9+d72*m_10-d12*m_11-d24*m_12+0.25*m_15-0.125*m_16+0.125*m_18;
		F_IN_BE1 = d19*m_0+8.*d2394*m_1+d252*m_2+0.1*m_3+0.025*m_4-0.1*m_7-0.025*m_8+d36*m_9+d72*m_10-d12*m_11-d24*m_12-0.25*m_15-0.125*m_16-0.125*m_18;
		F_IN_BW1 = d19*m_0+8.*d2394*m_1+d252*m_2-0.1*m_3-0.025*m_4-0.1*m_7-0.025*m_8+d36*m_9+d72*m_10-d12*m_11-d24*m_12+0.25*m_15+0.125*m_16-0.125*m_18;
		F_IN_TW1 = d19*m_0+8.*d2394*m_1+d252*m_2-0.1*m_3-0.025*m_4+0.1*m_7+0.025*m_8+d36*m_9+d72*m_10-d12*m_11-d24*m_12-0.25*m_15+0.125*m_16+0.125*m_18;
	
	}else if(geoD[k] == SOLID)
	{	//bounce back: modify F_IN_R,F_IN_E,...,F_IN_SE, swap antiparallel distributions 
		dummy = F_IN_E1;
		F_IN_E1 = F_IN_W1;
		F_IN_W1 = dummy;
		
		dummy = F_IN_N1;
		F_IN_N1 = F_IN_S1;
		F_IN_S1 = dummy;
		
		dummy = F_IN_B1;
		F_IN_B1 = F_IN_T1;
		F_IN_T1 = dummy;
		
		dummy = F_IN_NE1;
		F_IN_NE1 = F_IN_SW1;
		F_IN_SW1 = dummy;
		
		dummy = F_IN_NW1;
		F_IN_NW1 = F_IN_SE1;
		F_IN_SE1 = dummy;
		
		dummy = F_IN_NT1;
		F_IN_NT1 = F_IN_SB1;
		F_IN_SB1 = dummy;
		
		dummy = F_IN_ST1;
		F_IN_ST1 = F_IN_NB1;
		F_IN_NB1 = dummy;
		
		dummy = F_IN_TE1;
		F_IN_TE1 = F_IN_BW1;
		F_IN_BW1 = dummy;
		
		dummy = F_IN_BE1;
		F_IN_BE1 = F_IN_TW1;
		F_IN_TW1 = dummy;	
		
	}
	
	//Propagation using shared memory for distributions having a shift in east or west direction
	F_OUT_E1[tx+1] = F_IN_E1;
	F_OUT_NE1[tx+1] = F_IN_NE1;
	F_OUT_SE1[tx+1] = F_IN_SE1;
	F_OUT_TE1[tx+1] = F_IN_TE1;
	F_OUT_BE1[tx+1] = F_IN_BE1;
	
	if(tx == 0)
	{
	  F_OUT_E1[tx] = F_IN_E1;
	  F_OUT_NE1[tx] = F_IN_NE1;
	  F_OUT_SE1[tx] = F_IN_SE1;
	  F_OUT_TE1[tx] = F_IN_TE1;
	  F_OUT_BE1[tx] = F_IN_BE1;	
	}

	  F_OUT_W1 [tx] = F_IN_W1;
	  F_OUT_NW1[tx] = F_IN_NW1;
	  F_OUT_SW1[tx] = F_IN_SW1;
	  F_OUT_TW1[tx] = F_IN_TW1;
	  F_OUT_BW1[tx] = F_IN_BW1;
	
	if(tx == nx - 2*xoff - 1)
	{
	  F_OUT_W1 [tx+1] = F_IN_W1;
	  F_OUT_NW1[tx+1] = F_IN_NW1;
	  F_OUT_SW1[tx+1] = F_IN_SW1;
	  F_OUT_TW1[tx+1] = F_IN_TW1;
	  F_OUT_BW1[tx+1] = F_IN_BW1;
	}	
		
	// synchronize threads
	__syncthreads();
	
	f1_c1[k] = F_IN_R1;
	
	nxny = nx*ny;
	
	kn = k + nx;
	ks = k - nx;
	kt = k + nxny;
	kb = k - nxny;
	
	kst = k + nxny - nx;
	knt = k + nxny + nx;
	knb = k - nxny + nx;
	ksb = k - nxny - nx; 
	
	if(xoff >0) 
	{
	ke = k + 1;
	kw = k - 1;
	
	kne = k + 1 + nx; 
	knw = k - 1 + nx;
	kse = k + 1 - nx;
	ksw = k - 1 - nx;
	
	kte = k + 1 + nxny;
	ktw = k - 1 + nxny;
	kbe = k + 1 - nxny;
	kbw = k - 1 - nxny;
	}
	
	f1_c1[k + 1*pitnum] = F_OUT_E1[tx];
	f1_c1[k + 2*pitnum] = F_OUT_W1[tx+1];

	f1_c1[kt + 5*pitnum] = F_IN_T1;
	f1_c1[kb + 6*pitnum] = F_IN_B1;
	
	f1_c1[kn + 3*pitnum] = F_IN_N1;
	f1_c1[ks + 4*pitnum] = F_IN_S1;
	
	f1_c1[knt+11*pitnum] = F_IN_NT1;
        f1_c1[kst+12*pitnum] = F_IN_ST1;
	f1_c1[ksb+13*pitnum] = F_IN_SB1;
	f1_c1[knb+14*pitnum] = F_IN_NB1;
	
	if(xStart > xoff)
	{
	 f1_c1[kn+7*pitnum] = F_OUT_NE1[tx];
	 f1_c1[ks+10*pitnum] = F_OUT_SE1[tx];
	 f1_c1[kt+15*pitnum] = F_OUT_TE1[tx];
	 f1_c1[kb+16*pitnum] = F_OUT_BE1[tx];
	}else if (xoff >0)// this part is for xstartoff >0, when parallel decomposion in X direction
	{
	 f1_c1[kw+ 2*pitnum] = F_IN_W1;
	 
	 f1_c1[knw+ 8*pitnum] = F_IN_NW1;
	 f1_c1[ksw+ 9*pitnum] = F_IN_SW1;
	 f1_c1[ktw+ 18*pitnum] = F_IN_TW1;
	 f1_c1[kbw+ 17*pitnum] = F_IN_BW1;	 
	} 
	
	if(xStart < nx - xoff -1)
	{
	 f1_c1[kb+17*pitnum] = F_OUT_BW1[tx+1];
	 f1_c1[kt+18*pitnum] = F_OUT_TW1[tx+1];
	 f1_c1[kn+8*pitnum] = F_OUT_NW1[tx+1];
	 f1_c1[ks+9*pitnum] = F_OUT_SW1[tx+1];
	}else if( xoff >0)
	{
	 f1_c1[ke + 1*pitnum] = F_IN_E1;
	  
	 f1_c1[kne + 7*pitnum] = F_IN_NE1;
	 f1_c1[kse + 10*pitnum] = F_IN_SE1;
	 f1_c1[kte + 15*pitnum] = F_IN_TE1;
	 f1_c1[kbe + 16*pitnum] = F_IN_BE1;
	}  
        
  if(geoD[k] == SOLID)
	{
	if( zlatticeId == globallatticenum[2] + zoff -1 ) 
	{
	      f1_c1[k+6*pitnum] = F_IN_B1;  //these five distribution function can not be propaged from adjacent lattice
        f1_c1[k+16*pitnum] = F_IN_BE1;
        f1_c1[k+17*pitnum] = F_IN_BW1;
        
        f1_c1[k+13*pitnum] = F_IN_SB1;
        f1_c1[k+14*pitnum] = F_IN_NB1;
	}else if( zlatticeId == zoff)
	{			
	f1_c1[k+5*pitnum] = F_IN_T1; //these five distribution function can not be propaged from adjacent lattice
	f1_c1[k+15*pitnum] = F_IN_TE1;
	f1_c1[k+18*pitnum] = F_IN_TW1;
	
	f1_c1[k+11*pitnum] = F_IN_NT1;
        f1_c1[k+12*pitnum] = F_IN_ST1;
	}
	
	if(ylatticeId == globallatticenum[1] + yoff -1)  //if yStart == ny
	{		
	f1_c1[k+4*pitnum] = F_IN_S1;
	f1_c1[k+9*pitnum] = F_IN_SW1;
	f1_c1[k+10*pitnum] = F_IN_SE1;
	
	f1_c1[k+12*pitnum] = F_IN_ST1;
	f1_c1[k+13*pitnum] = F_IN_SB1;
	  
	}else if (ylatticeId == yoff)//end of yStart == ny -1
	{
 	f1_c1[k+3*pitnum] = F_IN_N1;
	f1_c1[k+7*pitnum] = F_IN_NE1;
	f1_c1[k+8*pitnum] = F_IN_NW1;
	 
	f1_c1[k+11*pitnum] = F_IN_NT1;
	f1_c1[k+14*pitnum] = F_IN_NB1;
	}

	if(xlatticeId == globallatticenum[0] + xoff -1)
	{// for the distribution function in the x direction that can not be propogated 

	  f1_c1[k+17*pitnum] = F_IN_BW1;
	  f1_c1[k+18*pitnum] = F_IN_TW1;
	  f1_c1[k+2*pitnum] = F_IN_W1;
	  f1_c1[k+8*pitnum] = F_IN_NW1;
	  f1_c1[k+9*pitnum] = F_IN_SW1;
	}else if(xlatticeId == xoff) 
	{

	  f1_c1[k+15*pitnum] = F_IN_TE1;
	  f1_c1[k+16*pitnum] = F_IN_BE1;
	  f1_c1[k+1*pitnum] = F_IN_E1;
	  f1_c1[k+7*pitnum] = F_IN_NE1;
	  f1_c1[k+10*pitnum] = F_IN_SE1;
	}  	  
      }
      
   if(geoD[k]==WALL)
   {
      int k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15,k16,k17,k18;
      
      k1 = nx*(ny*zStart+ yStart)+ xStart+1;
      k2 = nx*(ny*zStart+ yStart)+ xStart-1;
      k3 = nx*(ny*zStart+ yStart+1)+ xStart;
      k4 = nx*(ny*zStart+ yStart-1)+ xStart;
      k5 = nx*(ny*(zStart+1)+ yStart)+ xStart;
      k6 = nx*(ny*(zStart-1)+ yStart)+ xStart;
      
      k7 = nx*(ny*zStart+ yStart+1)+ (xStart+1);
      k8 = nx*(ny*zStart+ yStart+1)+ (xStart-1);
      k9 = nx*(ny*zStart+ yStart-1)+ (xStart-1);
      k10 = nx*(ny*zStart+ yStart-1)+ (xStart+1);
      
      k11 = nx*(ny*(zStart+1)+ yStart+1)+ (xStart);
      k12 = nx*(ny*(zStart+1)+ yStart-1)+ (xStart);
      k13 = nx*(ny*(zStart-1)+ yStart-1)+ (xStart);
      k14 = nx*(ny*(zStart-1)+ yStart+1)+ (xStart);
      
      k15 = nx*(ny*(zStart+1)+ yStart)+ (xStart+1);
      k16 = nx*(ny*(zStart-1)+ yStart)+ (xStart+1);
      k17 = nx*(ny*(zStart-1)+ yStart)+ (xStart-1);
      k18 = nx*(ny*(zStart+1)+ yStart)+ (xStart-1);
      
			k1=geoD[k1];
			k2=geoD[k2];
			k3=geoD[k3];
			k4=geoD[k4];
			k5=geoD[k5];
			k6=geoD[k6];
			k7=geoD[k7];
			k8=geoD[k8];
			k9=geoD[k9];
			k10=geoD[k10];
			k11=geoD[k11];
			k12=geoD[k12];
			k13=geoD[k13];
			k14=geoD[k14];
			k15=geoD[k15];
			k16=geoD[k16];
			k17=geoD[k17];
			k18=geoD[k18];
			
			if(k1==FLUID)
			f1_c1[k+1*pitnum] = F_IN_W1;
			
			if(k2==FLUID)
			f1_c1[k+2*pitnum] = F_IN_E1;
			
			if(k3==FLUID)
			f1_c1[k+3*pitnum] = F_IN_S1;
			
			if(k4==FLUID)
			f1_c1[k+4*pitnum] = F_IN_N1;
			
			if(k5==FLUID)
			f1_c1[k+5*pitnum] = F_IN_B1;
			
			if(k6==FLUID)
			f1_c1[k+6*pitnum] = F_IN_T1;
			
			if(k7==FLUID)
			f1_c1[k+7*pitnum] = F_IN_SW1;
			
			if(k8==FLUID)
			f1_c1[k+8*pitnum] = F_IN_SE1;
			
			if(k9==FLUID)
			f1_c1[k+9*pitnum] = F_IN_NE1;
			
			if(k10==FLUID)
			f1_c1[k+10*pitnum] = F_IN_NW1;
			
			if(k11==FLUID)
			f1_c1[k+11*pitnum] = F_IN_SB1;
			
			if(k12==FLUID)
			f1_c1[k+12*pitnum] = F_IN_NB1;
			
			if(k13==FLUID)
			f1_c1[k+13*pitnum] = F_IN_NT1;
			
			if(k14==FLUID)
			f1_c1[k+14*pitnum] = F_IN_ST1;
			
			if(k15==FLUID)
			f1_c1[k+15*pitnum] = F_IN_BW1;
			
			if(k16==FLUID)
			f1_c1[k+16*pitnum] = F_IN_TW1;
			
			if(k17==FLUID)
			f1_c1[k+17*pitnum] = F_IN_TE1;
			
			if(k18==FLUID)
			f1_c1[k+18*pitnum] = F_IN_BE1;
   }	
}// end of kernel LBCollProp

//********************************Main function for LBM 2DQ9****************************
int main(int argc, char **argv)
{
	
	int step;
	
	int dir[2]= {1,0}; //transfer along the positive and negative direction of axis
	
	int i;

	int buffersize, min_dimlen;
	char fname[20];
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &h_numofproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &h_procid);
	M_h= (float*) malloc(sizeof(float)*DIRECTION_NUM*DIRECTION_NUM); 
	MI_h= (float*) malloc(sizeof(float)*DIRECTION_NUM*DIRECTION_NUM); 

	readparameter();	//read some parameters from file

	nodeR_h = (float *) malloc(sizeof(float)*NR*NDIM);
	nodeW_h = (float *) malloc(sizeof(float)*NW*NDIM);

	readnodes();
	setup_parallel(); // set up the information of parallel computation

	nx = h_GlobalLatticeNum[0]/h_procn[0];
	ny = h_GlobalLatticeNum[1]/h_procn[1];
	nz = h_GlobalLatticeNum[2]/h_procn[2];
		
	for(i=0; i<NDIM; i++)
	{
	   baseLatticeNum[i] = h_GlobalLatticeNum[i]*procpos[i]/h_procn[i];
	   h_procLatNum[i] = h_GlobalLatticeNum[i]/h_procn[i];
	}   
	
	if(h_procn[0] >1)
	{
          xstartoff = 1;
	}
	parallelnx = nx + 2*xstartoff;
	ystartoff = 1;
	parallelny = ny + 2*ystartoff;
	zstartoff = 1;
	parallelnz = nz + 2*zstartoff;
		   
	totallatticenum = parallelnx * parallelny * parallelnz;
	reallatticenum = nx * ny * nz;

	min_dimlen =  parallelnx <= parallelny ? parallelnx : parallelny;
	min_dimlen = min_dimlen <= parallelnz ? min_dimlen : parallelnz;

	buffersize = totallatticenum / min_dimlen;
	
	CUT_DEVICE_INIT(argc, argv);
	CUDA_SAFE_CALL(cudaSetDevice(gpu_bgnid + h_procid%gpu_num)); // each node with four GPUs
	
	//allocate fr0,fe0,fn0,fw0,fs0,fne0,fnw0,fsw0,fse0 and fr1,fe1,fn1,fw1,fs1,fne1,fnw1,fsw1,fse1	
	CUDA_SAFE_CALL(cudaMallocPitch((void **)&f0_d, &pitch, sizeof(float)*totallatticenum, DIRECTION_NUM));
	CUDA_SAFE_CALL(cudaMallocPitch((void **)&f1_d, &pitch, sizeof(float)*totallatticenum, DIRECTION_NUM));
	CUDA_SAFE_CALL(cudaMalloc((void **)&rho_d, sizeof(float)*totallatticenum));
	CUDA_SAFE_CALL(cudaMalloc((void **)&vt, sizeof(float)*totallatticenum));
	CUDA_SAFE_CALL(cudaMalloc((void **)&ux_d,sizeof(float)*totallatticenum));
	CUDA_SAFE_CALL(cudaMalloc((void **)&uy_d,sizeof(float)*totallatticenum));
	CUDA_SAFE_CALL(cudaMalloc((void **)&uz_d,sizeof(float)*totallatticenum));
	CUDA_SAFE_CALL(cudaMalloc((void **)&Fsf,sizeof(float)*totallatticenum*NDIM));
	CUDA_SAFE_CALL(cudaMalloc((void **)&nodeW_d,sizeof(float)*NDIM*NW));
	CUDA_SAFE_CALL(cudaMalloc((void **)&nodeR_d,sizeof(float)*NDIM*NR));
	CUDA_SAFE_CALL(cudaMalloc((void **)&Fw_d,sizeof(float)*NDIM*NW));
	CUDA_SAFE_CALL(cudaMalloc((void **)&Fr_d,sizeof(float)*NDIM*NR));
	CUDA_SAFE_CALL(cudaMalloc((void **)&con_d,sizeof(float)*(nx+4)*(ny+4)*(nz+4)));
	CUDA_SAFE_CALL(cudaMalloc((void**) &geoD, sizeof(int)*totallatticenum));
	CUDA_SAFE_CALL(cudaMalloc((void**) &geoDT, sizeof(int)*(nx+4)*(ny+4)*(nz+4)));

	pitchnum = pitch/sizeof(float);
	MPI_Barrier(MPI_COMM_WORLD);

	if(pitch == sizeof(float)*totallatticenum)
	{
	  disf = (float*) malloc(sizeof(float)*totallatticenum*DIRECTION_NUM); // allocate memory space for f_c1,f_c2
	  diffnum =0;	  
	}else
	{
	  disf = (float*) malloc(pitch*DIRECTION_NUM); // allocate memory space for f_c1,f_c2
	  diffnum = pitchnum -totallatticenum; 	  
	}	
	h_geoD = (unsigned int *) malloc(sizeof(int)*totallatticenum);
	h_geoDT = (unsigned int *) malloc(sizeof(int)*(nx+4)*(ny+4)*(nz+4));
	rho_h = (float *) malloc(sizeof(float)*totallatticenum);
	vt_h = (float *) malloc(sizeof(float)*totallatticenum);
	ux_h = (float *) malloc(sizeof(float)*totallatticenum);
	uy_h = (float *) malloc(sizeof(float)*totallatticenum);
	uz_h = (float *) malloc(sizeof(float)*totallatticenum);
	con_h = (float *) malloc(sizeof(float)*(nx+4)*(ny+4)*(nz+4));

	/* initialise mass and momentum (u=v=0, rho_0) */
	init();	
 	/********* initialize Geometry **********/
	init_geo();   
	
	  write_TEC1_file(0, postproc_intervall); // write data of TECplot format
	MPI_Barrier(MPI_COMM_WORLD);
		 
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(S_d,S_h,DIRECTION_NUM*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(sct,&sct_h,sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(U0,&U0_h,sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(rho0_d,&rho_0,sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(bodyf,&h_bodyf,sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(con_in,&conin_h,sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpy((void *)f0_d,(void *)disf, pitch*DIRECTION_NUM, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy((void *)f1_d,(void *)disf, pitch*DIRECTION_NUM, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(geoD, h_geoD, sizeof(int)*totallatticenum, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(geoDT, h_geoDT, sizeof(int)*(nx+4)*(ny+4)*(nz+4), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(con_d, con_h, sizeof(float)*(nx+4)*(ny+4)*(nz+4), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(rho_d, rho_h, sizeof(float)*totallatticenum, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(vt, vt_h, sizeof(float)*totallatticenum, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ux_d, ux_h, sizeof(float)*totallatticenum, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(uy_d, uy_h, sizeof(float)*totallatticenum, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(uz_d, uz_h, sizeof(float)*totallatticenum, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(nodeW_d, nodeW_h, sizeof(float)*NDIM*NW, cudaMemcpyHostToDevice));	
	CUDA_SAFE_CALL(cudaMemcpy(nodeR_d, nodeR_h, sizeof(float)*NDIM*NR, cudaMemcpyHostToDevice));	

	h_sendbuf = (float *)malloc(sizeof(float)*5* buffersize);
	h_recvbuf = (float *)malloc(sizeof(float)*5* buffersize);

	CUDA_SAFE_CALL(cudaMalloc((void**) &latticebuf, sizeof(float)*5*buffersize)); //device lattice buffer for send and recv
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_GlobalLatticeNum, sizeof(int)*NDIM)); // the global and base lattice index in device memory
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_baseLatticeNum, sizeof(int)*NDIM));
	CUDA_SAFE_CALL(cudaMemcpy(d_GlobalLatticeNum, h_GlobalLatticeNum, sizeof(int)*NDIM, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_baseLatticeNum,baseLatticeNum, sizeof(int)*NDIM, cudaMemcpyHostToDevice));
	
	dim3 threads(THREAD_NUM, 1, 1);
	dim3 grid(ny, nz , 1);
	
	dim3 gridx(parallelny/THREAD_NUM+1,parallelnz,1);	//this can be used to transfer the boundary lattice along X direction
	dim3 gridy(parallelnx/THREAD_NUM+1,parallelnz,1);	//this can be used to transfer the boundary lattice along Y direction
	dim3 gridz(parallelnx/THREAD_NUM+1,parallelny,1);	//this can be used to transfer the boundary lattice along Z direction
	
	dim3 gridnodew(NW/THREAD_NUM+1,1,1);
	dim3 gridnoder(NR/THREAD_NUM+1,1,1);
	//(void)gettime();
	unsigned int timer;
	CUT_SAFE_CALL(cutCreateTimer(&timer));
	CUT_SAFE_CALL(cutStartTimer(timer));
	
	if(h_procid == 0)
	 printf("Waiting......,the main integrate loop will begin \n");
	
	for(step = 1; step <= tend; step++)
	{
	 
	if(step%2==0)
	{
	 //interpolation fluid velocity to boundary nodes and calculate force exerted by nodes;
	 RotorForce <<<gridnoder,threads>>> (step,NR,nodeR_d,Fr_d,ux_d,uy_d,uz_d,parallelnx, parallelny, parallelnz,d_baseLatticeNum,d_GlobalLatticeNum);
	 WallForce <<<gridnodew,threads>>> (NW,nodeW_d,Fw_d,ux_d,uy_d,uz_d,parallelnx, parallelny, parallelnz,d_baseLatticeNum,d_GlobalLatticeNum);
	 //caculate force exerted on the fluids
	 //step 1 initilize Force exerted by fluids
	 Forceinitial <<< grid,threads >>> (parallelnx, parallelny, parallelnz,Fsf);
	 //step 2 caculate Forceexerted by fluids
	 RotorForceonFluid <<< gridnoder,threads>>> (step,NR,parallelnx, parallelny, parallelnz,nodeR_d,Fr_d, d_baseLatticeNum,d_GlobalLatticeNum,Fsf);
	 WallForceonFluid <<< gridnodew,threads>>> (NW,parallelnx, parallelny, parallelnz,nodeW_d,Fw_d,d_baseLatticeNum,d_GlobalLatticeNum, Fsf);
	 //update fluid velocity
	 velocity_update <<< grid,threads >>> (parallelnx, parallelny, parallelnz,ux_d,uy_d,uz_d,Fsf);
	 //collision and propagation
	 
	 LBCollProp<<< grid, threads >>> (parallelnx, parallelny, parallelnz, xstartoff, ystartoff, zstartoff, geoD, pitch, f0_d, f1_d,rho_d,con_d,ux_d,uy_d,uz_d,Fsf,vt,d_baseLatticeNum, d_GlobalLatticeNum);
	 CUDA_SAFE_CALL(cudaThreadSynchronize());
	// printf("Proc %d  lbcollprop computation is ok, step is %d  \n",h_procid, step);
	
	 /*************Transfer the boundary lattice to adjacent process, including data into and out from device memory***********************/
		 
	 if(h_procn[0] > 1)// the number of processors in X direction great 1, need communication
	 {	 
	  for(i=0; i<2; i++)
	  {
	     LBBorders_x<<< gridx,threads >>> (parallelnx, parallelny, parallelnz, latticebuf, dir[i], f1_d, pitch);
	 
	     CUDA_SAFE_CALL(cudaMemcpy(h_sendbuf, latticebuf, sizeof(float)*parallelny*parallelnz*5, cudaMemcpyDeviceToHost));
	 
	     time0 = MPI_Wtime();
	     MPI_Sendrecv(h_sendbuf, parallelny*parallelnz*5, MPI_FLOAT, h_nbprocid[0][i], 0, h_recvbuf, parallelny*parallelnz*5, MPI_FLOAT, h_nbprocid[0][1-i], 0, MPI_COMM_WORLD, &status);
	 
	     //MPI_Barrier(MPI_COMM_WORLD);
	     time1 = MPI_Wtime();
	    
	     commtime += time1 -time0;
	     
	     if(h_nbprocid[0][1-i] != MPI_PROC_NULL)
	     {
	      CUDA_SAFE_CALL(cudaMemcpy(latticebuf, h_recvbuf, sizeof(float)*parallelny*parallelnz*5, cudaMemcpyHostToDevice));
	 
	      LBPostborders_x<<< gridx, threads >>> (parallelnx, parallelny, parallelnz, xstartoff, ystartoff, zstartoff, latticebuf, dir[i], f1_d, pitch, d_baseLatticeNum, d_GlobalLatticeNum );
	     }
	   }
	}
	
	if(h_procn[1] > 1 ) //the number of processors in Y direction great 1, need communication
	{	 
	 for(i=0; i<2; i++) //transfer along the positive and negative directions
	 {	     
	     LBBorders_y<<< gridy,threads >>> (parallelnx, parallelny,parallelnz, latticebuf, dir[i], f1_d, pitch);
	    	     
	     CUDA_SAFE_CALL(cudaMemcpy(h_sendbuf, latticebuf, sizeof(float)*parallelnx*parallelnz*5, cudaMemcpyDeviceToHost));
	     
	     time0 = MPI_Wtime();
	 
	     MPI_Sendrecv(h_sendbuf, parallelnx*parallelnz*5, MPI_FLOAT, h_nbprocid[1][i], 0, h_recvbuf, parallelnx*parallelnz*5, MPI_FLOAT, h_nbprocid[1][1-i], 0, MPI_COMM_WORLD, &status);
	 
	     //MPI_Barrier(MPI_COMM_WORLD);
	     time1 = MPI_Wtime();
	     commtime += time1 - time0;
	 
	     if(h_nbprocid[1][1-i] != MPI_PROC_NULL)
	     {
	       CUDA_SAFE_CALL(cudaMemcpy(latticebuf, h_recvbuf, sizeof(float)*parallelnx*parallelnz*5, cudaMemcpyHostToDevice));
	 
	       LBPostborders_y <<<gridy, threads>>>(parallelnx, parallelny, parallelnz, xstartoff, ystartoff, zstartoff, latticebuf, dir[i], f1_d, pitch,d_baseLatticeNum, d_GlobalLatticeNum );	       
	     }	
	 } 	 
	  	  
	}

	if(h_procn[2] > 1 ) //the number of processors in Y direction great 1, need communication
	{	 
	 for(i=0; i<2; i++) //transfer along the positive and negative directions
	 {	     
	     LBBorders_z<<< gridz,threads >>> (parallelnx, parallelny, parallelnz, latticebuf, dir[i], f1_d, pitch);
	    	     
	     CUDA_SAFE_CALL(cudaMemcpy(h_sendbuf, latticebuf, sizeof(float)*parallelnx*parallelny*5, cudaMemcpyDeviceToHost));
	     
	     time0 = MPI_Wtime();
	     MPI_Sendrecv(h_sendbuf, parallelnx*parallelny*5, MPI_FLOAT, h_nbprocid[2][i], 0, h_recvbuf, parallelnx*parallelny*5, MPI_FLOAT, h_nbprocid[2][1-i], 0, MPI_COMM_WORLD, &status);
	 
	     //MPI_Barrier(MPI_COMM_WORLD);
	     time1 = MPI_Wtime();
	     commtime += time1 - time0;
	 
	     if(h_nbprocid[2][1-i] != MPI_PROC_NULL)
	     {
	       CUDA_SAFE_CALL(cudaMemcpy(latticebuf, h_recvbuf, sizeof(float)*parallelnx*parallelny*5, cudaMemcpyHostToDevice));
	 
	       LBPostborders_z <<<gridz, threads>>>(parallelnx, parallelny, parallelnz, xstartoff, ystartoff, zstartoff, latticebuf, dir[i], f1_d, pitch, d_baseLatticeNum, d_GlobalLatticeNum );	       
	     }	
	 } 	 
	  	  
	}
	MacroCal<<< grid, threads >>>(parallelnx, parallelny, parallelnz, xstartoff, ystartoff, zstartoff,geoD,pitch, f1_d,rho_d,ux_d,uy_d, uz_d);
	disfBorders_inlet<<< gridx, threads >>>(parallelnx, parallelny, parallelnz,f1_d,rho_d,ux_d,uy_d,uz_d,pitch,geoD,d_baseLatticeNum,d_GlobalLatticeNum);
	disfBorders_outlet<<< gridx, threads >>>(parallelnx, parallelny, parallelnz,f1_d,rho_d,ux_d,uy_d,uz_d,pitch,geoD,d_baseLatticeNum,d_GlobalLatticeNum);
	
	//boundary information transfer
	
	if(h_procn[0] > 1)// the number of processors in X direction great 1, need communication
	 {	 
	  for(i=0; i<2; i++)
	  {
	     TransBorders_x<<< gridx,threads >>> (parallelnx, parallelny, parallelnz, latticebuf, dir[i],con_d,ux_d,uy_d,uz_d, pitch);
	     CUDA_SAFE_CALL(cudaMemcpy(h_sendbuf, latticebuf, sizeof(float)*parallelny*parallelnz*5, cudaMemcpyDeviceToHost));
	     time0 = MPI_Wtime();
	     MPI_Sendrecv(h_sendbuf, parallelny*parallelnz*5, MPI_FLOAT, h_nbprocid[0][i], 0, h_recvbuf, parallelny*parallelnz*5, MPI_FLOAT, h_nbprocid[0][1-i], 0, MPI_COMM_WORLD, &status);
	 
	     //MPI_Barrier(MPI_COMM_WORLD);
	     time1 = MPI_Wtime();
	     commtime += time1 -time0;
	     
	     if(h_nbprocid[0][1-i] != MPI_PROC_NULL)
	     {
	      CUDA_SAFE_CALL(cudaMemcpy(latticebuf, h_recvbuf, sizeof(float)*parallelny*parallelnz*5, cudaMemcpyHostToDevice));
	      TransPostborders_x<<< gridx, threads >>> (parallelnx,parallelny,parallelnz,xstartoff,ystartoff,zstartoff, latticebuf, dir[i],con_d,ux_d,uy_d,uz_d, pitch,d_baseLatticeNum, d_GlobalLatticeNum);
	     }
	   }
	}
	
	if(h_procn[1] > 1 ) //the number of processors in Y direction great 1, need communication
	{	 
	 for(i=0; i<2; i++) //transfer along the positive and negative directions
	 {	 
	     TransBorders_y<<< gridy,threads >>>(parallelnx, parallelny,parallelnz, latticebuf, dir[i], con_d,ux_d,uy_d,uz_d,pitch);    
	     CUDA_SAFE_CALL(cudaMemcpy(h_sendbuf, latticebuf, sizeof(float)*parallelnx*parallelnz*5, cudaMemcpyDeviceToHost));
	     
	     time0 = MPI_Wtime();
	 
	     MPI_Sendrecv(h_sendbuf, parallelnx*parallelnz*5, MPI_FLOAT, h_nbprocid[1][i], 0, h_recvbuf, parallelnx*parallelnz*5, MPI_FLOAT, h_nbprocid[1][1-i], 0, MPI_COMM_WORLD, &status);
	     //MPI_Barrier(MPI_COMM_WORLD);
	     time1 = MPI_Wtime();
	     commtime += time1 - time0;
	 
	     if(h_nbprocid[1][1-i] != MPI_PROC_NULL)
	     {
	       CUDA_SAFE_CALL(cudaMemcpy(latticebuf, h_recvbuf, sizeof(float)*parallelnx*parallelnz*5, cudaMemcpyHostToDevice));
	       TransPostborders_y <<<gridy, threads>>>(parallelnx, parallelny, parallelnz, xstartoff, ystartoff, zstartoff, latticebuf, dir[i],con_d,ux_d,uy_d,uz_d, pitch, d_baseLatticeNum, d_GlobalLatticeNum);	       
	     }	
	 } 	 
	  	  
	}

	if(h_procn[2] > 1 ) //the number of processors in Y direction great 1, need communication
	{	 
	 for(i=0; i<2; i++) //transfer along the positive and negative directions
	 {
	     TransBorders_z <<< gridz,threads >>>(parallelnx, parallelny, parallelnz, latticebuf, dir[i], con_d,ux_d,uy_d,uz_d, pitch);
	 		 	     
	     CUDA_SAFE_CALL(cudaMemcpy(h_sendbuf, latticebuf, sizeof(float)*parallelnx*parallelny*5, cudaMemcpyDeviceToHost));
	     
	     time0 = MPI_Wtime();
	     MPI_Sendrecv(h_sendbuf, parallelnx*parallelny*5, MPI_FLOAT, h_nbprocid[2][i], 0, h_recvbuf, parallelnx*parallelny*5, MPI_FLOAT, h_nbprocid[2][1-i], 0, MPI_COMM_WORLD, &status);
	 
	     //MPI_Barrier(MPI_COMM_WORLD);
	     time1 = MPI_Wtime();
	     commtime += time1 - time0;
	 
	     if(h_nbprocid[2][1-i] != MPI_PROC_NULL)
	     {
	       CUDA_SAFE_CALL(cudaMemcpy(latticebuf, h_recvbuf, sizeof(float)*parallelnx*parallelny*5, cudaMemcpyHostToDevice));
	       TransPostborders_z <<<gridz, threads>>>(parallelnx, parallelny, parallelnz, xstartoff, ystartoff, zstartoff, latticebuf, dir[i],con_d, ux_d,uy_d,uz_d,pitch, d_baseLatticeNum, d_GlobalLatticeNum);     
	     }	
	 } 	 
	  	  
	}
	
	TransCal<<< grid, threads >>>(parallelnx, parallelny, parallelnz, xstartoff, ystartoff, zstartoff, geoD,geoDT,pitch,con_d,ux_d,uy_d, uz_d,vt,d_baseLatticeNum,d_GlobalLatticeNum);
		
	if (step % postproc_intervall == 0)
	{
	  CUDA_SAFE_CALL(cudaMemcpy(con_h, con_d, sizeof(float)*(nx+4)*(ny+4)*(nz+4), cudaMemcpyDeviceToHost));
	  CUDA_SAFE_CALL(cudaMemcpy(rho_h, rho_d, sizeof(float)*totallatticenum, cudaMemcpyDeviceToHost));
	  CUDA_SAFE_CALL(cudaMemcpy(ux_h, ux_d, sizeof(float)*totallatticenum, cudaMemcpyDeviceToHost));
	  CUDA_SAFE_CALL(cudaMemcpy(uy_h, uy_d, sizeof(float)*totallatticenum, cudaMemcpyDeviceToHost));
	  CUDA_SAFE_CALL(cudaMemcpy(uz_h, uz_d, sizeof(float)*totallatticenum, cudaMemcpyDeviceToHost));
	
	  CUDA_SAFE_CALL(cudaThreadSynchronize());
	  write_TEC1_file(step, postproc_intervall); // write data of TECplot format
	}
	}
	else
	{
	 //interpolation fluid velocity to boundary nodes and calculate force exerted by nodes;
	 RotorForce <<<gridnoder,threads>>> (step,NR,nodeR_d,Fr_d,ux_d,uy_d,uz_d,parallelnx, parallelny, parallelnz,d_baseLatticeNum,d_GlobalLatticeNum);
	 WallForce <<<gridnodew,threads>>> (NW,nodeW_d,Fw_d,ux_d,uy_d,uz_d,parallelnx, parallelny, parallelnz,d_baseLatticeNum,d_GlobalLatticeNum);
	 //caculate force exerted on the fluids
	 //step 1 initilize Force exerted by fluids
	 Forceinitial <<< grid,threads >>> (parallelnx, parallelny, parallelnz,Fsf);
	 //step 2 caculate Forceexerted by fluids
	 RotorForceonFluid <<< gridnoder,threads>>> (step,NR,parallelnx, parallelny, parallelnz,nodeR_d,Fr_d, d_baseLatticeNum,d_GlobalLatticeNum,Fsf);
	 WallForceonFluid <<< gridnodew,threads>>> (NW,parallelnx, parallelny, parallelnz,nodeW_d,Fw_d,d_baseLatticeNum,d_GlobalLatticeNum, Fsf);
	 //update fluid velocity
	 velocity_update <<< grid,threads >>> (parallelnx, parallelny, parallelnz,ux_d,uy_d,uz_d,Fsf);

	  //collision and propagation
	 LBCollProp<<< grid, threads >>> (parallelnx, parallelny, parallelnz, xstartoff, ystartoff, zstartoff, geoD, pitch, f1_d, f0_d,rho_d,con_d,ux_d,uy_d,uz_d,Fsf,vt,d_baseLatticeNum, d_GlobalLatticeNum);
	 CUDA_SAFE_CALL(cudaThreadSynchronize());

	 /*************Transfer the boundary lattice to adjacent process, including data into and out from device memory***********************/
	 if(h_procn[0] > 1)// the number of processors in X direction great 1, need communication
	 {	 
	  for(i=0; i<2; i++)
	  {
	     LBBorders_x<<< gridx,threads >>> (parallelnx, parallelny, parallelnz, latticebuf, dir[i], f0_d, pitch);
	 
	     CUDA_SAFE_CALL(cudaMemcpy(h_sendbuf, latticebuf, sizeof(float)*parallelny*parallelnz*5, cudaMemcpyDeviceToHost));
	 
	     time0 = MPI_Wtime();
	     MPI_Sendrecv(h_sendbuf, parallelny*parallelnz*5, MPI_FLOAT, h_nbprocid[0][i], 0, h_recvbuf, parallelny*parallelnz*5, MPI_FLOAT, h_nbprocid[0][1-i], 0, MPI_COMM_WORLD, &status);
	 
	     //MPI_Barrier(MPI_COMM_WORLD);
	     time1 = MPI_Wtime();
	    
	     commtime += time1 -time0;
	     
	     if(h_nbprocid[0][1-i] != MPI_PROC_NULL)
	     {
	      CUDA_SAFE_CALL(cudaMemcpy(latticebuf, h_recvbuf, sizeof(float)*parallelny*parallelnz*5, cudaMemcpyHostToDevice));
	 
	      LBPostborders_x<<< gridx, threads >>> (parallelnx, parallelny, parallelnz, xstartoff, ystartoff, zstartoff, latticebuf, dir[i], f0_d, pitch, d_baseLatticeNum, d_GlobalLatticeNum );
	     }
	   }
	}
	
	 //printf("Proc %d  postx is ok, step is %d  \n",h_procid, step);
	if(h_procn[1] > 1 ) //the number of processors in Y direction great 1, need communication
	{	 
	 for(i=0; i<2; i++) //transfer along the positive and negative directions
	 {	     
	     LBBorders_y<<< gridy,threads >>> (parallelnx, parallelny,parallelnz, latticebuf, dir[i], f0_d, pitch);
	    	     
	     CUDA_SAFE_CALL(cudaMemcpy(h_sendbuf, latticebuf, sizeof(float)*parallelnx*parallelnz*5, cudaMemcpyDeviceToHost));
	     //CUDA_SAFE_CALL(cudaThreadSynchronize());
	     
	     time0 = MPI_Wtime();
	 
	     MPI_Sendrecv(h_sendbuf, parallelnx*parallelnz*5, MPI_FLOAT, h_nbprocid[1][i], 0, h_recvbuf, parallelnx*parallelnz*5, MPI_FLOAT, h_nbprocid[1][1-i], 0, MPI_COMM_WORLD, &status);
	 
	     //MPI_Barrier(MPI_COMM_WORLD);
	     time1 = MPI_Wtime();
	     commtime += time1 - time0;
	 
	     if(h_nbprocid[1][1-i] != MPI_PROC_NULL)
	     {
	       CUDA_SAFE_CALL(cudaMemcpy(latticebuf, h_recvbuf, sizeof(float)*parallelnx*parallelnz*5, cudaMemcpyHostToDevice));
	 
	       LBPostborders_y <<<gridy, threads>>>(parallelnx, parallelny, parallelnz, xstartoff, ystartoff, zstartoff, latticebuf, dir[i], f0_d, pitch, d_baseLatticeNum, d_GlobalLatticeNum );	       
	     }	
	 } 	 
	  	  
	}
	 //printf("Proc %d  posty is ok, step is %d  \n",h_procid, step);
	
	if(h_procn[2] > 1 ) //the number of processors in Y direction great 1, need communication
	{	 
	 for(i=0; i<2; i++) //transfer along the positive and negative directions
	 {	     
	     LBBorders_z<<< gridz,threads >>> (parallelnx, parallelny, parallelnz, latticebuf, dir[i], f0_d, pitch);  
	     CUDA_SAFE_CALL(cudaMemcpy(h_sendbuf, latticebuf, sizeof(float)*parallelnx*parallelny*5, cudaMemcpyDeviceToHost));
	     
	     time0 = MPI_Wtime();
	 
	     MPI_Sendrecv(h_sendbuf, parallelnx*parallelny*5, MPI_FLOAT, h_nbprocid[2][i], 0, h_recvbuf, parallelnx*parallelny*5, MPI_FLOAT, h_nbprocid[2][1-i], 0, MPI_COMM_WORLD, &status);
	 
	     //MPI_Barrier(MPI_COMM_WORLD);
	     time1 = MPI_Wtime();
	     commtime += time1 - time0;
	 
	     if(h_nbprocid[2][1-i] != MPI_PROC_NULL)
	     {
	       CUDA_SAFE_CALL(cudaMemcpy(latticebuf, h_recvbuf, sizeof(float)*parallelnx*parallelny*5, cudaMemcpyHostToDevice));
	 
	       LBPostborders_z <<<gridz, threads>>>(parallelnx, parallelny, parallelnz, xstartoff, ystartoff, zstartoff, latticebuf, dir[i], f0_d, pitch,d_baseLatticeNum, d_GlobalLatticeNum );	       
	     }	
	 } 	 
	  	  
	}
	
	MacroCal<<< grid, threads >>>(parallelnx, parallelny, parallelnz, xstartoff, ystartoff, zstartoff,geoD,pitch, f0_d,rho_d,ux_d,uy_d, uz_d);
	
	disfBorders_inlet<<< gridx, threads >>>(parallelnx, parallelny, parallelnz,f0_d,rho_d,ux_d,uy_d,uz_d,pitch,geoD,d_baseLatticeNum,d_GlobalLatticeNum);
	disfBorders_outlet<<< gridx, threads >>>(parallelnx, parallelny, parallelnz,f0_d,rho_d,ux_d,uy_d,uz_d,pitch,geoD,d_baseLatticeNum,d_GlobalLatticeNum);
	
	//boundary information transfer
	
	if(h_procn[0] > 1)// the number of processors in X direction great 1, need communication
	 {	 
	  for(i=0; i<2; i++)
	  {
	     TransBorders_x<<< gridx,threads >>> (parallelnx, parallelny, parallelnz, latticebuf, dir[i],con_d,ux_d,uy_d,uz_d, pitch);
	     CUDA_SAFE_CALL(cudaMemcpy(h_sendbuf, latticebuf, sizeof(float)*parallelny*parallelnz*5, cudaMemcpyDeviceToHost));
	     time0 = MPI_Wtime();
	     MPI_Sendrecv(h_sendbuf, parallelny*parallelnz*5, MPI_FLOAT, h_nbprocid[0][i], 0, h_recvbuf, parallelny*parallelnz*5, MPI_FLOAT, h_nbprocid[0][1-i], 0, MPI_COMM_WORLD, &status);
	 
	     //MPI_Barrier(MPI_COMM_WORLD);
	     time1 = MPI_Wtime();
	    
	     commtime += time1 -time0;
	     
	     if(h_nbprocid[0][1-i] != MPI_PROC_NULL)
	     {
	      CUDA_SAFE_CALL(cudaMemcpy(latticebuf, h_recvbuf, sizeof(float)*parallelny*parallelnz*5, cudaMemcpyHostToDevice));
	      TransPostborders_x<<< gridx, threads >>> (parallelnx,parallelny,parallelnz,xstartoff,ystartoff,zstartoff, latticebuf, dir[i],con_d,ux_d,uy_d,uz_d, pitch,d_baseLatticeNum, d_GlobalLatticeNum);
	     }
	   }
	}
	
		if(h_procn[1] > 1 ) //the number of processors in Y direction great 1, need communication
	{	 
	 for(i=0; i<2; i++) //transfer along the positive and negative directions
	 {	 
	     TransBorders_y<<< gridy,threads >>>(parallelnx, parallelny,parallelnz, latticebuf, dir[i], con_d,ux_d,uy_d,uz_d,pitch);    
	     CUDA_SAFE_CALL(cudaMemcpy(h_sendbuf, latticebuf, sizeof(float)*parallelnx*parallelnz*5, cudaMemcpyDeviceToHost));
	     
	     time0 = MPI_Wtime();
	 
	     MPI_Sendrecv(h_sendbuf, parallelnx*parallelnz*5, MPI_FLOAT, h_nbprocid[1][i], 0, h_recvbuf, parallelnx*parallelnz*5, MPI_FLOAT, h_nbprocid[1][1-i], 0, MPI_COMM_WORLD, &status);
	     //MPI_Barrier(MPI_COMM_WORLD);
	     time1 = MPI_Wtime();
	     commtime += time1 - time0;
	 
	     if(h_nbprocid[1][1-i] != MPI_PROC_NULL)
	     {
	       CUDA_SAFE_CALL(cudaMemcpy(latticebuf, h_recvbuf, sizeof(float)*parallelnx*parallelnz*5, cudaMemcpyHostToDevice));
	       TransPostborders_y <<<gridy, threads>>>(parallelnx, parallelny, parallelnz, xstartoff, ystartoff, zstartoff, latticebuf, dir[i],con_d,ux_d,uy_d,uz_d, pitch, d_baseLatticeNum, d_GlobalLatticeNum);	       
	     }	
	 } 	 
	  	  
	}

	if(h_procn[2] > 1 ) //the number of processors in Y direction great 1, need communication
	{	 
	 for(i=0; i<2; i++) //transfer along the positive and negative directions
	 {
	     TransBorders_z <<< gridz,threads >>>(parallelnx, parallelny, parallelnz, latticebuf, dir[i], con_d,ux_d,uy_d,uz_d, pitch);
	 		 	     
	     CUDA_SAFE_CALL(cudaMemcpy(h_sendbuf, latticebuf, sizeof(float)*parallelnx*parallelny*5, cudaMemcpyDeviceToHost));
	     
	     time0 = MPI_Wtime();
	     MPI_Sendrecv(h_sendbuf, parallelnx*parallelny*5, MPI_FLOAT, h_nbprocid[2][i], 0, h_recvbuf, parallelnx*parallelny*5, MPI_FLOAT, h_nbprocid[2][1-i], 0, MPI_COMM_WORLD, &status);
	 
	     //MPI_Barrier(MPI_COMM_WORLD);
	     time1 = MPI_Wtime();
	     commtime += time1 - time0;
	 
	     if(h_nbprocid[2][1-i] != MPI_PROC_NULL)
	     {
	       CUDA_SAFE_CALL(cudaMemcpy(latticebuf, h_recvbuf, sizeof(float)*parallelnx*parallelny*5, cudaMemcpyHostToDevice));
	       TransPostborders_z <<<gridz, threads>>>(parallelnx, parallelny, parallelnz, xstartoff, ystartoff, zstartoff, latticebuf, dir[i],con_d, ux_d,uy_d,uz_d,pitch, d_baseLatticeNum, d_GlobalLatticeNum);     
	     }	
	 } 	 
	  	  
	}
	
	TransCal<<< grid, threads >>>(parallelnx, parallelny, parallelnz, xstartoff, ystartoff, zstartoff, geoD,geoDT,pitch,con_d,ux_d,uy_d, uz_d,vt,d_baseLatticeNum,d_GlobalLatticeNum);
	
	if (step % postproc_intervall == 0)
	{
	   CUDA_SAFE_CALL(cudaMemcpy(con_h, con_d, sizeof(float)*(nx+4)*(ny+4)*(nz+4), cudaMemcpyDeviceToHost));
	   CUDA_SAFE_CALL(cudaMemcpy(rho_h, rho_d, sizeof(float)*totallatticenum, cudaMemcpyDeviceToHost));
	   CUDA_SAFE_CALL(cudaMemcpy(ux_h, ux_d, sizeof(float)*totallatticenum, cudaMemcpyDeviceToHost));
	   CUDA_SAFE_CALL(cudaMemcpy(uy_h, uy_d, sizeof(float)*totallatticenum, cudaMemcpyDeviceToHost));
	   CUDA_SAFE_CALL(cudaMemcpy(uz_h, uz_d, sizeof(float)*totallatticenum, cudaMemcpyDeviceToHost));
	  
	   CUDA_SAFE_CALL(cudaThreadSynchronize());
	   write_TEC1_file(step, postproc_intervall); // write data of TECplot format*/
	}
	  
    } //end if step %2 ==0 

	 //Postprocess results
	 if(step %100 ==0 && h_procid ==0)
	 {
	 		printf("Now step is %d \n", step); 
	 } 	
	
	}// end of main loop integrate
	
	
	CUT_SAFE_CALL(cutStopTimer(timer));
	if(h_procid == 0) 
	printf("Processing time measured by cuda is %f ms \n",cutGetTimerValue(timer));
	
	CUT_SAFE_CALL(cutDeleteTimer(timer));
		
	free(disf);	
	free(h_geoD);
	free(h_geoDT);
	free(rho_h);
	free(con_h);
	free(ux_h);
	free(uy_h);
	free(uz_h);
	free(M_h);
	free(MI_h);
	free(h_sendbuf);
	free(h_recvbuf);
	free(nodeR_h);
	free(nodeW_h);
	
	//printf("free host mem ok \n");
	
	CUDA_SAFE_CALL(cudaFree(f0_d));
	CUDA_SAFE_CALL(cudaFree(f1_d));
	CUDA_SAFE_CALL(cudaFree(geoD));	
	CUDA_SAFE_CALL(cudaFree(rho_d));	
	CUDA_SAFE_CALL(cudaFree(con_d));	
	CUDA_SAFE_CALL(cudaFree(ux_d));	
	CUDA_SAFE_CALL(cudaFree(uy_d));	
	CUDA_SAFE_CALL(cudaFree(uz_d));	
	CUDA_SAFE_CALL(cudaFree(nodeR_d));
	CUDA_SAFE_CALL(cudaFree(nodeW_d));
	CUDA_SAFE_CALL(cudaFree(Fw_d));
	CUDA_SAFE_CALL(cudaFree(Fr_d));
	CUDA_SAFE_CALL(cudaFree(Fsf));
	CUDA_SAFE_CALL(cudaFree(latticebuf));	
	
	//printf("free device mem ok \n");
	
	return 0;
}
