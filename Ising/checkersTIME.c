/* gcc -std=c99 -Wall -O3 -fopenmp isingparalelofinal.c -lm */
// Para compilar "gcc ..." -> "export OMP_NUM_THREADS=*" -> "./a.out"

/***************************************************************
 *                            INCLUDES                      
 **************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>
/****************************************************************
 *                            DEFINITIONS                      
 ***************************************************************/
#define L           128
#define J           1.
#define L2          (L*L)
#define K           1.
#define T           2.2
#define MCS_EQ      0
#define MEASURES    500000
#define SAMPLES     1
#define PRODUCTION  1
#define real double

/***************************************************************
 *                            FUNCTIONS                       
 **************************************************************/
void sweep(void); //Evolui (PARALELO)
void inicializacao(void); //Inicia (PARALELO)
void openfiles(void); //Abre arquivos de saida
void vizualizacao(void); //Habilita vizualizacao
void chess(void); //Divide estrutura
void energia(void); //Calcula energia (PARALELO)
void seed_rng_states(unsigned* rngs, const unsigned seed); //Gera estados aleatórios
void evolui(int,int,double);
/***************************************************************
 *                         GLOBAL VARIABLES                   
 **************************************************************/
FILE *fp1,*fp2;
int nthreads,M, Et, s[L2],viz[L2][4],c[2][L2/2],hist[2*L2],hist2[2*L2];
double w[9];
unsigned* rngs;

//Gera uniformemente entre [0,1)
inline real rand_uniform(unsigned* rng) {
  *rng = 1664525 * (*rng) + 1013904223;
  return (2.32830643653869629E-10 * (*rng));
}

/***************************************************************
 *                          MAIN PROGRAM  
 **************************************************************/
int main(void){
  int i,j;

  rngs = (unsigned*)malloc(sizeof(unsigned) * omp_get_max_threads());
  seed_rng_states(rngs, 123456789);

  inicializacao();
  openfiles();
  for(i = 0; i < MCS_EQ; i++) {
    sweep();
  }

  for(j=0; j<SAMPLES; j++) {
    double start = omp_get_wtime();
    for(i = 0; i < MEASURES; i++) {
      sweep();
    }
    double end = omp_get_wtime();
    fprintf(fp1,"Elapsed time for %d run: %.4fs\n", nthreads, (end-start));
  }
}
/***************************************************************
 *                        INICIALIZAÇÃO  
 **************************************************************/
void inicializacao(void) {
  for (int i = 0; i < L2; i++) {
    double r = (double)rand()/RAND_MAX;
    if (r < 0.5)s[i] = (int)-1;
    else s[i] = (int)1;
  }
  for (int i = 0; i < L2; i++) {
  //Direita
    if (i + 1 > L2 - 1)
      viz[i][0] = (i + 1) - L2;
    else
      viz[i][0] = i + 1;
    //Esquerda
    if (i - 1 < 0)
      viz[i][1] = L2 + (i - 1);
    else
      viz[i][1] = i - 1;
    //Cima
    if (i - L < 0)
      viz[i][2] = L2 + (i - L);
    else
      viz[i][2] = i - L;
    //Baixo
    if (i + L > L2 - 1)
      viz[i][3] = (i + L) - L2;
    else
      viz[i][3] = i + L;
  }
  for (int i = -4; i <= 4; i++) {
    w [4 + i] = exp(-2*i/(K*T));
  }
  chess();
  energia();
}
/****************************************************************
 *               MCS routine (PARALELIZADO)                                    
 ***************************************************************/
void sweep(void) {
  for (int chess = 0; chess < 2; chess++) {
    #pragma omp parallel
    {
      unsigned rng_copy = rngs[omp_get_thread_num()];
      #pragma omp for schedule (static)
      for (int i=0; i<L2/2; i++) {
        double r = rand_uniform(&rng_copy);
        evolui(chess,i,r);
      }
      rngs[omp_get_thread_num()] = rng_copy;
    }
  }
}
/**************************************************************
 *               Open output files routine                   
 *************************************************************/
void openfiles(void) {
  char output_file1[100];

  sprintf(output_file1,"time-checker-lg-%d-%dthreads.dsf",L,nthreads);
  fp1 = fopen(output_file1,"w");
  fflush(fp1);

  return;
}
/**************************************************************
 *                       Vizualização                   
 *************************************************************/
void vizualizacao(void) {
  int l;
  printf("pl '-' matrix w image\n");
  for(l = L2-1; l >= 0; l--) {
    printf("%d ", s[l]);
    if( l%L == 0 ) printf("\n");
  }
  printf("e\ne\n");
}

/**************************************************************
 *               Chess Routine (NÃO PARALELIZA EFICIENTEMENTE)          
 *************************************************************/

void chess(void) {
  int i,k,l,m,b;
  for(i=0,l=0,m=0;i<L*L;i++) {
    k=i/L;
    b=k%2;
    if(b==0) {
      if(i%2==0) {
	c[0][l]=i;
	l++;
      }
      else {
	c[1][m]=i;
	m++;	
      }
    }
    else {
      if(i%2==0) {
	c[1][m]=i;
	m++;
      }
      else {
	c[0][l]=i;
	l++;
      }
    }
  }
}
/**************************************************************
 *               Energia  (PARALELIZADO)                
 *************************************************************/
void energia(void) {
  int i,j;
  M = 0;
  Et = 0;
for (i = 0; i < L2; i++) {
  for (j = 0; j < 4; j++)
    Et += s[i]*s[viz[i][j]];
    M += s[i];
  }
  Et *= -J/2.0;
}

/**************************************************************
 *               Seed Random States (PARALELIZADO)                 
 *************************************************************/
void seed_rng_states(unsigned* rngs, const unsigned seed) {
  #pragma omp parallel
  {
    nthreads = omp_get_num_threads();
    const int i = omp_get_thread_num();
    unsigned state = seed + 11 * i;
    rngs[i] = (unsigned)(1e6 * rand_uniform(&state));
  }
  return;
}

/**************************************************************
 *                        Evolui sitio        
 *************************************************************/
void evolui(int _chess, int _i, double _r) {
  int j = c[_chess][_i];
  int dE = J*s[j]*(s[viz[j][0]]+s[viz[j][1]]+s[viz[j][2]]+s[viz[j][3]]);
  if (dE <= 0) {
    s[j] = - s[j];
  }
  else {
    if(_r < w[4+dE]) {
      s[j]= - s[j];
    }
  }
}
