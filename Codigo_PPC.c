/*
 * mandelbrot_openmp_baseline.c
 * Proyecto Final - Programacion Paralela y Concurrente
 *
 * Compilar:
 *   gcc -O2 -fopenmp -o mandelbrot_omp_base mandelbrot_openmp_baseline.c -lm
 *
 * Uso:
 *   OMP_NUM_THREADS=8 ./mandelbrot_omp_base
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>

/* omp_get_wtime() mide tiempo de pared (wall-clock), ideal para paralelo */

/* ──────────────────────────── Configuracion ──────────────────────────── */
#define WIDTH       7680
#define HEIGHT      4320
#define MAX_ITER    1000
#define CHANNELS    3

#define RE_MIN  -2.5
#define RE_MAX   1.0
#define IM_MIN  -1.25
#define IM_MAX   1.25

#define GAUSS_RADIUS 15
#define GAUSS_SIZE   (2*GAUSS_RADIUS+1)

#define HIST_BINS 256

typedef unsigned char uchar;

typedef struct {
    uchar *data;
    int    width;
    int    height;
} Image;

/* ──────────────────────────── Utilidades ────────────────────────────── */

static Image img_alloc(int w, int h)
{
    Image img;
    img.width  = w;
    img.height = h;
    img.data   = (uchar *)calloc((size_t)w * h * CHANNELS, 1);
    if (!img.data) { fprintf(stderr, "OOM\n"); exit(1); }
    return img;
}
static void img_free(Image *img) { free(img->data); img->data = NULL; }

static void img_save_ppm(const Image *img, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return; }
    fprintf(f, "P6\n%d %d\n255\n", img->width, img->height);
    fwrite(img->data, 1, (size_t)img->width * img->height * CHANNELS, f);
    fclose(f);
    printf("  Guardada: %s\n", path);
}

/* ──────────────────────────── Paleta de color ──────────────────────────────── */
static void iter_to_rgb(int iter, double zr, double zi,
                        uchar *r, uchar *g, uchar *b)
{
    if (iter == MAX_ITER) { *r = *g = *b = 0; return; }

    double log_zn = log(zr*zr + zi*zi) * 0.5;
    double nu     = log(log_zn / log(2.0)) / log(2.0);
    double t      = ((double)iter + 1.0 - nu) / (double)MAX_ITER;

    *r = (uchar)((0.5 + 0.5*cos(6.28318*(1.0*t + 0.0))) * 255);
    *g = (uchar)((0.5 + 0.5*cos(6.28318*(1.0*t + 0.33))) * 255);
    *b = (uchar)((0.5 + 0.5*cos(6.28318*(1.0*t + 0.67))) * 255);
}

/* ─────────── Tarea A: Mandelbrot paralelizado (baseline IA) ─────────── */

static void generate_mandelbrot(Image *img)
{
    printf("[Tarea A] Generando Mandelbrot %dx%d con OpenMP ...\n",
           img->width, img->height);

    double re_scale = (RE_MAX - RE_MIN) / (img->width  - 1);
    double im_scale = (IM_MAX - IM_MIN) / (img->height - 1);

    // mejoramos el balanceo de carga cambiandolo de static a dynamic, 8
    #pragma omp parallel for schedule(dynamic, 8) default(none) \
            shared(img, re_scale, im_scale)
    for (int py = 0; py < img->height; py++) {
        double ci = IM_MAX - py * im_scale;

        for (int px = 0; px < img->width; px++) {
            double cr = RE_MIN + px * re_scale;
            double zr = 0.0, zi = 0.0;
            int    iter = 0;

            while (zr*zr + zi*zi < 4.0 && iter < MAX_ITER) {
                double tmp = zr*zr - zi*zi + cr;
                zi = 2.0*zr*zi + ci;
                zr = tmp;
                iter++;
            }

            uchar r, g, b;
            iter_to_rgb(iter, zr, zi, &r, &g, &b);

            size_t idx = ((size_t)py * img->width + px) * CHANNELS;
            img->data[idx + 0] = r;
            img->data[idx + 1] = g;
            img->data[idx + 2] = b;
        }
    }
}


// agrtegamos Bechmark de schedulers
// nos ayuda a comparar static, dynbamic y guided con distintos chucnks sobre una imagen
static void benchmark_schedulers(void)
{
    int BW = 1920, BH = 1080;
    Image bimg = img_alloc(BW, BH);

    double re_scale = (RE_MAX - RE_MIN) / (BW - 1);
    double im_scale = (IM_MAX - IM_MIN) / (BH - 1);

    printf("\n=== BENCHMARK SCHEDULERS (1920x1080) ===\n");
    printf("%-20s %-10s\n", "Scheduler", "Tiempo (s)");
    printf("----------------------------------------\n");

    // Static <---------------
    {
        double t0 = omp_get_wtime();
        #pragma omp parallel for schedule(static) default(none) \
                shared(bimg, re_scale, im_scale, BW, BH)
        for (int py = 0; py < BH; py++) {
            double ci = IM_MAX - py * im_scale;
            for (int px = 0; px < BW; px++) {
                double cr = RE_MIN + px * re_scale, zr=0, zi=0; int it=0;
                while (zr*zr+zi*zi<4.0 && it<MAX_ITER){
                    double t=zr*zr-zi*zi+cr; zi=2*zr*zi+ci; zr=t; it++;
                }
                uchar r,g,b; iter_to_rgb(it,zr,zi,&r,&g,&b);
                size_t idx=((size_t)py*BW+px)*CHANNELS;
                bimg.data[idx]=r; bimg.data[idx+1]=g; bimg.data[idx+2]=b;
            }
        }
        double t1 = omp_get_wtime();
        printf("%-20s %.4f\n", "static (default)", t1-t0);
    }

    // Dynamic con distintos chunks <-----------------
    int chunks[] = {1, 4, 8, 16, 32};
    int nchunks  = 5;
    for (int ci = 0; ci < nchunks; ci++) {
        int ch = chunks[ci];
        double t0 = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic, ch) default(none) \
                shared(bimg, re_scale, im_scale, BW, BH) firstprivate(ch)
        for (int py = 0; py < BH; py++) {
            double cii = IM_MAX - py * im_scale;
            for (int px = 0; px < BW; px++) {
                double cr = RE_MIN + px * re_scale, zr=0, zi=0; int it=0;
                while (zr*zr+zi*zi<4.0 && it<MAX_ITER){
                    double t=zr*zr-zi*zi+cr; zi=2*zr*zi+cii; zr=t; it++;
                }
                uchar r,g,b; iter_to_rgb(it,zr,zi,&r,&g,&b);
                size_t idx=((size_t)py*BW+px)*CHANNELS;
                bimg.data[idx]=r; bimg.data[idx+1]=g; bimg.data[idx+2]=b;
            }
        }
        double t1 = omp_get_wtime();
        char label[32]; snprintf(label, 32, "dynamic, chunk=%d", ch);
        printf("%-20s %.4f\n", label, t1-t0);
    }

    // Guided con distintos chunks <---------------
    for (int ci = 0; ci < 3; ci++) {
        int ch = chunks[ci];
        double t0 = omp_get_wtime();
        #pragma omp parallel for schedule(guided, ch) default(none) \
                shared(bimg, re_scale, im_scale, BW, BH) firstprivate(ch)
        for (int py = 0; py < BH; py++) {
            double cii = IM_MAX - py * im_scale;
            for (int px = 0; px < BW; px++) {
                double cr = RE_MIN + px * re_scale, zr=0, zi=0; int it=0;
                while (zr*zr+zi*zi<4.0 && it<MAX_ITER){
                    double t=zr*zr-zi*zi+cr; zi=2*zr*zi+cii; zr=t; it++;
                }
                uchar r,g,b; iter_to_rgb(it,zr,zi,&r,&g,&b);
                size_t idx=((size_t)py*BW+px)*CHANNELS;
                bimg.data[idx]=r; bimg.data[idx+1]=g; bimg.data[idx+2]=b;
            }
        }
        double t1 = omp_get_wtime();
        char label[32]; snprintf(label, 32, "guided,  chunk=%d", ch);
        printf("%-20s %.4f\n", label, t1-t0);
    }

    printf("----------------------------------------\n\n");
    img_free(&bimg);
}

/* ─────────── Tarea B: Gaussiano paralelizado (baseline IA) ─────────── */

static void build_gaussian_kernel(double kernel[GAUSS_SIZE][GAUSS_SIZE],
                                  double sigma)
{
    double sum = 0.0;
    for (int ky = -GAUSS_RADIUS; ky <= GAUSS_RADIUS; ky++) {
        for (int kx = -GAUSS_RADIUS; kx <= GAUSS_RADIUS; kx++) {
            double val = exp(-(kx*kx + ky*ky) / (2.0*sigma*sigma));
            kernel[ky + GAUSS_RADIUS][kx + GAUSS_RADIUS] = val;
            sum += val;
        }
    }
    for (int ky = 0; ky < GAUSS_SIZE; ky++)
        for (int kx = 0; kx < GAUSS_SIZE; kx++)
            kernel[ky][kx] /= sum;
}

static void apply_gaussian_blur(const Image *src, Image *dst)
{
    printf("[Tarea B] Aplicando Gaussiano (radio=%d) ...\n",
           GAUSS_RADIUS);

    double sigma = GAUSS_RADIUS / 3.0;
    double kernel[GAUSS_SIZE][GAUSS_SIZE];
    build_gaussian_kernel(kernel, sigma);

    int W = src->width, H = src->height;

    #pragma omp parallel for schedule(static) default(none) \
            shared(src, dst, kernel, W, H)
    for (int py = 0; py < src->height; py++) {
        for (int px = 0; px < src->width; px++) {

            // separamos los acumuladores por canales
            double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0;

            for (int ky = -GAUSS_RADIUS; ky <= GAUSS_RADIUS; ky++) {
                int sy = py + ky;
                if (sy < 0)                sy = 0;
                else if (sy >= H) sy = H - 1;

                #pragma omp simd reduction(+:acc0,acc1,acc2)
                for (int kx = -GAUSS_RADIUS; kx <= GAUSS_RADIUS; kx++) {
                    int sx = px + kx;
                    if (sx < 0)               sx = 0;
                    else if (sx >= W) sx = W - 1;

                    double w = kernel[ky + GAUSS_RADIUS][kx + GAUSS_RADIUS];
                    size_t sidx = ((size_t)sy * W + sx) * CHANNELS;

                    acc0 += src->data[sidx + 0] * w;
                    acc1 += src->data[sidx + 1] * w;
                    acc2 += src->data[sidx + 2] * w;
                }
            }

            size_t didx = ((size_t)py * W + px) * CHANNELS;
            dst->data[didx + 0] = (uchar)(acc0 + 0.5);
            dst->data[didx + 1] = (uchar)(acc1 + 0.5);
            dst->data[didx + 2] = (uchar)(acc2 + 0.5);
        }
    }
}

// ------------------------ Histograma de colores -----------------------------------
/* se hace una reducción local donde cada hilo acumula en arreglos privados en su propio stack
    no hace false sharing puesto que cada hilo escribe en su propia memoria
    ademas que solo se dincroniza una vez al final con lo de #pragma omp critical */
static void contar_pixeles_atomic(const Image *img,
                                   long hist_r[HIST_BINS],
                                   long hist_g[HIST_BINS],
                                   long hist_b[HIST_BINS])
{
    // Limpiamos los contadores antes de empezar 
    memset(hist_r, 0, HIST_BINS * sizeof(long));
    memset(hist_g, 0, HIST_BINS * sizeof(long));
    memset(hist_b, 0, HIST_BINS * sizeof(long));

    size_t num_pixeles = (size_t)img->width * img->height;

    // Todos los hilos escriben en el mismo arreglo global, atomic evita que se pisen pero genera mucha contension 
    #pragma omp parallel for default(none) \
            shared(img, hist_r, hist_g, hist_b) firstprivate(num_pixeles)
    for (size_t i = 0; i < num_pixeles; i++) {
        #pragma omp atomic
        hist_r[img->data[i*CHANNELS + 0]]++;
        #pragma omp atomic
        hist_g[img->data[i*CHANNELS + 1]]++;
        #pragma omp atomic
        hist_b[img->data[i*CHANNELS + 2]]++;
    }
}

// Version optimizada con contadores locales — sin false sharing 
static void contar_pixeles_reduccion(const Image *img,
                                      long hist_r[HIST_BINS],
                                      long hist_g[HIST_BINS],
                                      long hist_b[HIST_BINS])
{
    // Limpiamos los contadores globales 
    memset(hist_r, 0, HIST_BINS * sizeof(long));
    memset(hist_g, 0, HIST_BINS * sizeof(long));
    memset(hist_b, 0, HIST_BINS * sizeof(long));

    size_t num_pixeles = (size_t)img->width * img->height;

    #pragma omp parallel default(none) \
            shared(img, hist_r, hist_g, hist_b) firstprivate(num_pixeles)
    {
        /* Cada hilo tiene su propia copia de los contadores en el stack,
            eso evita el false sharing porque cada quien escribe en su
            propia memoria y no se pisan entre hilos */
        long conteo_r[HIST_BINS] = {0};
        long conteo_g[HIST_BINS] = {0};
        long conteo_b[HIST_BINS] = {0};

        // Contamos libremente sin necesitar sincronizacion 
        #pragma omp for
        for (size_t i = 0; i < num_pixeles; i++) {
            conteo_r[img->data[i*CHANNELS + 0]]++;
            conteo_g[img->data[i*CHANNELS + 1]]++;
            conteo_b[img->data[i*CHANNELS + 2]]++;
        }

        // Solo al final cada hilo suma su conteo al total, el critical aqui es barato porque solo pasa una vez por hilo 
        #pragma omp critical
        {
            for (int k = 0; k < HIST_BINS; k++) {
                hist_r[k] += conteo_r[k];
                hist_g[k] += conteo_g[k];
                hist_b[k] += conteo_b[k];
            }
        }
    }
}

// --------------------------- Demostración de false sharing -----------------------
/* En este caso creamos un arrglo 2D global sin padding especial para que cada fila ocupe
    128 bytes(aprox. 2 lineas de chache) tambien las filas de hilos adyacentes quedan en la misma linea de cache. */
#define BINS_SMALL 16
static void histogram_false_sharing_demo(const Image *img, int nthreads)
{
    long (*shared_hist)[BINS_SMALL] =
        calloc(nthreads, BINS_SMALL * sizeof(long));
    if (!shared_hist) { fprintf(stderr,"OOM hist\n"); return; }

    size_t npix = (size_t)img->width * img->height;
    uchar  mask = BINS_SMALL - 1;

    #pragma omp parallel num_threads(nthreads) default(none) \
            shared(img, shared_hist, npix, mask)
    {
        int tid = omp_get_thread_num();
        /* Escribe en su fila — adyacente a las de otros hilos en memoria */
        #pragma omp for
        for (size_t i = 0; i < npix; i++)
            shared_hist[tid][ img->data[i*CHANNELS] & mask ]++;
    }

    free(shared_hist);
}


/* ──────────────────────────── main ─────────────────────────────────── */
int main(void)
{

    int max_threads = omp_get_max_threads();
    printf("=== Mandelbrot 8K + Gaussiano (OpenMP OPTIMIZADO) ===\n");
    printf("Hilos disponibles: %d\n", max_threads);
    printf("Resolucion: %d x %d | MaxIter: %d | Kernel: %dx%d\n\n",
           WIDTH, HEIGHT, MAX_ITER, GAUSS_SIZE, GAUSS_SIZE);

    // Se muestra la configuracion de afinidad si esta activa 
    char *proc_bind = getenv("OMP_PROC_BIND");
    char *places    = getenv("OMP_PLACES");
    if (proc_bind || places) {
        printf("Afinidad: OMP_PROC_BIND=%s  OMP_PLACES=%s\n",
               proc_bind ? proc_bind : "(no set)",
               places    ? places    : "(no set)");
    }
    printf("\n");

    benchmark_schedulers();

    Image raw    = img_alloc(WIDTH, HEIGHT);
    Image blured = img_alloc(WIDTH, HEIGHT);

    /* Tarea A */
    double t0 = omp_get_wtime();
    generate_mandelbrot(&raw);
    double t1 = omp_get_wtime();
    printf("  Tiempo Tarea A: %.3f s\n\n", t1 - t0);

    /* Tarea B */
    double t2 = omp_get_wtime();
    apply_gaussian_blur(&raw, &blured);
    double t3 = omp_get_wtime();
    printf("  Tiempo Tarea B: %.3f s\n\n", t3 - t2);

    printf("Tiempo total: %.3f s\n\n", (t1-t0) + (t3-t2));

    // Histograma: comparacion de estrategias 
    printf("=== HISTOGRAMA DE COLORES ===\n");
    long hist_r[HIST_BINS], hist_g[HIST_BINS], hist_b[HIST_BINS];

    // 1. con atomic
    double ha0 = omp_get_wtime();
    contar_pixeles_atomic(&blured, hist_r, hist_g, hist_b);
    double ha1 = omp_get_wtime();
    printf("  Atomic:          %.4f s\n", ha1 - ha0);

    // 2. con reduccion local 
    double hr0 = omp_get_wtime();
    contar_pixeles_reduccion(&blured, hist_r, hist_g, hist_b);
    double hr1 = omp_get_wtime();
    printf("  Reduccion local: %.4f s\n", hr1 - hr0);
    printf("  Speedup reduccion vs atomic: %.2fx\n", (ha1-ha0)/(hr1-hr0));

    // 3. false sharing demo 
    double hf0 = omp_get_wtime();
    histogram_false_sharing_demo(&blured, max_threads);
    double hf1 = omp_get_wtime();
    printf("  False sharing demo: %.4f s\n\n", hf1 - hf0);


    img_save_ppm(&raw,    "mandelbrot_raw_omp.ppm");
    img_save_ppm(&blured, "mandelbrot_blurred_omp.ppm");

    img_free(&raw);
    img_free(&blured);

    printf("Listo.\n");
    return 0;
}