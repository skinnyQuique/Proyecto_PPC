/*
 * mandelbrot_openmp_baseline.c
 * Proyecto Final - Programacion Paralela y Concurrente
 *
 * "Linea Base Paralela de la IA" — primera paralelizacion con OpenMP
 * generada a partir del codigo secuencial base.
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

/* ──────────────────────────── Paleta ───────────────────────────────── */
static void iter_to_rgb(int iter, double zr, double zi,
                        uchar *r, uchar *g, uchar *b)
{
    if (iter == MAX_ITER) { *r = *g = *b = 0; return; }

    double log_zn = log(zr*zr + zi*zi) * 0.5;
    double nu     = log(log_zn / log(2.0)) / log(2.0);
    double smooth = (double)iter + 1.0 - nu;
    double t      = smooth / (double)MAX_ITER;

    double a[3]  = {0.5, 0.5, 0.5};
    double bk[3] = {0.5, 0.5, 0.5};
    double c[3]  = {1.0, 1.0, 1.0};
    double d[3]  = {0.0, 0.33, 0.67};

    *r = (uchar)((a[0] + bk[0]*cos(6.28318*(c[0]*t + d[0]))) * 255);
    *g = (uchar)((a[1] + bk[1]*cos(6.28318*(c[1]*t + d[1]))) * 255);
    *b = (uchar)((a[2] + bk[2]*cos(6.28318*(c[2]*t + d[2]))) * 255);
}

/* ─────────── Tarea A: Mandelbrot paralelizado (baseline IA) ─────────── */
/*
 * NOTA DE LA IA: Se aplica "#pragma omp parallel for" al bucle externo
 * (filas). Cada hilo es independiente: no comparten estado mutable,
 * por lo que no se necesita sincronizacion. El planificador por defecto
 * es "static", que reparte filas en bloques contiguos iguales.
 *
 * PROBLEMA CONOCIDO DEL BASELINE: el conjunto de Mandelbrot tiene
 * carga MUY desigual — las filas cerca del borde del conjunto requieren
 * MAX_ITER iteraciones, mientras que las del interior se resuelven
 * rapidamente. Con static scheduling esto genera desbalance de carga.
 * La optimizacion manual corregira esto con dynamic/guided.
 */
static void generate_mandelbrot(Image *img)
{
    printf("[Tarea A] Generando Mandelbrot %dx%d con OpenMP ...\n",
           img->width, img->height);

    double re_scale = (RE_MAX - RE_MIN) / (img->width  - 1);
    double im_scale = (IM_MAX - IM_MIN) / (img->height - 1);

    #pragma omp parallel for schedule(static) default(none) \
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

/* ─────────── Tarea B: Gaussiano paralelizado (baseline IA) ─────────── */
/*
 * NOTA DE LA IA: Se paraleliza el bucle externo (filas de salida).
 * La imagen fuente es solo lectura; la imagen destino se escribe en
 * posiciones disjuntas por cada hilo => sin condicion de carrera.
 * El kernel se construye fuera del paralelo para no duplicar trabajo.
 *
 * PROBLEMA CONOCIDO DEL BASELINE: el kernel Gaussiano se declara como
 * arreglo VLA dentro de la region paralela en algunas versiones; aqui
 * se eleva al scope de la funcion para evitar esa ineficiencia.
 */
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
    printf("[Tarea B] Aplicando Gaussiano (radio=%d) con OpenMP ...\n",
           GAUSS_RADIUS);

    double sigma = GAUSS_RADIUS / 3.0;
    double kernel[GAUSS_SIZE][GAUSS_SIZE];
    build_gaussian_kernel(kernel, sigma);

    #pragma omp parallel for schedule(static) default(none) \
            shared(src, dst, kernel)
    for (int py = 0; py < src->height; py++) {
        for (int px = 0; px < src->width; px++) {

            double acc[CHANNELS] = {0.0, 0.0, 0.0};

            for (int ky = -GAUSS_RADIUS; ky <= GAUSS_RADIUS; ky++) {
                int sy = py + ky;
                if (sy < 0)                sy = 0;
                else if (sy >= src->height) sy = src->height - 1;

                for (int kx = -GAUSS_RADIUS; kx <= GAUSS_RADIUS; kx++) {
                    int sx = px + kx;
                    if (sx < 0)               sx = 0;
                    else if (sx >= src->width) sx = src->width - 1;

                    double w = kernel[ky + GAUSS_RADIUS][kx + GAUSS_RADIUS];
                    size_t sidx = ((size_t)sy * src->width + sx) * CHANNELS;

                    acc[0] += src->data[sidx + 0] * w;
                    acc[1] += src->data[sidx + 1] * w;
                    acc[2] += src->data[sidx + 2] * w;
                }
            }

            size_t didx = ((size_t)py * src->width + px) * CHANNELS;
            dst->data[didx + 0] = (uchar)(acc[0] + 0.5);
            dst->data[didx + 1] = (uchar)(acc[1] + 0.5);
            dst->data[didx + 2] = (uchar)(acc[2] + 0.5);
        }
    }
}

/* ──────────────────────────── main ─────────────────────────────────── */
int main(void)
{
    double t0, t1, t2, t3;

    int max_threads = omp_get_max_threads();
    printf("=== Mandelbrot 8K + Gaussiano (OpenMP BASELINE IA) ===\n");
    printf("Hilos disponibles: %d\n", max_threads);
    printf("Resolucion: %d x %d | MaxIter: %d | Kernel: %dx%d\n\n",
           WIDTH, HEIGHT, MAX_ITER, GAUSS_SIZE, GAUSS_SIZE);

    Image raw    = img_alloc(WIDTH, HEIGHT);
    Image blured = img_alloc(WIDTH, HEIGHT);

    /* Tarea A */
    t0 = omp_get_wtime();
    generate_mandelbrot(&raw);
    t1 = omp_get_wtime();
    printf("  Tiempo Tarea A: %.3f s\n\n", t1 - t0);

    /* Tarea B */
    t2 = omp_get_wtime();
    apply_gaussian_blur(&raw, &blured);
    t3 = omp_get_wtime();
    printf("  Tiempo Tarea B: %.3f s\n\n", t3 - t2);

    double total = (t1 - t0) + (t3 - t2);
    printf("Tiempo total: %.3f s\n\n", total);

    img_save_ppm(&raw,    "mandelbrot_raw_omp.ppm");
    img_save_ppm(&blured, "mandelbrot_blurred_omp.ppm");

    img_free(&raw);
    img_free(&blured);

    printf("Listo.\n");
    return 0;
}