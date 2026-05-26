/*
 * mandelbrot_sequential.c
 * Proyecto Final - Programacion Paralela y Concurrente
 *
 * Tarea A: Genera imagen 8K del conjunto de Mandelbrot (PPM)
 * Tarea B: Aplica filtro Gaussiano 2D (radio 15) sobre la imagen
 *
 * Compilar: gcc -O2 -o mandelbrot_seq mandelbrot_sequential.c -lm
 * Uso:      ./mandelbrot_seq
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/* Tiempo en segundos usando clock() — portable en C89/C99/C11 */
static double now_sec(void)
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

/* ──────────────────────────── Configuracion ──────────────────────────── */
#define WIDTH       7680   /* 8K horizontal */
#define HEIGHT      4320   /* 8K vertical   */
#define MAX_ITER    1000
#define CHANNELS    3      /* RGB            */

/* Region del plano complejo */
#define RE_MIN  -2.5
#define RE_MAX   1.0
#define IM_MIN  -1.25
#define IM_MAX   1.25

/* Radio del filtro Gaussiano */
#define GAUSS_RADIUS 15
#define GAUSS_SIZE   (2*GAUSS_RADIUS+1)   /* 31x31 kernel */

typedef unsigned char uchar;

/* ──────────────────────────── Estructuras ────────────────────────────── */
typedef struct {
    uchar *data;   /* interleaved RGB: [r0,g0,b0, r1,g1,b1, ...] */
    int    width;
    int    height;
} Image;

/* ──────────────────────────── Utilidades ────────────────────────────── */

/* Asigna imagen; aborta si no hay memoria */
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

/* Escribe formato PPM binario (P6) */
static void img_save_ppm(const Image *img, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return; }
    fprintf(f, "P6\n%d %d\n255\n", img->width, img->height);
    fwrite(img->data, 1, (size_t)img->width * img->height * CHANNELS, f);
    fclose(f);
    printf("  Guardada: %s\n", path);
}

/* ──────────────────────────── Paleta de color ───────────────────────── */
/*
 * Coloracion suave (smooth coloring) para evitar bandas abruptas.
 * Mapea iteraciones [0..MAX_ITER] a un color RGB mediante una rampa
 * de color ciclica basada en seno.
 */
static void iter_to_rgb(int iter, double zr, double zi,
                        uchar *r, uchar *g, uchar *b)
{
    if (iter == MAX_ITER) { *r = *g = *b = 0; return; }

    /* Normalizacion suave */
    double log_zn  = log(zr*zr + zi*zi) * 0.5;
    double nu      = log(log_zn / log(2.0)) / log(2.0);
    double smooth  = (double)iter + 1.0 - nu;
    double t       = smooth / (double)MAX_ITER;

    /* Paleta coseno (inspirada en Inigo Quilez) */
    double a[3] = {0.5, 0.5, 0.5};
    double b_[3] = {0.5, 0.5, 0.5};
    double c[3] = {1.0, 1.0, 1.0};
    double d[3] = {0.0, 0.33, 0.67};

    *r = (uchar)((a[0] + b_[0]*cos(6.28318*(c[0]*t + d[0]))) * 255);
    *g = (uchar)((a[1] + b_[1]*cos(6.28318*(c[1]*t + d[1]))) * 255);
    *b = (uchar)((a[2] + b_[2]*cos(6.28318*(c[2]*t + d[2]))) * 255);
}

/* ──────────────────────────── Tarea A: Mandelbrot ───────────────────── */
static void generate_mandelbrot(Image *img)
{
    printf("[Tarea A] Generando Mandelbrot %dx%d ...\n", img->width, img->height);

    double re_scale = (RE_MAX - RE_MIN) / (img->width  - 1);
    double im_scale = (IM_MAX - IM_MIN) / (img->height - 1);

    for (int py = 0; py < img->height; py++) {
        double ci = IM_MAX - py * im_scale;   /* de arriba hacia abajo */

        for (int px = 0; px < img->width; px++) {
            double cr = RE_MIN + px * re_scale;

            double zr = 0.0, zi = 0.0;
            int iter = 0;

            /* Iteracion Z = Z^2 + C */
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

/* ──────────────────────────── Tarea B: Filtro Gaussiano ─────────────── */

/* Construye kernel Gaussiano normalizado en coma flotante */
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
    /* Normalizar */
    for (int ky = 0; ky < GAUSS_SIZE; ky++)
        for (int kx = 0; kx < GAUSS_SIZE; kx++)
            kernel[ky][kx] /= sum;
}

static void apply_gaussian_blur(const Image *src, Image *dst)
{
    printf("[Tarea B] Aplicando Gaussiano %dx%d (radio=%d) ...\n",
           src->width, src->height, GAUSS_RADIUS);

    /* Sigma = radius/3 es convencion comun */
    double sigma = GAUSS_RADIUS / 3.0;
    double kernel[GAUSS_SIZE][GAUSS_SIZE];
    build_gaussian_kernel(kernel, sigma);

    /* Convolución 2D con manejo de bordes por clamp */
    for (int py = 0; py < src->height; py++) {
        for (int px = 0; px < src->width; px++) {

            double acc[CHANNELS] = {0.0, 0.0, 0.0};

            for (int ky = -GAUSS_RADIUS; ky <= GAUSS_RADIUS; ky++) {
                int sy = py + ky;
                if (sy < 0)           sy = 0;
                else if (sy >= src->height) sy = src->height - 1;

                for (int kx = -GAUSS_RADIUS; kx <= GAUSS_RADIUS; kx++) {
                    int sx = px + kx;
                    if (sx < 0)            sx = 0;
                    else if (sx >= src->width)  sx = src->width - 1;

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

    printf("=== Mandelbrot 8K + Filtro Gaussiano (SECUENCIAL) ===\n");
    printf("Resolucion: %d x %d\n", WIDTH, HEIGHT);
    printf("MaxIter: %d  |  Kernel Gauss: %dx%d\n\n",
           MAX_ITER, GAUSS_SIZE, GAUSS_SIZE);

    /* ── Asignacion de memoria ── */
    Image raw    = img_alloc(WIDTH, HEIGHT);
    Image blured = img_alloc(WIDTH, HEIGHT);

    /* ── Tarea A ── */
    t0 = now_sec();
    generate_mandelbrot(&raw);
    t1 = now_sec();
    printf("  Tiempo Tarea A: %.3f s\n\n", t1 - t0);

    /* ── Tarea B ── */
    t2 = now_sec();
    apply_gaussian_blur(&raw, &blured);
    t3 = now_sec();
    printf("  Tiempo Tarea B: %.3f s\n\n", t3 - t2);

    /* ── Tiempos totales ── */
    double total = (t1 - t0) + (t3 - t2);
    printf("Tiempo total: %.3f s\n\n", total);

    /* ── Guardado ── */
    img_save_ppm(&raw,    "mandelbrot_raw.ppm");
    img_save_ppm(&blured, "mandelbrot_blurred.ppm");

    img_free(&raw);
    img_free(&blured);

    printf("\nListo.\n");
    return 0;
}