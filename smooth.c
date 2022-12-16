#include <stdio.h>
#include <stdlib.h>
#include "defs.h"
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

/* 
 * Please fill in the following team struct 
 */
who_t who = {
    "jlars789",           /* Scoreboard name */

    "Joseph Larson",      /* First member full name */
    "gha3bn@virginia.edu",     /* First member email address */
};

/*** UTILITY FUNCTIONS ***/

/* You are free to use these utility functions, or write your own versions
 * of them. */

/* A struct used to compute averaged pixel value */
typedef struct {
    unsigned short red;
    unsigned short green;
    unsigned short blue;
    unsigned short alpha;
    unsigned short num;
} pixel_sum;

/* Compute min and max of two integers, respectively */
static int min(int a, int b) { return (a < b ? a : b); }
static int max(int a, int b) { return (a > b ? a : b); }

/* 
 * initialize_pixel_sum - Initializes all fields of sum to 0 
 */
static void initialize_pixel_sum(pixel_sum *sum) 
{
    sum->red = sum->green = sum->blue = sum->alpha = 0;
    sum->num = 0;
    return;
}

/* 
 * accumulate_sum - Accumulates field values of p in corresponding 
 * fields of sum 
 */
static void accumulate_sum(pixel_sum *sum, pixel p) 
{
    sum->red += (int) p.red;
    sum->green += (int) p.green;
    sum->blue += (int) p.blue;
    sum->alpha += (int) p.alpha;
    sum->num++;
    return;
}

/* 
 * assign_sum_to_pixel - Computes averaged pixel value in current_pixel 
 */
static void assign_sum_to_pixel(pixel *current_pixel, pixel_sum sum) 
{
    current_pixel->red = (unsigned short) (sum.red/sum.num);
    current_pixel->green = (unsigned short) (sum.green/sum.num);
    current_pixel->blue = (unsigned short) (sum.blue/sum.num);
    current_pixel->alpha = (unsigned short) (sum.alpha/sum.num);
    return;
}

/* 
 * avg - Returns averaged pixel value at (i,j) 
 */
static pixel avg(int dim, int i, int j, pixel *src) 
{
    pixel_sum sum;
    pixel current_pixel;

    initialize_pixel_sum(&sum);
    for(int jj=max(j-1, 0); jj <= min(j+1, dim-1); jj++) 
	for(int ii=max(i-1, 0); ii <= min(i+1, dim-1); ii++) 
	    accumulate_sum(&sum, src[RIDX(ii,jj,dim)]);

    assign_sum_to_pixel(&current_pixel, sum);
 
    return current_pixel;
}

static pixel new_avg(int dim, int i, int j, pixel *src){


    pixel_sum sum;
    pixel current_pixel;

    initialize_pixel_sum(&sum);
    if(i > 0 && i < dim-1 && j > 0 && j < dim-1){
        accumulate_sum(&sum, src[RIDX(i-1,j-1,dim)]);
        accumulate_sum(&sum, src[RIDX(i-1,j,dim)]);
        accumulate_sum(&sum, src[RIDX(i-1,j+1,dim)]);
        accumulate_sum(&sum, src[RIDX(i,j-1,dim)]);
        accumulate_sum(&sum, src[RIDX(i,j,dim)]);
        accumulate_sum(&sum, src[RIDX(i,j+1,dim)]);
        accumulate_sum(&sum, src[RIDX(i+1,j-1,dim)]);
        accumulate_sum(&sum, src[RIDX(i+1,j,dim)]);
        accumulate_sum(&sum, src[RIDX(i+1,j+1,dim)]);
    }
    else if(i == 0){

        if(j > 0 && j < dim-1){
            accumulate_sum(&sum, src[RIDX(i,j-1,dim)]);
            accumulate_sum(&sum, src[RIDX(i,j,dim)]);
            accumulate_sum(&sum, src[RIDX(i,j+1,dim)]);
            accumulate_sum(&sum, src[RIDX(i+1,j-1,dim)]);
            accumulate_sum(&sum, src[RIDX(i+1,j,dim)]);
            accumulate_sum(&sum, src[RIDX(i+1,j+1,dim)]);
        }
        else if(j == 0){
            accumulate_sum(&sum, src[RIDX(i,j,dim)]);
            accumulate_sum(&sum, src[RIDX(i,j+1,dim)]);
            accumulate_sum(&sum, src[RIDX(i+1,j,dim)]);
            accumulate_sum(&sum, src[RIDX(i+1,j+1,dim)]);
        }
        else {
            accumulate_sum(&sum, src[RIDX(i,j,dim)]);
            accumulate_sum(&sum, src[RIDX(i,j-1,dim)]);
            accumulate_sum(&sum, src[RIDX(i+1,j,dim)]);
            accumulate_sum(&sum, src[RIDX(i+1,j-1,dim)]);
        } 
    } 
    else if (i == dim-1) {
        if(j > 0 && j < dim-1){
            accumulate_sum(&sum, src[RIDX(i-1,j-1,dim)]);
            accumulate_sum(&sum, src[RIDX(i-1,j,dim)]);
            accumulate_sum(&sum, src[RIDX(i-1,j+1,dim)]);
            accumulate_sum(&sum, src[RIDX(i,j-1,dim)]);
            accumulate_sum(&sum, src[RIDX(i,j,dim)]);
            accumulate_sum(&sum, src[RIDX(i,j+1,dim)]);
        }
        else if(j == 0){
            accumulate_sum(&sum, src[RIDX(i-1,j,dim)]);
            accumulate_sum(&sum, src[RIDX(i-1,j+1,dim)]);
            accumulate_sum(&sum, src[RIDX(i,j,dim)]);
            accumulate_sum(&sum, src[RIDX(i,j+1,dim)]);
        }
        else {
            accumulate_sum(&sum, src[RIDX(i-1,j,dim)]);
            accumulate_sum(&sum, src[RIDX(i-1,j-1,dim)]);
            accumulate_sum(&sum, src[RIDX(i,j,dim)]);
            accumulate_sum(&sum, src[RIDX(i,j-1,dim)]);
        } 
    } 
    else if(j == 0){
        accumulate_sum(&sum, src[RIDX(i-1,j,dim)]);
        accumulate_sum(&sum, src[RIDX(i-1,j+1,dim)]);
        accumulate_sum(&sum, src[RIDX(i,j,dim)]);
        accumulate_sum(&sum, src[RIDX(i,j+1,dim)]);
        accumulate_sum(&sum, src[RIDX(i+1,j,dim)]);
        accumulate_sum(&sum, src[RIDX(i+1,j+1,dim)]);
    } else {
        accumulate_sum(&sum, src[RIDX(i-1,j,dim)]);
        accumulate_sum(&sum, src[RIDX(i-1,j-1,dim)]);
        accumulate_sum(&sum, src[RIDX(i,j,dim)]);
        accumulate_sum(&sum, src[RIDX(i,j-1,dim)]);
        accumulate_sum(&sum, src[RIDX(i+1,j,dim)]);
        accumulate_sum(&sum, src[RIDX(i+1,j-1,dim)]);
    }

    assign_sum_to_pixel(&current_pixel, sum);
 
    return current_pixel;
    
}



/******************************************************
 * Your different versions of the smooth go here
 ******************************************************/

/* 
 * naive_smooth - The naive baseline version of smooth
 */
char naive_smooth_descr[] = "naive_smooth: Naive baseline implementation";
void naive_smooth(int dim, pixel *src, pixel *dst) 
{
    for (int i = 0; i < dim; i++)
	for (int j = 0; j < dim; j++)
            dst[RIDX(i,j, dim)] = avg(dim, i, j, src);
}

char naive_smooth_na_descr[] = "Naive with new average";
void naive_smooth_na(int dim, pixel *src, pixel *dst) 
{
    for (int i = 0; i < dim; i++)
	for (int j = 0; j < dim; j++){
            pixel_sum sum;
            pixel current_pixel;

            initialize_pixel_sum(&sum);
            if(i > 0 && i < dim-1 && j > 0 && j < dim-1){
                accumulate_sum(&sum, src[RIDX(i-1,j-1,dim)]);
                accumulate_sum(&sum, src[RIDX(i-1,j,dim)]);
                accumulate_sum(&sum, src[RIDX(i-1,j+1,dim)]);
                accumulate_sum(&sum, src[RIDX(i,j-1,dim)]);
                accumulate_sum(&sum, src[RIDX(i,j,dim)]);
                accumulate_sum(&sum, src[RIDX(i,j+1,dim)]);
                accumulate_sum(&sum, src[RIDX(i+1,j-1,dim)]);
                accumulate_sum(&sum, src[RIDX(i+1,j,dim)]);
                accumulate_sum(&sum, src[RIDX(i+1,j+1,dim)]);
            }
            else if(i == 0){

                if(j > 0 && j < dim-1){
                    accumulate_sum(&sum, src[RIDX(i,j-1,dim)]);
                    accumulate_sum(&sum, src[RIDX(i,j,dim)]);
                    accumulate_sum(&sum, src[RIDX(i,j+1,dim)]);
                    accumulate_sum(&sum, src[RIDX(i+1,j-1,dim)]);
                    accumulate_sum(&sum, src[RIDX(i+1,j,dim)]);
                    accumulate_sum(&sum, src[RIDX(i+1,j+1,dim)]);
                }
                else if(j == 0){
                    accumulate_sum(&sum, src[RIDX(i,j,dim)]);
                    accumulate_sum(&sum, src[RIDX(i,j+1,dim)]);
                    accumulate_sum(&sum, src[RIDX(i+1,j,dim)]);
                    accumulate_sum(&sum, src[RIDX(i+1,j+1,dim)]);
                }
                else {
                    accumulate_sum(&sum, src[RIDX(i,j,dim)]);
                    accumulate_sum(&sum, src[RIDX(i,j-1,dim)]);
                    accumulate_sum(&sum, src[RIDX(i+1,j,dim)]);
                    accumulate_sum(&sum, src[RIDX(i+1,j-1,dim)]);
                } 
            } 
            else if (i == dim-1) {
                if(j > 0 && j < dim-1){
                    accumulate_sum(&sum, src[RIDX(i-1,j-1,dim)]);
                    accumulate_sum(&sum, src[RIDX(i-1,j,dim)]);
                    accumulate_sum(&sum, src[RIDX(i-1,j+1,dim)]);
                    accumulate_sum(&sum, src[RIDX(i,j-1,dim)]);
                    accumulate_sum(&sum, src[RIDX(i,j,dim)]);
                    accumulate_sum(&sum, src[RIDX(i,j+1,dim)]);
                }
                else if(j == 0){
                    accumulate_sum(&sum, src[RIDX(i-1,j,dim)]);
                    accumulate_sum(&sum, src[RIDX(i-1,j+1,dim)]);
                    accumulate_sum(&sum, src[RIDX(i,j,dim)]);
                    accumulate_sum(&sum, src[RIDX(i,j+1,dim)]);
                }
                else {
                    accumulate_sum(&sum, src[RIDX(i-1,j,dim)]);
                    accumulate_sum(&sum, src[RIDX(i-1,j-1,dim)]);
                    accumulate_sum(&sum, src[RIDX(i,j,dim)]);
                    accumulate_sum(&sum, src[RIDX(i,j-1,dim)]);
                } 
            } 
            else if(j == 0){
                accumulate_sum(&sum, src[RIDX(i-1,j,dim)]);
                accumulate_sum(&sum, src[RIDX(i-1,j+1,dim)]);
                accumulate_sum(&sum, src[RIDX(i,j,dim)]);
                accumulate_sum(&sum, src[RIDX(i,j+1,dim)]);
                accumulate_sum(&sum, src[RIDX(i+1,j,dim)]);
                accumulate_sum(&sum, src[RIDX(i+1,j+1,dim)]);
            } else {
                accumulate_sum(&sum, src[RIDX(i-1,j,dim)]);
                accumulate_sum(&sum, src[RIDX(i-1,j-1,dim)]);
                accumulate_sum(&sum, src[RIDX(i,j,dim)]);
                accumulate_sum(&sum, src[RIDX(i,j-1,dim)]);
                accumulate_sum(&sum, src[RIDX(i+1,j,dim)]);
                accumulate_sum(&sum, src[RIDX(i+1,j-1,dim)]);
            }

        assign_sum_to_pixel(&current_pixel, sum);
        dst[RIDX(i,j, dim)] = current_pixel;
    }
}


char avx_smooth_descr[] = "AVX Smooth";
void avx_smooth(int dim, pixel *src, pixel *dst) 
{
    for(int i = 0; i < dim; i++){
        for(int j = 0; j < dim; j++){
            
          __m256i total_sum = _mm256_setzero_si256();
            
            if(i > 0 && i < dim-1 && j > 0 && j < dim-1){
                
                __m128i tl_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j-1, dim)]);
                __m128i tm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j, dim)]);
                __m128i tr_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j+1, dim)]);
                __m128i ml_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j-1, dim)]);
                __m128i mm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j, dim)]);
                __m128i mr_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j+1, dim)]);
                __m128i bl_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j-1, dim)]);
                __m128i bm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j, dim)]);
                __m128i br_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j+1, dim)]);


                __m256i tl_converted = _mm256_cvtepu8_epi16(tl_pixel);
                __m256i tm_converted = _mm256_cvtepu8_epi16(tm_pixel);
                __m256i tr_converted = _mm256_cvtepu8_epi16(tr_pixel);
                __m256i ml_converted = _mm256_cvtepu8_epi16(ml_pixel);
                __m256i mm_converted = _mm256_cvtepu8_epi16(mm_pixel);
                __m256i mr_converted = _mm256_cvtepu8_epi16(mr_pixel);
                __m256i bl_converted = _mm256_cvtepu8_epi16(bl_pixel);
                __m256i bm_converted = _mm256_cvtepu8_epi16(bm_pixel);
                __m256i br_converted = _mm256_cvtepu8_epi16(br_pixel);

                
                __m256i top_sum = tl_converted;
                top_sum = _mm256_add_epi16(top_sum, tm_converted);
                top_sum = _mm256_add_epi16(top_sum, tr_converted);

                __m256i mid_sum = ml_converted;
                mid_sum = _mm256_add_epi16(mid_sum, mm_converted);
                mid_sum = _mm256_add_epi16(mid_sum, mr_converted);

                __m256i bot_sum = bl_converted;
                bot_sum = _mm256_add_epi16(bot_sum, bm_converted);
                bot_sum = _mm256_add_epi16(bot_sum, br_converted);

                total_sum = top_sum;
                total_sum = _mm256_add_epi16(total_sum, mid_sum);
                total_sum = _mm256_add_epi16(total_sum, bot_sum);

                unsigned short pixel_elements[16];
                _mm256_storeu_si256((__m256i*) pixel_elements, total_sum);

                dst[RIDX(i,j, dim)].red = (pixel_elements[0] * 7282) >> 16;
                dst[RIDX(i,j, dim)].green = (pixel_elements[1] * 7282) >> 16;
                dst[RIDX(i,j, dim)].blue = (pixel_elements[2] * 7282) >> 16;
                dst[RIDX(i,j, dim)].alpha = (pixel_elements[3] * 7282) >> 16;
               
            } else {
                int div = 6;
                __m128i mm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j, dim)]);
                __m256i mm_converted = _mm256_cvtepu8_epi16(mm_pixel);

                if(i == 0){

                    __m128i bm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j, dim)]);
                    __m256i bm_converted = _mm256_cvtepu8_epi16(bm_pixel);

                    total_sum = _mm256_add_epi16(mm_converted, bm_converted);
                    __m256i right_sum = _mm256_setzero_si256();
                    __m256i left_sum = _mm256_setzero_si256();

                    if(j < dim-1){
                        __m128i mr_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j+1, dim)]);
                        __m128i br_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j+1, dim)]);

                        __m256i mr_converted = _mm256_cvtepu8_epi16(mr_pixel);
                        __m256i br_converted = _mm256_cvtepu8_epi16(br_pixel);

                        right_sum = _mm256_add_epi16(mr_converted, br_converted);
                        total_sum = _mm256_add_epi16(total_sum, right_sum);
                    } else {
                        div = 4;
                    }

                    if(j > 0){
                        __m128i ml_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j-1, dim)]);
                        __m128i bl_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j-1, dim)]);

                        __m256i ml_converted = _mm256_cvtepu8_epi16(ml_pixel);
                        __m256i bl_converted = _mm256_cvtepu8_epi16(bl_pixel);

                        left_sum = _mm256_add_epi16(ml_converted, bl_converted);
                        total_sum = _mm256_add_epi16(total_sum, left_sum);
                    } else {
                        div = 4;
                    }
                }
//###############################################################################################################
                else if(i == dim-1){
                    __m128i tm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j, dim)]);
                    __m256i tm_converted = _mm256_cvtepu8_epi16(tm_pixel);

                    total_sum = _mm256_add_epi16(mm_converted, tm_converted);
                    __m256i right_sum = _mm256_setzero_si256();
                    __m256i left_sum = _mm256_setzero_si256();

                    if(j < dim-1){
                        __m128i mr_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j+1, dim)]);
                        __m128i tr_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j+1, dim)]);

                        __m256i mr_converted = _mm256_cvtepu8_epi16(mr_pixel);
                        __m256i tr_converted = _mm256_cvtepu8_epi16(tr_pixel);

                        right_sum = _mm256_add_epi16(mr_converted, tr_converted);
                        total_sum = _mm256_add_epi16(total_sum, right_sum);
                    } else {
                        div = 4;
                    }

                    if(j > 0){
                        __m128i ml_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j-1, dim)]);
                        __m128i tl_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j-1, dim)]);

                        __m256i ml_converted = _mm256_cvtepu8_epi16(ml_pixel);
                        __m256i tl_converted = _mm256_cvtepu8_epi16(tl_pixel);

                        left_sum = _mm256_add_epi16(ml_converted, tl_converted);
                        total_sum = _mm256_add_epi16(total_sum, left_sum);
                    } else {
                        div = 4;
                    }
                }

                else if(j == dim-1){
                   
                    __m128i ml_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j-1, dim)]);
                    __m128i tl_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j-1, dim)]);
                    __m128i bl_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j-1, dim)]);

                    __m128i tm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j, dim)]);
                    __m128i bm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j, dim)]);

                    __m256i tl_converted = _mm256_cvtepu8_epi16(tl_pixel);
                    __m256i tm_converted = _mm256_cvtepu8_epi16(tm_pixel);
                    __m256i ml_converted = _mm256_cvtepu8_epi16(ml_pixel);
                    __m256i bl_converted = _mm256_cvtepu8_epi16(bl_pixel);
                    __m256i bm_converted = _mm256_cvtepu8_epi16(bm_pixel);

                    total_sum = _mm256_add_epi16(mm_converted, ml_converted);
                    __m256i top_sum = _mm256_add_epi16(tm_converted, tl_converted);
                    total_sum = _mm256_add_epi16(total_sum, top_sum);
                    __m256i bottom_sum = _mm256_add_epi16(bm_converted, bl_converted);
                    total_sum = _mm256_add_epi16(total_sum, bottom_sum);

                }
                else if(j == 0){
        
                    __m128i tm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j, dim)]);
                    __m128i tr_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j+1, dim)]);
                    __m128i mr_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j+1, dim)]);
                    __m128i bm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j, dim)]);
                    __m128i br_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j+1, dim)]);


                    __m256i tm_converted = _mm256_cvtepu8_epi16(tm_pixel);
                    __m256i tr_converted = _mm256_cvtepu8_epi16(tr_pixel);
                    __m256i mr_converted = _mm256_cvtepu8_epi16(mr_pixel);
                    __m256i bm_converted = _mm256_cvtepu8_epi16(bm_pixel);
                    __m256i br_converted = _mm256_cvtepu8_epi16(br_pixel);

                    total_sum = _mm256_add_epi16(mm_converted, mr_converted);
                    __m256i top_sum = _mm256_add_epi16(tm_converted, tr_converted);
                    total_sum = _mm256_add_epi16(total_sum, top_sum);
                    __m256i bottom_sum = _mm256_add_epi16(bm_converted, br_converted);
                    total_sum = _mm256_add_epi16(total_sum, bottom_sum);
                }

                unsigned short pixel_elements[16];
                _mm256_storeu_si256((__m256i*) pixel_elements, total_sum);

                dst[RIDX(i,j, dim)].red = pixel_elements[0]/div;
                dst[RIDX(i,j, dim)].green = pixel_elements[1]/div;
                dst[RIDX(i,j, dim)].blue = pixel_elements[2]/div;
                dst[RIDX(i,j, dim)].alpha = pixel_elements[3]/div;

            }
        }
    }
}
/* 
 * smooth - Your current working version of smooth
 *          Our supplied version simply calls naive_smooth
 */

char avx_split_descr[] = "AVX Split";
void avx_split(int dim, pixel *src, pixel *dst){
    for(int i = 0; i < dim; i++){
        for(int j = 0; j < dim; j++){
            
           __m256i total_sum = _mm256_setzero_si256();
            
            if(i > 0 && i < dim-1 && j > 0 && j < dim-1){
                
                __m128i tl_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j-1, dim)]); //tl for j + 1, j + 2, j + 3
                __m128i tm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j, dim)]); //tm for j + 1, j + 2, j + 3
                __m128i tr_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j+1, dim)]); //tr for j + 1, j + 2, j + 3
                __m128i ml_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j-1, dim)]); //ml for j + 1, j + 2, j + 3
                __m128i mm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j, dim)]); //mm for j + 1, j + 2, j + 3
                __m128i mr_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j+1, dim)]); //mr for j + 1, j + 2, j + 3
                __m128i bl_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j-1, dim)]); //bl for j + 1, j + 2, j + 3
                __m128i bm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j, dim)]); //bm for j + 1, j + 2, j + 3
                __m128i br_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j+1, dim)]); //br for j + 1, j + 2, j + 3


                __m256i tl_converted = _mm256_cvtepu8_epi16(tl_pixel);
                __m256i tm_converted = _mm256_cvtepu8_epi16(tm_pixel);
                __m256i tr_converted = _mm256_cvtepu8_epi16(tr_pixel);
                __m256i ml_converted = _mm256_cvtepu8_epi16(ml_pixel);
                __m256i mm_converted = _mm256_cvtepu8_epi16(mm_pixel);
                __m256i mr_converted = _mm256_cvtepu8_epi16(mr_pixel);
                __m256i bl_converted = _mm256_cvtepu8_epi16(bl_pixel);
                __m256i bm_converted = _mm256_cvtepu8_epi16(bm_pixel);
                __m256i br_converted = _mm256_cvtepu8_epi16(br_pixel);

                
                __m256i top_sum = tl_converted;
                top_sum = _mm256_add_epi16(top_sum, tm_converted);
                top_sum = _mm256_add_epi16(top_sum, tr_converted);

                __m256i mid_sum = ml_converted;
                mid_sum = _mm256_add_epi16(mid_sum, mm_converted);
                mid_sum = _mm256_add_epi16(mid_sum, mr_converted);

                __m256i bot_sum = bl_converted;
                bot_sum = _mm256_add_epi16(bot_sum, bm_converted);
                bot_sum = _mm256_add_epi16(bot_sum, br_converted);

                total_sum = top_sum;
                total_sum = _mm256_add_epi16(total_sum, mid_sum);
                total_sum = _mm256_add_epi16(total_sum, bot_sum);

                int jAddr = 1;
                short t = 0;
                 //__m256i mask = _mm256_setr_epi16(-1,-1, -1, -1, -1, -1, -1, -1, 0, 0, 0 , 0 ,0 ,0 ,0, 0);
                if(j+4 < dim){
                    jAddr = 3;
                    t = 1;
                    //mask = _mm256_setr_epi16(-1,-1, -1, -1, -1, -1, -1, -1, -1,-1, -1, -1, -1, -1, -1, -1);
                } 

                unsigned short pixel_elements[16];
                //__m256i mask = __m256_setr_epi16(-1,-1, -1, -1, -1, -1, -1, -1);

                _mm256_storeu_si256((__m256i*) pixel_elements, total_sum);
                //_mm256_mask_storeu_epi16(&pixel_elements[0], mask, total_sum);


                dst[RIDX(i,j, dim)].red = (pixel_elements[0] * 7282) >> 16;
                dst[RIDX(i,j, dim)].green = (pixel_elements[1] * 7282) >> 16;
                dst[RIDX(i,j, dim)].blue = (pixel_elements[2] * 7282) >> 16;
                dst[RIDX(i,j, dim)].alpha = (pixel_elements[3] * 7282) >> 16;
                dst[RIDX(i,j+1, dim)].red = (pixel_elements[4] * 7282) >> 16;
                dst[RIDX(i,j+1, dim)].green = (pixel_elements[5] * 7282) >> 16;
                dst[RIDX(i,j+1, dim)].blue = (pixel_elements[6] * 7282) >> 16;
                dst[RIDX(i,j+1, dim)].alpha = (pixel_elements[7] * 7282) >> 16;
                if(t > 0){
                    dst[RIDX(i,j+2, dim)].red = (pixel_elements[8] * 7282) >> 16;
                    dst[RIDX(i,j+2, dim)].green = (pixel_elements[9] * 7282) >> 16;
                    dst[RIDX(i,j+2, dim)].blue = (pixel_elements[10] * 7282) >> 16;
                    dst[RIDX(i,j+2, dim)].alpha = (pixel_elements[11] * 7282) >> 16;
                    dst[RIDX(i,j+3, dim)].red = (pixel_elements[12] * 7282) >> 16;
                    dst[RIDX(i,j+3, dim)].green = (pixel_elements[13] * 7282) >> 16;
                    dst[RIDX(i,j+3, dim)].blue = (pixel_elements[14] * 7282) >> 16;
                    dst[RIDX(i,j+3, dim)].alpha = (pixel_elements[15] * 7282) >> 16;
                }
                

               j+=jAddr;
               
            } else {
                int div = 6;
                __m128i mm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j, dim)]);
                __m256i mm_converted = _mm256_cvtepu8_epi16(mm_pixel);

                if(i == 0){

                    __m128i bm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j, dim)]);
                    __m256i bm_converted = _mm256_cvtepu8_epi16(bm_pixel);

                    total_sum = _mm256_add_epi16(mm_converted, bm_converted);
                    __m256i right_sum = _mm256_setzero_si256();
                    __m256i left_sum = _mm256_setzero_si256();

                    if(j < dim-1){
                        __m128i mr_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j+1, dim)]);
                        __m128i br_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j+1, dim)]);

                        __m256i mr_converted = _mm256_cvtepu8_epi16(mr_pixel);
                        __m256i br_converted = _mm256_cvtepu8_epi16(br_pixel);

                        right_sum = _mm256_add_epi16(mr_converted, br_converted);
                        total_sum = _mm256_add_epi16(total_sum, right_sum);
                    } else {
                        div = 4;
                    }

                    if(j > 0){
                        __m128i ml_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j-1, dim)]);
                        __m128i bl_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j-1, dim)]);

                        __m256i ml_converted = _mm256_cvtepu8_epi16(ml_pixel);
                        __m256i bl_converted = _mm256_cvtepu8_epi16(bl_pixel);

                        left_sum = _mm256_add_epi16(ml_converted, bl_converted);
                        total_sum = _mm256_add_epi16(total_sum, left_sum);
                    } else {
                        div = 4;
                    }
                }
//###############################################################################################################
                else if(i == dim-1){
                    __m128i tm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j, dim)]);
                    __m256i tm_converted = _mm256_cvtepu8_epi16(tm_pixel);

                    total_sum = _mm256_add_epi16(mm_converted, tm_converted);
                    __m256i right_sum = _mm256_setzero_si256();
                    __m256i left_sum = _mm256_setzero_si256();

                    if(j < dim-1){
                        __m128i mr_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j+1, dim)]);
                        __m128i tr_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j+1, dim)]);

                        __m256i mr_converted = _mm256_cvtepu8_epi16(mr_pixel);
                        __m256i tr_converted = _mm256_cvtepu8_epi16(tr_pixel);

                        right_sum = _mm256_add_epi16(mr_converted, tr_converted);
                        total_sum = _mm256_add_epi16(total_sum, right_sum);
                    } else {
                        div = 4;
                    }

                    if(j > 0){
                        __m128i ml_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j-1, dim)]);
                        __m128i tl_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j-1, dim)]);

                        __m256i ml_converted = _mm256_cvtepu8_epi16(ml_pixel);
                        __m256i tl_converted = _mm256_cvtepu8_epi16(tl_pixel);

                        left_sum = _mm256_add_epi16(ml_converted, tl_converted);
                        total_sum = _mm256_add_epi16(total_sum, left_sum);
                    } else {
                        div = 4;
                    }
                }

                else if(j == dim-1){
                   
                    __m128i ml_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j-1, dim)]);
                    __m128i tl_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j-1, dim)]);
                    __m128i bl_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j-1, dim)]);

                    __m128i tm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j, dim)]);
                    __m128i bm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j, dim)]);

                    __m256i tl_converted = _mm256_cvtepu8_epi16(tl_pixel);
                    __m256i tm_converted = _mm256_cvtepu8_epi16(tm_pixel);
                    __m256i ml_converted = _mm256_cvtepu8_epi16(ml_pixel);
                    __m256i bl_converted = _mm256_cvtepu8_epi16(bl_pixel);
                    __m256i bm_converted = _mm256_cvtepu8_epi16(bm_pixel);

                    total_sum = _mm256_add_epi16(mm_converted, ml_converted);
                    __m256i top_sum = _mm256_add_epi16(tm_converted, tl_converted);
                    total_sum = _mm256_add_epi16(total_sum, top_sum);
                    __m256i bottom_sum = _mm256_add_epi16(bm_converted, bl_converted);
                    total_sum = _mm256_add_epi16(total_sum, bottom_sum);

                }
                else if(j == 0){
        
                    __m128i tm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j, dim)]);
                    __m128i tr_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i-1, j+1, dim)]);
                    __m128i mr_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i, j+1, dim)]);
                    __m128i bm_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j, dim)]);
                    __m128i br_pixel = _mm_loadu_si128((__m128i*) &src[RIDX(i+1, j+1, dim)]);


                    __m256i tm_converted = _mm256_cvtepu8_epi16(tm_pixel);
                    __m256i tr_converted = _mm256_cvtepu8_epi16(tr_pixel);
                    __m256i mr_converted = _mm256_cvtepu8_epi16(mr_pixel);
                    __m256i bm_converted = _mm256_cvtepu8_epi16(bm_pixel);
                    __m256i br_converted = _mm256_cvtepu8_epi16(br_pixel);

                    total_sum = _mm256_add_epi16(mm_converted, mr_converted);
                    __m256i top_sum = _mm256_add_epi16(tm_converted, tr_converted);
                    total_sum = _mm256_add_epi16(total_sum, top_sum);
                    __m256i bottom_sum = _mm256_add_epi16(bm_converted, br_converted);
                    total_sum = _mm256_add_epi16(total_sum, bottom_sum);
                }

                unsigned short pixel_elements[16];
                _mm256_storeu_si256((__m256i*) pixel_elements, total_sum);

                dst[RIDX(i,j, dim)].red = pixel_elements[0]/div;
                dst[RIDX(i,j, dim)].green = pixel_elements[1]/div;
                dst[RIDX(i,j, dim)].blue = pixel_elements[2]/div;
                dst[RIDX(i,j, dim)].alpha = pixel_elements[3]/div;

            }
        }
    }
}

char another_smooth_descr[] = "Unrolling outer loop";
void another_smooth(int dim, pixel *src, pixel *dst) 
{
    int SKIP = 4;
    int i = 0;
    for(; i+SKIP < dim; i+=SKIP){
        for(int j = 0; j < dim; j++){
            dst[RIDX(i,j, dim)] = avg(dim, i, j, src);
            dst[RIDX(i+1,j, dim)] = avg(dim, i+1, j, src);
            dst[RIDX(i+2,j, dim)] = avg(dim, i+2, j, src);
            dst[RIDX(i+3,j, dim)] = avg(dim, i+3, j, src);
        }
    }
    for(; i < dim; i++){
        for(int j = 0; j < dim; j++){
            dst[RIDX(i,j, dim)] = avg(dim, i, j, src);
        }
    }
}

char inner_smooth_descr[] = "Unrolling inner loop";
void inner_smooth(int dim, pixel *src, pixel *dst) 
{
    int SKIP = 4;
    
    for(int i = 0; i < dim; i++){
        int j = 0;
        for(; j + SKIP < dim; j+=SKIP){
            dst[RIDX(i,j, dim)] = avg(dim, i, j, src);
            dst[RIDX(i,j+1, dim)] = avg(dim, i, j+1, src);
            dst[RIDX(i,j+2, dim)] = avg(dim, i, j+2, src);
            dst[RIDX(i,j+3, dim)] = avg(dim, i, j+3, src);
        }
        for(; j < dim; j++){
            dst[RIDX(i,j, dim)] = avg(dim, i, j, src);
        }
    }
}

char inner_smooth_na_descr[] = "Unrolling inner loop with new avg function";
void inner_smooth_na(int dim, pixel *src, pixel *dst) 
{
    int SKIP = 4;
    
    for(int i = 0; i < dim; i++){
        int j = 0;
        for(; j + SKIP < dim; j+=SKIP){
            dst[RIDX(i,j, dim)] = new_avg(dim, i, j, src);
            dst[RIDX(i,j+1, dim)] = new_avg(dim, i, j+1, src);
            dst[RIDX(i,j+2, dim)] = new_avg(dim, i, j+2, src);
            dst[RIDX(i,j+3, dim)] = new_avg(dim, i, j+3, src);
        }
        for(; j < dim; j++){
            dst[RIDX(i,j, dim)] = new_avg(dim, i, j, src);
        }
    }
}

/*********************************************************************
 * register_smooth_functions - Register all of your different versions
 *     of the smooth function by calling the add_smooth_function() for
 *     each test function. When you run the benchmark program, it will
 *     test and report the performance of each registered test
 *     function.  
 *********************************************************************/

void register_smooth_functions() {
    add_smooth_function(&naive_smooth, naive_smooth_descr);
    add_smooth_function(&naive_smooth_na, naive_smooth_na_descr);
    add_smooth_function(&avx_smooth, avx_smooth_descr);
    add_smooth_function(&avx_split, avx_split_descr);
    //add_smooth_function(&another_smooth, another_smooth_descr);
    //add_smooth_function(&inner_smooth, inner_smooth_descr);
    //add_smooth_function(&inner_smooth_na, inner_smooth_na_descr);
}
