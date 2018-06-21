#include <bits/stdc++.h>
#define frames 300

#ifdef CIF
#define height 288
#define width 352
#endif

#ifdef QCIF
#define height 144
#define width 176
#endif

unsigned char Y[frames][height][width] = {{{0}}};
unsigned char Pred_Y[frames][height][width] = {{{0}}};

double SAD(int f, int x, int y, int a, int b, int blk_size){
    double sad = 0;
    for ( int i = 0; i < blk_size; i++ ) 
        for ( int j = 0; j < blk_size; j++ ) 
            sad += abs(Y[f][x+i][y+j] - Y[f-1][a+i][b+j]);
    sad /= pow(blk_size, 2);
    return sad;
}

int PSNR(int f){
    double mse = 0;
    for ( int i = 0; i < height; i++ )
        for ( int j = 0; j < width; j++ )
            mse += pow(Y[f][i][j] - Pred_Y[f][i][j], 2);
    mse /= (height*width);
    return round(10 * log10( 255*255 / mse ));
}

void full_search(int f, int region, int blk_size){
    for ( int x = 0; x < height; x+=blk_size ) {
        for ( int y = 0; y < width; y+=blk_size ) {
            double min = 1e5;
            int min_x, min_y;
            for ( int i = x-region; i < x+region; i++ ) 
                for ( int j = y-region; j < y+region; j++ ){
                    if ( i < 0 || j < 0 || i+blk_size >= height || j+blk_size >= width )   continue;
                    double tmp = SAD(f, x, y, i, j, blk_size);
                    if (tmp < min){
                        min = tmp;
                        min_x = i;
                        min_y = j;
                    }
                }
            for ( int i = 0; i < blk_size; i++ )
                for ( int j = 0; j < blk_size; j++ )
                    Pred_Y[f][x+i][y+j] = Y[f-1][min_x+i][min_y+j];
        }
    }
}

void three_step_search(int f, int step, int blk_size){
    for ( int x = 0; x < height; x+=blk_size ) {
        for ( int y = 0; y < width; y+=blk_size ) {
            int tmp_step = step;
            int min_x, min_y;
            int base_x = x, base_y = y;
            while (tmp_step) {
                double min = 1e5;
                for ( int i = base_x-tmp_step; i <= base_x+tmp_step; i+=tmp_step ) 
                    for ( int j = base_y-tmp_step; j <= base_y+tmp_step; j+=tmp_step ){
                        if ( i < 0 || j < 0 || i+blk_size >= height || j+blk_size >= width)   continue;
                        double tmp = SAD(f, x, y, i, j, blk_size);
                        if (tmp < min){
                            min = tmp;
                            min_x = i;
                            min_y = j;
                        }
                    }
                if (min_x == base_x && min_y == base_y) break;
                tmp_step /= 2;
                base_x = min_x, base_y = min_y;
            }
            for ( int i = 0; i < blk_size; i++ )
                for ( int j = 0; j < blk_size; j++ )
                    Pred_Y[f][x+i][y+j] = Y[f-1][min_x+i][min_y+j];
        }
    }
}

void new_three_step_search(int f, int step, int blk_size){
    for ( int x = 0; x < height; x+=blk_size ) {
        for ( int y = 0; y < width; y+=blk_size ) {
            int tmp_step = step;
            int min_x, min_y;
            int base_x = x, base_y = y;
            while (tmp_step) {
                double min = 1e5;
                for ( int i = base_x-tmp_step; i <= base_x+tmp_step; i+=tmp_step ) 
                    for ( int j = base_y-tmp_step; j <= base_y+tmp_step; j+=tmp_step ){
                        if ( i < 0 || j < 0 || i+blk_size >= height || j+blk_size >= width)   continue;
                        double tmp = SAD(f, x, y, i, j, blk_size);
                        if (tmp < min){
                            min = tmp;
                            min_x = i;
                            min_y = j;
                        }
                    }
                int times = 2;
                while (times--) {
                    for ( int i = base_x-1; i <= base_x+1; i++ )
                        for ( int j = base_y-1; j <= base_y+1; j++ ) {
                            if ( i < 0 || j < 0 || i+blk_size >= height || j+blk_size >= width)   continue;
                            double tmp = SAD(f, x, y, i, j, blk_size);
                            if (tmp < min){
                                min = tmp;
                                min_x = i;
                                min_y = j;
                            }
                        }
                    if (min_x == base_x && min_y == base_y) break;
                    if (abs(min_x - base_x) > 1 || abs(min_y - base_y) > 1) break;
                    base_x = min_x, base_y = min_y;
                }
                if (min_x == base_x && min_y == base_y) break;
                tmp_step /= 2;
                base_x = min_x, base_y = min_y;
            }
            for ( int i = 0; i < blk_size; i++ )
                for ( int j = 0; j < blk_size; j++ )
                    Pred_Y[f][x+i][y+j] = Y[f-1][min_x+i][min_y+j];
        }
    }
}

void _2D_Log_search(int f, int step, int blk_size){
    for ( int x = 0; x < height; x+=blk_size ) {
        for ( int y = 0; y < width; y+=blk_size ) {
            int tmp_step = step;
            int min_x, min_y;
            int base_x = x, base_y = y;
            while (tmp_step) {
                double min = 1e5;
                for ( int i = base_x-tmp_step; i <= base_x+tmp_step; i+=tmp_step ) 
                    for ( int j = base_y-tmp_step; j <= base_y+tmp_step; j+=tmp_step ){
                        if ( i < 0 || j < 0 || i+blk_size >= height || j+blk_size >= width)   continue;
                        if ( i != base_x && j != base_y ) continue;
                        double tmp = SAD(f, x, y, i, j, blk_size);
                        if (tmp < min){
                            min = tmp;
                            min_x = i;
                            min_y = j;
                        }
                    }
                if (min_x == base_x && min_y == base_y) {
                    tmp_step /= 2;
                }
                base_x = min_x, base_y = min_y;
            }
            for ( int i = 0; i < blk_size; i++ )
                for ( int j = 0; j < blk_size; j++ )
                    Pred_Y[f][x+i][y+j] = Y[f-1][min_x+i][min_y+j];
        }
    }
}

void orthogonal_search(int f, int step, int blk_size){
    for ( int x = 0; x < height; x+=blk_size ) {
        for ( int y = 0; y < width; y+=blk_size ) {
            int tmp_step = step;
            int min_x, min_y;
            int base_x = x, base_y = y;
            while (tmp_step) {
                double min = 1e5;
                for ( int i = base_x-tmp_step; i <= base_x+tmp_step; i+=tmp_step ) {
                    if ( i < 0 || i+blk_size >= height )   continue;
                    double tmp = SAD(f, x, y, i, base_y, blk_size);
                    if (tmp < min){
                        min = tmp;
                        min_x = i;
                        min_y = base_y;
                    }
                }
                base_x = min_x, base_y = min_y;
                for ( int j = base_y-tmp_step; j <= base_y+tmp_step; j+=tmp_step ){
                    if ( j < 0 || j+blk_size >= width)   continue;
                    double tmp = SAD(f, x, y, base_x, j, blk_size);
                    if (tmp < min){
                        min = tmp;
                        min_x = base_x;
                        min_y = j;
                    }
                }
                tmp_step /= 2;
                base_x = min_x, base_y = min_y;
            }
            for ( int i = 0; i < blk_size; i++ )
                for ( int j = 0; j < blk_size; j++ )
                    Pred_Y[f][x+i][y+j] = Y[f-1][min_x+i][min_y+j];
        }
    }
}

int main(int argc, char* argv[]) {
    printf("height: %d width: %d\n", height, width);
	FILE *fp = fopen(argv[1], "r");
	if (!fp){
		printf("can't open file !\n");
		exit(1);
	}
    for ( int i = 0; i < frames; i++ ){
        fread(Y[i], sizeof(unsigned char), height*width, fp);
        fseek(fp, height*width/2, SEEK_CUR);
    }
    int blk_size = 8;
    int full_psnr[frames] = {0};
    int TSS_psnr[frames] = {0};
    int NTSS_psnr[frames] = {0};
    int _2D_Log_psnr[frames] = {0};
    int orthogonal_psnr[frames] = {0};
    for ( int i = 1; i < frames; i++ ){
        full_search(i, 7, blk_size);
        full_psnr[i] = PSNR(i);
        three_step_search(i, 16, blk_size);
        TSS_psnr[i] = PSNR(i);
        new_three_step_search(i, 16, blk_size);
        NTSS_psnr[i] = PSNR(i);
        _2D_Log_search(i, 16, blk_size);
        _2D_Log_psnr[i] = PSNR(i);
        orthogonal_search(i, 16, blk_size);
        orthogonal_psnr[i] = PSNR(i);
    }
    
    FILE *output = fopen(argv[2], "w");
    fprintf(output, "%s", "FULL:\n");
    for ( int i = 0; i < 10; i++ )
        for ( int j = 0; j < 30; j++ )
            fprintf(output, "%d%s", full_psnr[i*30+j], (j==29)?"\n":" ");
    fprintf(output, "%s", "TSS:\n");
    for ( int i = 0; i < 10; i++ )
        for ( int j = 0; j < 30; j++ )
            fprintf(output, "%d%s", TSS_psnr[i*30+j], (j==29)?"\n":" ");
    fprintf(output, "%s", "NTSS:\n");
    for ( int i = 0; i < 10; i++ )
        for ( int j = 0; j < 30; j++ )
            fprintf(output, "%d%s", NTSS_psnr[i*30+j], (j==29)?"\n":" ");
    fprintf(output, "%s", "2D_Log:\n");
    for ( int i = 0; i < 10; i++ )
        for ( int j = 0; j < 30; j++ )
            fprintf(output, "%d%s", _2D_Log_psnr[i*30+j], (j==29)?"\n":" ");
    fprintf(output, "%s", "Orthogonal:\n");
    for ( int i = 0; i < 10; i++ )
        for ( int j = 0; j < 30; j++ )
            fprintf(output, "%d%s", orthogonal_psnr[i*30+j], (j==29)?"\n":" ");
		
	return 0;
}
