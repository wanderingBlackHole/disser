#include "filteredobject.h"
#include <cmath>
#include <QPixmap>
#include <QLabel>

#include <iostream>

#include "pathopenclose.h"

#include <QRandomGenerator>

FilteredObject::FilteredObject()
{
}
void FilteredObject::dbgForm(QString name, Mat image, QString dbgSave)
{
    QWidget *form = new QWidget();
    form->setFixedSize(400,400);
    imwrite(dbgSave.toStdString(), image);
    QPixmap pix1(dbgSave);
    QLabel *label = new QLabel(form);
    label->setPixmap(pix1);
    label->setPixmap(pix1.scaled(400,400,Qt::KeepAspectRatio));
    form->setWindowTitle(name);
    form->show();
}
float FilteredObject::SEffective()
{

    float seffpercent = 0.0f;
    int pixs = this->m_usefulPixs;
    qDebug()<<"pixs = "<<pixs;

    qDebug() << "rows = " << this->m_fObjectMatrixSrc.rows << "cols = " << this->m_fObjectMatrixSrc.cols;
    seffpercent = (float)pixs / (this->m_fObjectMatrixSrc.rows*this->m_fObjectMatrixSrc.cols) * 100;

    qDebug() <<"seffpercent = " <<seffpercent;

    this->m_SEff = this->m_Smkm * seffpercent;

    return this->m_SEff;
}
void FilteredObject::contrast(Mat imageToApply, double alpha, int beta)
{   
   imageToApply.convertTo(imageToApply, -1, alpha, beta);
}

/*
 * Эта функция использует фильтр Гаусса и медианный фильтр для сглаживания изображения и уменьшения шума
 * а также фильтр Лапласа для повышения четкости изображения
*/
void FilteredObject::preprocessing()
{
    // посчитаем число пикселей, которые занимает полезная информация, и сохраним в поле класса
    Mat usefulPixels;
    this->m_fObjectMatrixSrc.copyTo(usefulPixels);
    threshold(usefulPixels, usefulPixels, 0, 255, THRESH_BINARY);
    this->dbgForm("Useful area", usefulPixels, "/home/daria/wwwm/useful_pixs.jpg");

    int countPixs = 0;
    for(int i = 0; i < usefulPixels.rows; i++)
    {
        for(int j = 0; j < usefulPixels.cols; j++)
        {
            if ( (usefulPixels.at<cv::Vec3b>(i,j)[0] == 255) && (usefulPixels.at<cv::Vec3b>(i,j)[1] == 255) && (usefulPixels.at<cv::Vec3b>(i,j)[2] == 255))
                countPixs += 1;
        }
    }
    qDebug() << "number of useful pixels = " << countPixs;
    this->m_usefulPixs = countPixs;
    /*-------------------------------------------------------------------------------------------------*/

    this->contrast(this->m_fObjectMatrixSrc, 5.0, 5);
    this->dbgForm("Source Contrast", this->m_fObjectMatrixSrc, "/home/daria/wwwm/source_contrast.jpg");

    GaussianBlur(this->m_fObjectMatrixSrc, this->m_fObjectMatrixSrc, Size( 7, 7 ), 0, 0 );
    this->dbgForm("Gaussian blur", this->m_fObjectMatrixSrc, "/home/daria/wwwm/gau_blur_source.jpg");

    Mat kernel = (Mat_<float>(3,3) <<
                       1,  1, 1,
                       1, -8, 1,
                       1,  1, 1);

         Mat imgLaplacian;
         filter2D(this->m_fObjectMatrixSrc, imgLaplacian, CV_32F, kernel);
         Mat sharp;
         this->m_fObjectMatrixSrc.convertTo(sharp, CV_32F);
         Mat imgResult = sharp - imgLaplacian;
         this->m_fObjectMatrixSrc = imgResult;
         this->dbgForm("Laplassian sharp", this->m_fObjectMatrixSrc, "/home/daria/wwwm/lapl_sharp_source.jpg");
         medianBlur(this->m_fObjectMatrixSrc,this->m_fObjectMatrixSrc, 3);
         this->dbgForm("Median blur", this->m_fObjectMatrixSrc, "/home/daria/wwwm/median_blur_source.jpg");
          this->m_fObjectMatrixSrc.copyTo(this->m_fObjectMatrixTmp);
}
void FilteredObject::postprocessing()
{
    this->contrast(this->m_fObjectMatrixDst, 500.0, 5);
     this->dbgForm("Frangi contrast", this->m_fObjectMatrixDst, "/home/daria/wwwm/frangi_contrast.jpg");

    Mat kernel = (Mat_<float>(3,3) <<
                       1,  1, 1,
                       1, -8, 1,
                       1,  1, 1);

         Mat imgLaplacianDst;
         filter2D(this->m_fObjectMatrixDst, imgLaplacianDst, CV_32F, kernel);
         Mat sharp;
         this->m_fObjectMatrixDst.convertTo(sharp, CV_32F);
         Mat imgResult = sharp - imgLaplacianDst;
         this->m_fObjectMatrixDst = imgResult;
         this->dbgForm("Laplassian sharp output", this->m_fObjectMatrixDst, "/home/daria/wwwm/lapl_sharp_out.jpg");




      //cvtColor( this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, COLOR_RGB2GRAY );

//     //GaussianBlur(this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, Size( 13, 13 ), 0, 0 );
//     this->m_fObjectMatrixDst.convertTo(this->m_fObjectMatrixDst, CV_8UC3);
//     Canny(this->m_fObjectMatrixDst,this->m_fObjectMatrixDst,20,50);
//     this->dbgForm("Canny", this->m_fObjectMatrixDst, "/home/daria/wwwm/dbg6.jpg");

     //Бинаризация
     double thresh = 50;
     double maxValue = 255;
     threshold(this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, thresh, maxValue, THRESH_BINARY);
     this->dbgForm("Frangi binary", this->m_fObjectMatrixDst, "/home/daria/wwwm/frangi_binary.jpg");

//     medianBlur(this->m_fObjectMatrixDst,this->m_fObjectMatrixDst, 3);
//     this->dbgForm("Median blur2", this->m_fObjectMatrixDst, "/home/daria/wwwm/dbg7.jpg");


    this->m_fObjectMatrixDst.copyTo(this->m_fObjectMatrixSearchForFibers);

}
/*
 *
*/
void FilteredObject::hessian(float sigma)
{
    Mat srcImage = this->m_fObjectMatrixSrc;
    int kernel_x = 2*round(3*sigma) + 1;
    int kernel_y = kernel_x;
    float *xx = new float[kernel_x * kernel_y]();
    float *xy = new float[kernel_x * kernel_y]();
    float *yy = new float[kernel_x * kernel_y]();
int i=0, j=0;
    for (int x = -round(3*sigma); x <= round(3*sigma); x++)
    {
        j=0;
        for (int y = -round(3*sigma); y <= round(3*sigma); y++){
            xx[i*kernel_y + j] = 1.0f/(2.0f*M_PI*pow(sigma,4)) * (x*x/pow(sigma,2) - 1) * exp(-(x*x + y*y)/(2.0f*pow(sigma,2)));
            xy[i*kernel_y + j] = 1.0f/(2.0f*M_PI*pow(sigma,6))*(x*y)*exp(-(x*x + y*y)/(2.0f*pow(sigma,2)));
            j++;
        }
        i++;
    }
    for (int j=0; j < kernel_y; j++){
        for (int i=0; i < kernel_x; i++){
            yy[j*kernel_x + i] = xx[i*kernel_x + j];
        }
    }
//    flip kernels since kernels aren't symmetric and opencv's filter2D operation performs a correlation, not a convolution
        Mat kern_xx;
        flip(Mat(kernel_y, kernel_x, CV_32FC1, xx), kern_xx, -1);

        Mat kern_xy;
        flip(Mat(kernel_y, kernel_x, CV_32FC1, xy), kern_xy, -1);

        Mat kern_yy;
        flip(Mat(kernel_y, kernel_x, CV_32FC1, yy), kern_yy, -1);

    //specify anchor since we are to perform a convolution, not a correlation
    Point anchor(kernel_x - kernel_x/2 - 1, kernel_y - kernel_y/2 - 1);
    filter2D(this->m_fObjectMatrixSrc, this->m_Dxx, -1, kern_xx, anchor);
    filter2D(this->m_fObjectMatrixSrc, this->m_Dxy, -1, kern_xy, anchor);
    filter2D(this->m_fObjectMatrixSrc, this->m_Dyy, -1, kern_yy, anchor);

//this->m_fObjectMatrixDst = this->m_Dxx*this->m_Dyy - (0.9*this->m_Dxy)*(0.9*this->m_Dxy);
    //filter2D(this->m_fObjectMatrixDst, this->m_Dxx*this->m_Dyy - (0.9*this->m_Dxy)*(0.9*this->m_Dxy), -1, )

    delete[] xx;
    delete[] xy;
    delete[] yy;

}

/*
 * Эта функция вычисляет "похожесть" региона изображения на коллагеновое волокно,
 * согласно методу, описанному Франжи.
 * Функция использует собственные векторы гессиана.
 */
void FilteredObject::frangiFilter()
{
    // коэффициенты Франжи
//    float beta = 0.5f;
float beta = 15.0f;
    float c = 450.0f;
    for (int sigma = 1; sigma <= 11; sigma += 1)
    {
        this->hessian(sigma);
        this->m_Dxx = this->m_Dxx*sigma*sigma;
        this->m_Dyy = this->m_Dyy*sigma*sigma;
        this->m_Dxy = this->m_Dxy*sigma*sigma;
        Mat tmp = (this->m_Dxx - this->m_Dyy).mul(this->m_Dxx - this->m_Dyy) + 4*((this->m_Dxy).mul(this->m_Dxy));
        cv::sqrt(tmp,tmp);
        // собственные векторы гессиана
        Mat v2x = 2*this->m_Dxy;
        Mat v2y = this->m_Dyy - this->m_Dxx + tmp;

        // нормализуем
        Mat norm;
        sqrt((v2x.mul(v2x) + v2y.mul(v2y)), norm);
        Mat v2xtmp = v2x.mul(1.0f/norm);
        v2xtmp.copyTo(v2x, norm != 0);
        Mat v2ytmp = v2y.mul(1.0f/norm);
        v2ytmp.copyTo(v2y, norm != 0);

        // собственные векторы ортогональны
        Mat v1x, v1y;
        v2y.copyTo(v1x);
        v1x = -1*v1x;
        v2x.copyTo(v1y);

        // собственные значения гессиана
        Mat lambda1 = 0.5*( this->m_Dxx + this->m_Dyy + tmp );
        Mat lambda2 = 0.5*( this->m_Dxx + this->m_Dyy - tmp );

        // отсортируем собственные значения по модулю abs(lambda1) < abs(lamda2)
        Mat check = abs(lambda1) > abs(lambda2);
        lambda1.copyTo(this->m_lambda1); lambda2.copyTo(this->m_lambda1, check);
        lambda2.copyTo(this->m_lambda2); lambda1.copyTo(this->m_lambda2, check);

        v1x.copyTo(this->m_vx); v2x.copyTo(this->m_vx, check);
        v1y.copyTo(this->m_vy); v2y.copyTo(this->m_vy, check);

        // вычисляем значение наименьшего собственного вектора
        Mat angles;
        phase(this->m_vx, this->m_vy, angles);
        this->m_angles.push_back(angles);

        // коэффициенты
        Mat Rb = this->m_lambda1.mul(1/this->m_lambda2);
        Mat Stmp = this->m_lambda1.mul(this->m_lambda1) + this->m_lambda2.mul(this->m_lambda2);
        Mat S;
        sqrt(Stmp, S);


        // Выходное изображение
        Mat Rbtmp1, Stmp2;
        exp(-Rb/beta, Rbtmp1);
        exp(-S/c, Stmp2);


        Mat Ifiltered = Rbtmp1.mul(Mat::ones(this->m_fObjectMatrixSrc.rows, this->m_fObjectMatrixSrc.cols, this->m_fObjectMatrixSrc.type()) - Stmp2);
        Ifiltered.setTo(0, this->m_lambda2 > 0);

        // складируем
        this->m_filtered.push_back(Ifiltered);
    }

        float sigma = 1;
        this->m_filtered[0].copyTo(this->m_fObjectMatrixDst);
//        this->m_filtered[0].copyTo(whatScale);
//        this->m_filtered[0].copyTo(outAngles);
//        whatScale.setTo(sigma);

        //find element-wise maximum across all accumulated filter results
        for (int i=1; i < this->m_filtered.size(); i++){
            this->m_fObjectMatrixDst = max(this->m_fObjectMatrixDst, this->m_filtered[i]);
//            whatScale.setTo(sigma, this->m_filtered[i] == maxVals);
//            this->m_filtered[i].copyTo(outAngles, this->m_filtered[i] == maxVals);
            sigma += 1;
        }

        this->dbgForm("Frangi", this->m_fObjectMatrixDst, "/home/daria/wwwm/frangi.jpg");

}

void FilteredObject::erodeDiliate()
{
    int erosion_size = 1;
    Mat element = getStructuringElement( MORPH_RECT,
                         Size( erosion_size+1, erosion_size+1 ),
                         Point( erosion_size, erosion_size ) );
    erode( this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, element);
    dilate( this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, element);
    erode( this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, element);
    //erode( this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, Mat());

    Mat element1 = getStructuringElement( MORPH_RECT,
                         Size( erosion_size+3, erosion_size+3 ),
                         Point( erosion_size, erosion_size ) );
    morphologyEx(this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, MORPH_CLOSE ,element1);
    erode( this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, element);
    this->dbgForm("Erode/dilate ", this->m_fObjectMatrixDst, "/home/daria/wwwm/ede.jpg");

}
void FilteredObject::erodeDiliate1()
{
    int erosion_size = 1;
    Mat element = getStructuringElement( MORPH_RECT,
                         Size( erosion_size+2, erosion_size+6 ),
                         Point( erosion_size, erosion_size ) );

    Mat element1 = getStructuringElement( MORPH_RECT,
                         Size( erosion_size+3, erosion_size+3 ),
                         Point( erosion_size, erosion_size ) );

    morphologyEx(this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, MORPH_CLOSE ,element);
    this->dbgForm("CLOSE", this->m_fObjectMatrixDst, "/home/daria/wwwm/close.jpg");

    morphologyEx(this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, MORPH_OPEN ,element1);
    this->dbgForm("OPEN", this->m_fObjectMatrixDst, "/home/daria/wwwm/open.jpg");

}
void FilteredObject::dstOverlaid()
{
    this->visualizeFibers();
    this->m_fObjectMatrixTmp.convertTo(this->m_fObjectMatrixTmp, CV_8UC3);
    cvtColor(this->m_fObjectMatrixTmp, this->m_fObjectMatrixTmp, COLOR_GRAY2RGB);
    //this->m_fObjectMatrixDst = this->m_fObjectMatrixDst + this->m_fObjectMatrixTmp;

    addWeighted (this->m_fObjectMatrixTmp, 0.3, this->m_fObjectMatrixDst, 0.7, 0.0, this->m_fObjectMatrixDst);
}

void FilteredObject::path_opening()
{
//    unsigned char   *input_image;         /* The input image */
//    int             nx, ny;               /* Image dimensions */
//    int             L;                    /* The threshold line length */
//    int             K;                    /* The maximum number of gaps in the path */
//    unsigned char   *output_image ;       /* Output image */




    //pathopen(this->m_fObjectURLCharSrc,nx,ny,L,K,output_image);

    /*--------------------------------------------------------------------------------------------------------------*/


    const char* currentPath = "/home/daria/wwwm";
    const char* currentFileName = "/home/daria/wwwm/frangi_binary.jpg";
    const char* outputFileName = "/home/daria/wwwm/frangi_binaryPATH.jpg";


    /* Open an image from file */
        BIMAGE * input_bimage = read_grayscale_image(currentPath, currentFileName);
        // Allocate remaining images
        BIMAGE * output_bimage = BIMAGE_constructor(input_bimage->dim);
        int nx = input_bimage->dim->buf[0];
        int ny = input_bimage->dim->buf[1];
            int num_pixels = nx*ny;
        PATHOPEN_PIX_TYPE * input_image = new PATHOPEN_PIX_TYPE[nx * ny];
        PATHOPEN_PIX_TYPE * output_image = new PATHOPEN_PIX_TYPE[nx * ny];

    // Convert intermediate float to PATHOPEN_PIX_TYPE (unsigned char)
    for (int i = 0; i < num_pixels; ++i) {
        input_image[i] = static_cast<PATHOPEN_PIX_TYPE>(input_bimage->buf[i]);
    }

    int L = 27;
    int K = 0;

   pathopen(input_image, nx, ny, L, K, output_image);

    for (int i = 0; i < num_pixels; ++i) {
                output_bimage->buf[i] = static_cast<PATHOPEN_PIX_TYPE>(output_image[i]);
        }
        // Write file
        write_grayscale_image(
            output_bimage,
            outputFileName
        );

        this->m_fObjectPathURL = outputFileName;

//        // Deallocate
//        delete[] input_image;
//        delete[] output_image;
//        BIMAGE_destructor(input_bimage);
//        BIMAGE_destructor(output_bimage);

    //this->dbgForm("HUGUE OPENING PATH", this->m_pathOutput, "/home/daria/wwwm/path_output.jpg");
    /*--------------------------------------------------------------------------------------------------------------*/

}

void FilteredObject::startPoints()
{
    Mat mask;
    Mat element = getStructuringElement( MORPH_RECT,
                         Size( 9, 9 )
                          );
    // начальные точки определяются как максимумы яркости функции расстояния
    cv::dilate(this->m_fObjectMatrixDst, mask, element);
    cv::compare(this->m_fObjectMatrixDst, mask, mask, cv::CMP_GE);

    cv::Mat non_plateau_mask;
           cv::erode(this->m_fObjectMatrixDst, non_plateau_mask, element);
           cv::compare(this->m_fObjectMatrixDst, non_plateau_mask, non_plateau_mask, cv::CMP_GT);
           cv::bitwise_and(mask, non_plateau_mask, mask);

           Scalar color(0,0,255);
       //convert grayscale to color image
          cvtColor(mask, mask, COLOR_GRAY2RGB);
          for(int i = 0; i < mask.rows; i++)
          {
              for(int j = 0; j < mask.cols; j++)
              {
                  if((mask.at<cv::Vec3b>(i,j)[0] != 0) && (mask.at<cv::Vec3b>(i,j)[1] != 0) && (mask.at<cv::Vec3b>(i,j)[2] != 0))
                  {
//                      mask.at<cv::Vec3b>(i,j)[0] = 0;
//                      mask.at<cv::Vec3b>(i,j)[1] = 0;
//                      mask.at<cv::Vec3b>(i,j)[2] = 255;

                      this->m_startP.push_back({i,j});
                  }

              }
          }

          this->m_fObjectMatrixDst.convertTo(this->m_fObjectMatrixDst, CV_8UC3);
          cvtColor(this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, COLOR_GRAY2RGB);
          this->contrast(this->m_fObjectMatrixDst, 255,5);
          this->contrast(mask, 255,5);

          mask.convertTo(mask, CV_8UC3);

          addWeighted (mask, 0.8, this->m_fObjectMatrixDst, 0.2, 0.0, this->m_fObjectMatrixDst);

    this->dbgForm("Start points", mask, "/home/daria/wwwm/start_points.jpg");

}

void FilteredObject::distance()
{

//    this->m_fObjectMatrixDst.convertTo(this->m_fObjectMatrixDst, CV_8UC1);
    distanceTransform(this->m_pathOutput, this->m_pathOutput, DIST_L2, CV_32F);
        this->m_pathOutput.convertTo(this->m_pathOutput, -1, 5,5);
    this->dbgForm("Distance ", this->m_pathOutput, "/home/daria/wwwm/dist.jpg");
//    GaussianBlur(this->m_pathOutput, this->m_pathOutput, Size( 7, 7 ), 0, 0 );
//    this->contrast(this->m_pathOutput, 255,5);
//    this->dbgForm("Gaussian blur after distance", this->m_pathOutput, "/home/daria/wwwm/gau_blur_dist.jpg");

    //Бинаризация
    double thresh = 15 ;
    double maxValue = 255;
    threshold(this->m_pathOutput, this->m_pathOutput, thresh, maxValue, THRESH_BINARY);
   this->dbgForm("PATH after distance binary", this->m_pathOutput, "/home/daria/wwwm/path_after_distance_binary.jpg");

//    int erosion_size = 1;

//    Mat element = getStructuringElement( MORPH_RECT,
//                         Size( erosion_size, erosion_size+5 )
//                         );



//    morphologyEx(this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, MORPH_CLOSE ,element);
//    morphologyEx(this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, MORPH_CLOSE ,element);
//    Mat element1 = getStructuringElement( MORPH_RECT,
//                         Size( erosion_size+1, erosion_size+1 )
//                         );

//    morphologyEx(this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, MORPH_OPEN ,element1);
//    morphologyEx(this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, MORPH_OPEN ,element1);
//    morphologyEx(this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, MORPH_OPEN ,element1);
//    this->dbgForm("Erode/dilate ", this->m_fObjectMatrixDst, "/home/daria/wwwm/ede .jpg");

//    this->m_fObjectMatrixDst.convertTo(this->m_fObjectMatrixDst, CV_8UC1);
//    // sure background
//    Mat background;
//    dilate(this->m_fObjectMatrixDst, background, Mat());
//    // sure foreground
//    Mat foreground;
//    Mat distance;
//    distanceTransform(this->m_fObjectMatrixDst, distance, DIST_L2, CV_32F);
//    distance.convertTo(distance, -1, 255,5);
//    this->dbgForm("Distance ", distance, "/home/daria/wwwm/dist.jpg");
//    qDebug() << "distance type " << distance.type();
//    threshold(distance, foreground, 100, 255, THRESH_BINARY);

//    background.convertTo(background, CV_32F);

//    Mat hz;
//    qDebug() << "rows of background = " << background.rows << "cols = " << background.cols << "type" << background.type();
//    qDebug() << "rows of foreground = " << foreground.rows << "cols = " << foreground.cols << "type" << foreground.type();

//    this->m_fObjectMatrixDst.convertTo(this->m_fObjectMatrixDst, CV_32F);

//    subtract(background,foreground, this->m_fObjectMatrixDst);




    //this->contrast(this->m_fObjectMatrixDst, 500, 5);
    //normalize(this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, 0, 1, NORM_MINMAX);
}

void FilteredObject::visualizeFibers()
{
    //Бинаризация
    double thresh = 50;
    double maxValue = 255;
    threshold(this->m_pathOutput, this->m_pathOutput, thresh, maxValue, THRESH_BINARY);
    GaussianBlur(this->m_pathOutput, this->m_pathOutput, Size( 7, 7 ), 0, 0 );
    this->m_pathOutput.convertTo(this->m_pathOutput, CV_8UC3);

         //int numberOfContours = 0;
         std::vector<std::vector<Point>> contours;
//         Scalar color(0,0,255);

         findContours(this->m_pathOutput, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

     //convert grayscale to color image
        cvtColor(this->m_pathOutput, this->m_pathOutput, COLOR_GRAY2RGB);

        for (unsigned long cont = 0; cont < contours.size(); cont++)
        {
            uint8_t first = QRandomGenerator::global()->bounded(0, 255);
            uint8_t second = QRandomGenerator::global()->bounded(0, 255);
            uint8_t third = QRandomGenerator::global()->bounded(0, 255);
            Scalar color(first,second,third);
            drawContours(this->m_pathOutput, contours, cont, color, FILLED);
        }
        this->dbgForm("Fibers", this->m_pathOutput, "/home/daria/wwwm/FIBERS.jpg");

        this->m_fObjectMatrixTmp.convertTo(this->m_fObjectMatrixTmp, CV_8UC3);
        cvtColor(this->m_fObjectMatrixTmp, this->m_fObjectMatrixTmp, COLOR_GRAY2RGB);
//        this->dbgForm("tmp", this->m_fObjectMatrixTmp, "/home/daria/wwwm/dbg8.jpg");


        addWeighted (this->m_fObjectMatrixTmp, 0.4, this->m_pathOutput, 0.6, 0.0, this->m_fObjectMatrixDst);


int numberOfContours = 0;

        for (unsigned long i = 0; i < contours.size(); i++)
        {
                numberOfContours +=1;
//        //morphologyEx(this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, MORPH_CLOSE, contours[i]);
        }


        float s = this->SEffective();
        float relNumb = numberOfContours/s;
        qDebug() << "number of fibers = " << numberOfContours;
        qDebug() << "relative number of bibers = " << relNumb;

this->m_numberOfCont = numberOfContours;
this->m_relnumberOfCont = relNumb;



}
