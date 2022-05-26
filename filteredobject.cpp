#include "filteredobject.h"
#include <cmath>
#include <QPixmap>
#include <QLabel>
#include <QList>

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
     double thresh = 40;
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
                         Size( erosion_size+2, erosion_size+2 ),
                         Point( -1, -1 ) );
    //erode( this->m_pathOutput, this->m_pathOutput,element,Point(-1,-1), 1);
    erode( this->m_pathOutput, this->m_pathOutput,element,Point(-1,-1), 1);
    this->dbgForm("Erode ", this->m_pathOutput, "/home/daria/wwwm/e.jpg");

    Mat element2 = getStructuringElement( MORPH_RECT,
                         Size( erosion_size, erosion_size ),Point( -1, -1 ) );
    dilate( this->m_pathOutput, this->m_pathOutput, element2, Point(-1,-1), 1);
      this->dbgForm("dilate ", this->m_pathOutput, "/home/daria/wwwm/d.jpg");
//    dilate( this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, element);
//    erode( this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, element);
    //erode( this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, Mat());

//    Mat element1 = getStructuringElement( MORPH_RECT,
//                         Size( erosion_size+3, erosion_size+3 ),
//                         Point( erosion_size, erosion_size ) );
//    morphologyEx(this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, MORPH_CLOSE ,element1);
//    erode( this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, element);
//    this->dbgForm("Erode/dilate ", this->m_fObjectMatrixDst, "/home/daria/wwwm/ede.jpg");

}
//void FilteredObject::erodeDiliate1()
//{
//    int erosion_size = 1;
//    Mat element = getStructuringElement( MORPH_RECT,
//                         Size( erosion_size+2, erosion_size+6 ),
//                         Point( erosion_size, erosion_size ) );

//    Mat element1 = getStructuringElement( MORPH_RECT,
//                         Size( erosion_size+3, erosion_size+3 ),
//                         Point( erosion_size, erosion_size ) );

//    morphologyEx(this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, MORPH_CLOSE ,element);
//    this->dbgForm("CLOSE", this->m_fObjectMatrixDst, "/home/daria/wwwm/close.jpg");

//    morphologyEx(this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, MORPH_OPEN ,element1);
//    this->dbgForm("OPEN", this->m_fObjectMatrixDst, "/home/daria/wwwm/open.jpg");

//}
void FilteredObject::dstOverlaid()
{
    this->visualizeFibers();
    this->path_opening();
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

        // Deallocate
        delete[] input_image;
        delete[] output_image;
        BIMAGE_destructor(input_bimage);
        BIMAGE_destructor(output_bimage);

    //this->dbgForm("HUGUE OPENING PATH", this->m_pathOutput, "/home/daria/wwwm/path_output.jpg");
    /*--------------------------------------------------------------------------------------------------------------*/

}

void FilteredObject::path_opening2()
{
//    unsigned char   *input_image;         /* The input image */
//    int             nx, ny;               /* Image dimensions */
//    int             L;                    /* The threshold line length */
//    int             K;                    /* The maximum number of gaps in the path */
//    unsigned char   *output_image ;       /* Output image */




    //pathopen(this->m_fObjectURLCharSrc,nx,ny,L,K,output_image);

    /*--------------------------------------------------------------------------------------------------------------*/


    const char* currentPath = "/home/daria/wwwm";
    const char* currentFileName = "/home/daria/wwwm/pop2.jpg";
    const char* outputFileName = "/home/daria/wwwm/pop2PATH.jpg";


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
    /*--------------------------------------------------------------------------------------------------------------*/

}


void FilteredObject::distance()
{
    cv::Mat src = cv::imread("/home/daria/wwwm/FIBERS.jpg");
    if(!src.data)
        return;

    cv::Mat bw;
    cv::cvtColor(src,bw,COLOR_BGR2GRAY);
    cv::threshold(bw,bw,40,255,THRESH_BINARY);
    this->dbgForm("FIBERS BINARY", bw, "/home/daria/wwwm/fibers_bw.jpg");


}

void FilteredObject::visualizeFibers()
{
//    this->erodeDiliate();
//    this->path_opening2();
    //Бинаризация

    double thresh = 10;
    double maxValue = 255;
    threshold(this->m_pathOutput, this->m_pathOutput, thresh, maxValue, THRESH_BINARY);
    //GaussianBlur(this->m_pathOutput, this->m_pathOutput, Size( 7, 7 ), 0, 0 );
    this->m_pathOutput.convertTo(this->m_pathOutput, CV_8UC3);


         std::vector<std::vector<Point>> allCont; // все контуры
         std::vector<std::vector<Point>> divCont; // разбитые на несколько контуров изначально длинные контуры

         findContours(this->m_pathOutput, allCont, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

         std::vector<std::vector<Point>> shortCont(allCont); // сюда сложим все кроме изначально длинных контуров
         qDebug() <<"allCont.size()"<< allCont.size();
         qDebug() <<"shortCont.size()"<< shortCont.size();
         QList<unsigned long> delete_list = {};

         cvtColor(this->m_pathOutput, this->m_pathOutput, COLOR_GRAY2RGB);

/*---------------------------CHECKING TOO BIG CONTOURS AGAIN--------------------------------------*/
        for (unsigned long cont = 0; cont < allCont.size(); cont++)
        {

            uint8_t first = QRandomGenerator::global()->bounded(0, 255);
            uint8_t second = QRandomGenerator::global()->bounded(0, 255);
            uint8_t third = QRandomGenerator::global()->bounded(0, 255);
            Scalar color(first,second,third);
            Scalar colorRect(0,0,255);

            int max_sz = 500;

            if (cv::arcLength(allCont.at(cont), true) > max_sz) // если периметр контура слишком большой, проверим его....
                                                                // может, там скрывается еще один контур...... или два)))))))))
            {
                qDebug()<< "need to erase" << cont;
                delete_list.append(cont);
                //contours.erase(contours.begin()+cont);
                int erosion_size = 1;
                Mat element = getStructuringElement( MORPH_RECT,
                                     Size( erosion_size+2, erosion_size+2 ),
                                     Point( -1, -1 ) );

//                int x1 = boundingRect(allCont.at(cont)).x;
//                int x2 = boundingRect(allCont.at(cont)).x + boundingRect(allCont.at(cont)).width;
//                int y1 = boundingRect(allCont.at(cont)).y;
//                int y2 = boundingRect(allCont.at(cont)).y + boundingRect(allCont.at(cont)).height;

//                // очертим контур, с которым сейчас работаем, прямоугольником
//                rectangle( this->m_pathOutput, Point(x1,y1) , Point(x2,y2) , colorRect, 2);

                Mat tempArea;
                Mat mask (this->m_pathOutput.size(), CV_8UC1, Scalar(0));

                drawContours(mask, allCont, cont, Scalar(255), FILLED);
                this->m_pathOutput.copyTo(tempArea, mask);

                cvtColor(tempArea, tempArea, COLOR_RGB2GRAY);
                // сузим контур чтобы выявить потенциальные точки соприкосновения
                erode( tempArea, tempArea, element, Point(-1,-1), 1);
                // попробуем сделать из большого контура несколько контуров (нет так нет)
                //this->dbgForm("For pop2", tempArea, "/home/daria/wwwm/pop2.jpg");
                this->path_opening2();
                findContours(tempArea, divCont, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); // и сохраним их в векторе divCont


                cvtColor(tempArea, tempArea, COLOR_GRAY2RGB);
                // нарисуем найденные разделенные контуры
                for (unsigned long contdiv = 0; contdiv < divCont.size(); contdiv++)
                {
                    uint8_t firstl = QRandomGenerator::global()->bounded(0, 255);
                    uint8_t secondl = QRandomGenerator::global()->bounded(0, 255);
                    uint8_t thirdl = QRandomGenerator::global()->bounded(0, 255);
                    Scalar colordiv(firstl,secondl,thirdl);
                    drawContours(this->m_pathOutput, divCont, contdiv, colordiv, FILLED);
                }
                //shortCont.erase(shortCont.begin() + cont);

                //this->dbgForm("Only long Fibers", tempArea, "/home/daria/wwwm/LONGF.jpg");
            } // конец проверки длинных контуров

            //drawContours(this->m_pathOutput, shortCont, cont, color, FILLED);

        }
        std::vector<std::vector<Point>>::iterator iter = shortCont.begin();
        for (int i = 0; i < delete_list.size(); i++)
        {
            iter += delete_list[i];
            shortCont.erase(iter);
            qDebug()<< "erasing " << delete_list[i];
        }
                 qDebug() <<"shortCont.size()"<< shortCont.size();

/*---------------------------------------------------------------------------------------------------*/
        this->dbgForm("Only long Fibers", this->m_pathOutput, "/home/daria/wwwm/LONGF.jpg");

//        for (unsigned long scont = 0; scont < shortCont.size(); scont++)
//        {
////            Mat tempArea;
////            Mat mask (this->m_pathOutput.size(), CV_8UC1, Scalar(0));

////            drawContours(mask, shortCont, scont, Scalar(255), FILLED);
////            this->m_pathOutput.copyTo(tempArea, mask);

//            uint8_t firsts = QRandomGenerator::global()->bounded(0, 255);
//                        uint8_t seconds = QRandomGenerator::global()->bounded(0, 255);
//                        uint8_t thirds = QRandomGenerator::global()->bounded(0, 255);
//                        Scalar colors(firsts,seconds,thirds);
//            drawContours(this->m_pathOutput, shortCont, scont, colors, FILLED);
//        }


        //Mat ccc (this->m_pathOutput.size(), CV_8UC1, Scalar(0));
        Scalar colors(0,0,255);
                  drawContours(this->m_pathOutput, shortCont, -1, colors, 3);
        this->dbgForm("ALLFibers", this->m_pathOutput, "/home/daria/wwwm/ALLFIBERS.jpg");


//        this->m_fObjectMatrixTmp.convertTo(this->m_fObjectMatrixTmp, CV_8UC3);
//        cvtColor(this->m_fObjectMatrixTmp, this->m_fObjectMatrixTmp, COLOR_GRAY2RGB);
////        this->dbgForm("tmp", this->m_fObjectMatrixTmp, "/home/daria/wwwm/dbg8.jpg");


//        addWeighted (this->m_fObjectMatrixTmp, 0.4, this->m_pathOutput, 0.6, 0.0, this->m_fObjectMatrixDst);


//int numberOfContours = 0;

//        for (unsigned long i = 0; i < contours.size(); i++)
//        {
//                numberOfContours +=1;
////        //morphologyEx(this->m_fObjectMatrixDst, this->m_fObjectMatrixDst, MORPH_CLOSE, contours[i]);
//        }


//        float s = this->SEffective();
//        float relNumb = numberOfContours/s;
//        qDebug() << "number of fibers = " << numberOfContours;
//        qDebug() << "relative number of bibers = " << relNumb;

//this->m_numberOfCont = numberOfContours;
//this->m_relnumberOfCont = relNumb;



}
