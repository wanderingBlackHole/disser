/*
* Данный класс описывает объект, представляющий собой изображение с "волоконными" структурами.
* Класс имеет методы, позволяющие повысить контрастность исходного изображения - contrast(),
* вычислить гессиан матрицы изображения - hessian(), применить фильтр Франжи(1998) - frangiFilter().
* Объект создается при нажатии на кнопку "обработать изображение" на главной форме.
*/

#ifndef FILTEREDOBJECT_H
#define FILTEREDOBJECT_H

#include "opencv2/core.hpp"
#include "opencv2/cvconfig.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/utility.hpp>

#include <QVector>
#include <QPair>
#include <QQueue>
#include <QWidget>
#include <QDebug>
#include <QObject>
extern "C" {
    #include "pde_toolbox_bimage.h"
    #include "pde_toolbox_defs.h"
    #include "pde_toolbox_LSTB.h"
    #include "ImageMagickIO.h"
}

//#include <magick/api.h>
//#include <magick/MagickCore.h>
//#include <Magick++.h>

using namespace cv;

class FilteredObject : public QObject
{
    Q_OBJECT
public:
    FilteredObject();
    void saveToLocal(Mat image, QString dbgSave);
    void clear();
    String m_fObjectURL;
    String m_fObjectPathURL;

    // доп окно для промежуточных вычислений
    void dbgForm(QString name, Mat image, QString dbgSave);

    // матрицы исходного и отфильтрованного изображений
    Mat m_fObjectMatrixSrc;
    Mat m_fObjectMatrixDst;

    void contrast(Mat imageToApply, double alpha/*< Simple contrast control */, int beta/*< Simple brightness control */);
    void preprocessing();
    void postprocessing();

    // Матрица Гессе описывает вторые производные каждого пикселя в одном из направлений.
    // Так как secondDerXY = secondDerYX, нам необходимо вычислить три матрицы.
    void hessian(float sigma);
    void frangiFilter();

    //void distance();
    void erodeDiliate();
    void erodeDiliate1();

    void deleteSmallBlobs();

    void dstOverlaid();

    void distance();
    void startPoints();

    void path_opening();
    void path_opening2();

    // Посчитать эффективную площадь среза, занимаемую миокардиальной связью
    float SEffective();

    // Визуализация и подсчет коллагеновых волокон
    void visualizeFibers();
    void countFibers();

    // get/set
    int GETusefulPix()
    {
        return m_usefulPixs;
    }
    void SETusefulPix(int usefulp)
    {
        m_usefulPixs = usefulp;
    }

    float GETSEff()
    {
        return m_SEff;
    }
    void SETSEff(float seff)
    {
        m_SEff = seff;
    }

    float GETSmkm()
    {
        return m_Smkm;
    }
    void SETSmkm(float smkm)
    {
        m_Smkm = smkm;
        qDebug()<< "SETTER: Smkm = "<<m_Smkm;
    }

    Mat GETpathOutput()
    {
        return m_pathOutput;
    }
    void SETpathOutput(Mat pop)
    {
        m_pathOutput = pop;
    }

    int GETnumberOfCont()
    {
        return m_numberOfCont;
    }
    void SETnumberOfCont(int numb)
    {
        m_numberOfCont = numb;
    }
    float GETrelnumberOfCont()
    {
        return m_relnumberOfCont;
    }
    void SETrelnumberOfCont(float numb)
    {
        m_relnumberOfCont = numb;
    }
    void SETerodeKernel(int kern_ind)
    {
        m_erodeKernel_ind = kern_ind;
    }
    void SETmaxFiberSize(int sz_ind)
    {
        m_maxFiberSize_ind = sz_ind;
    }

signals:
    void progress_changed(int progr);

private:
    QVector<QWidget*> dbgForms;
    // erode kernel size
    int m_erodeKernel_ind;
    // max fiber size
    int m_maxFiberSize_ind;
    // белые пиксели исходного изображения
    int m_usefulPixs;
    // эффективная площадь
    float m_SEff;

    Mat m_pathOutput;
    int m_numberOfCont;
    float m_relnumberOfCont;
    BIMAGE* m_fObjectURLCharSrc;
    BIMAGE* m_fObjectURLCharDst;

    // площадь изображения в мкм
    float m_Smkm;

    // матрица чтобы создать наложенное изображение
    Mat m_fObjectMatrixTmp;

    Mat m_fObjectMatrixSearchForFibers;
    // свертки по фильтру Гессе
    Mat m_Dxx;
    Mat m_Dxy;
    Mat m_Dyy;
    // собственные значения
    Mat m_lambda1;
    Mat m_lambda2;
    // собственные векторы
    Mat m_vx;
    Mat m_vy;
    // направление меньшего собственного вектора
    QVector<Mat> m_angles;
    // результаты применения фильтра Франжи
    QVector<Mat> m_filtered;

    struct m_fibersCoord{
        int x;
        int y;
    };
    //начальные точки
    QVector<m_fibersCoord> m_startP;



};

#endif // FILTEREDOBJECT_H
