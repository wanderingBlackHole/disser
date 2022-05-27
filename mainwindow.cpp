#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDir>
#include <QProcess>
#include <QFileDialog>
#include "filteredobject.h"
#include "pathopen.h"

#include <QGraphicsEffect>

QString fileName;
String filename;
Mat markerMask;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setWindowTitle("What's wrong with the mouse?");
    this->setWindowIcon(QIcon("/home/daria/wwwm/icon.jpg"));

    ui->MaxSize->setCurrentText("500");
    ui->ErodeKernel->setCurrentText("3,3");
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_downloadButton_clicked()
{
    int w = ui->sourceImage->width();
    int h = ui->sourceImage->height();
    fileName = QFileDialog::getOpenFileName(this,
        tr("Open Image"), "/home/daria/imagesforsw", tr("Image Files (*.png *.jpg *.bmp *tif)"));
    qDebug() << "url of chosen image is" << fileName;
    QPixmap pix(fileName);
    ui->sourceImage->setPixmap(pix);
    ui->sourceImage->setPixmap(pix.scaled(w,h,Qt::KeepAspectRatio));
}



void MainWindow::on_countButton_clicked()
{
    float ar = ui->area->text().toFloat();
    qDebug() <<"ar = "<<ar;

    FilteredObject fObject;
    fObject.SETSmkm(ar);
    fObject.m_fObjectURL = fileName.toStdString();
    fObject.m_fObjectMatrixSrc = imread(fObject.m_fObjectURL, IMREAD_GRAYSCALE);

    fObject.preprocessing();
    fObject.m_fObjectMatrixSrc.convertTo(fObject.m_fObjectMatrixSrc, CV_32FC1);

    fObject.frangiFilter();
    fObject.postprocessing();

    fObject.path_opening();

    fObject.SETpathOutput(imread(fObject.m_fObjectPathURL, IMREAD_GRAYSCALE));

    fObject.visualizeFibers();

    int check = imwrite("/home/daria/wwwm/test1.jpg", fObject.m_fObjectMatrixDst);

    // if the image is not saved
    if (check == false)
        qDebug() << "Mission - Saving the image, FAILED";

    QPixmap pix1("/home/daria/wwwm/test1.jpg");
    int w1 = ui->label_2->width();
    int h1 = ui->label_2->height();
    ui->label_2->setPixmap(pix1);
    ui->label_2->setPixmap(pix1.scaled(w1,h1,Qt::KeepAspectRatio));
    ui->label->setNum(fObject.GETnumberOfCont());
    ui->relative->setNum(fObject.GETrelnumberOfCont());

}
