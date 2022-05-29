#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDir>
#include <QProcess>
#include <QFileDialog>

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


    ui->MaxSize->addItems({"300", "350", "400", "450", "500", "550", "600", "650", "700"});
    ui->MaxSize->setCurrentIndex(4);

    ui->ErodeKernel->addItems({"1,1" , "2,2" , "3,3" , "4,4" , "5,5"});
    ui->ErodeKernel->setCurrentIndex(2);



}

MainWindow::~MainWindow()
{
    delete ui;
}

//void MainWindow::on_progress_changed(int progr)
//{
//    ui->progressBar->setValue(progr);
//}

void MainWindow::on_downloadButton_clicked()
{
    if(!this->m_object)
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
}



void MainWindow::on_countButton_clicked()
{
    if(fileName == nullptr)
    {
        qDebug() << "Please choose an image";
        return;
    }
    float ar = ui->area->text().toFloat();
    qDebug() <<"ar = "<<ar;
    int fs_ind = ui->MaxSize->currentIndex();
    int ec_ind = ui->ErodeKernel->currentIndex();

    //FilteredObject fObject;
    this->m_object = new FilteredObject();
    connect(this->m_object, &FilteredObject::progress_changed ,this, &MainWindow::on_progressBar_valueChanged);
    m_object->SETSmkm(ar);
    m_object->SETmaxFiberSize(fs_ind);
    m_object->SETerodeKernel(ec_ind);
    m_object->m_fObjectURL = fileName.toStdString();
    m_object->m_fObjectMatrixSrc = imread(m_object->m_fObjectURL, IMREAD_GRAYSCALE);

    m_object->preprocessing();
    m_object->m_fObjectMatrixSrc.convertTo(m_object->m_fObjectMatrixSrc, CV_32FC1);

    m_object->frangiFilter();
    m_object->postprocessing();

    m_object->path_opening();

    m_object->SETpathOutput(imread(m_object->m_fObjectPathURL, IMREAD_GRAYSCALE));

    m_object->visualizeFibers();

    int check = imwrite("/home/daria/wwwm/test1.jpg", m_object->m_fObjectMatrixDst);

    // if the image is not saved
    if (check == false)
        qDebug() << "Mission - Saving the image, FAILED";

    QPixmap pix1("/home/daria/wwwm/test1.jpg");
    int w1 = ui->label_2->width();
    int h1 = ui->label_2->height();
    ui->label_2->setPixmap(pix1);
    ui->label_2->setPixmap(pix1.scaled(w1,h1,Qt::KeepAspectRatio));
    ui->number->setNum(m_object->GETnumberOfCont());
    ui->relative->setNum(m_object->GETrelnumberOfCont());

}


void MainWindow::on_resetButton_clicked()
{
    ui->relative->clear();
    ui->number->clear();
    ui->sourceImage->clear();
    ui->label_2->clear();

    fileName = nullptr;
    this->m_object->clear();
    this->m_object = nullptr;

    ui->progressBar->setValue(0);


}


void MainWindow::on_progressBar_valueChanged(int value)
{
    ui->progressBar->setValue(value);
}

