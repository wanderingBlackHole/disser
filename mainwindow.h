#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "filteredobject.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_downloadButton_clicked();

    void on_countButton_clicked();

    void on_resetButton_clicked();
    void on_progressBar_valueChanged(int value);

//public slots:
//    void on_progress_changed(int progr);

private:
    Ui::MainWindow *ui;
    FilteredObject *m_object = nullptr;
};
#endif // MAINWINDOW_H
