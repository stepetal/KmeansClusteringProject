#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGridLayout>
#include <QTextEdit>
#include <QLabel>
#include <QFileDialog>
#include <QPushButton>
#include <QGroupBox>
#include <QRadioButton>
#include <QCheckBox>
#include <QIcon>
#include <QDebug>
#include <QMessageBox>
#include <QTimer>
#include <QAction>
#include <QToolBar>
#include <QSpinBox>
#include <QtMath>
#include <QRandomGenerator>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>



class MainWindow : public QMainWindow
{
    Q_OBJECT
    // элемент изображения
    struct FrameUnit
    {
        float red,green,blue; // цвета, соответствующие трем каналам элемента изображения
        int clust_numb;// текущий номер класетра
        float minDistance;//наименьшее расстояние до центра ближайшего кластера
        FrameUnit(float r,float g,float b) : red(r),green(g),blue(b),clust_numb(0),minDistance(FLT_MAX){}
        void setRedChValue(int r){red = r;}
        void setGreenChValue(int g){green = g;}
        void setBlueChValue(int b){blue = b;}
        //нахождение Евклидова расстояния между двумя точками
        float calculateDistance(FrameUnit fu) { return (float)(qSqrt(qPow(fu.red - red, 2) + qPow(fu.green - green, 2) + qPow(fu.blue - blue,2)));}
    };
    QTextEdit *framePathTextEdit;//путь к фрейму
    QSpinBox *kValueSpinBox;//выбор количества кластеров
    QRadioButton *rgbFeatureRadioButton;//признак кластеризации - значение яркости пикселя rgb
    QRadioButton *hsvFeatureRadioButton;//признак кластеризации - значение цвета
    QAction *openFrameAction;
    QPushButton *startPushButton;
    QCheckBox *validateClusteringCheckBox;//проверка полученной кластеризации
    int iter_numb;//количество итераций алгоритма кластеризации
    cv::Mat inputFrame;//кадр для анализа
    cv::Mat pixelMatrixForOpenCV;//матрица значений яркости пикселей, применяемая в алгоритме cv::kmeans
    cv::Mat resultClusteredMatrix;//матрица, полученная по результатам кластеризации
    cv::Mat resultClusteredMatrixWithOpenCV;//матрица, полученая по результатам кластеризации с помощью OpenCV

    QList<FrameUnit> frameUnitsList;//список элементов кадра
    QList<FrameUnit> clusterCenterUnits;//список элементов изображения - центров кластеров

public:
    MainWindow(QWidget *parent = 0);
    ~MainWindow();
// создание пользовательского интерфейса
    void createWidgets();
    void createActions();
    void createLayout();
    void createConnections();
// методы для работы с OpenCV
    void loadImageForClustering();
    void showInputFrame();
    void showClusteredImage();
    void performKmeansClustering();//реализация алгоритма кластеризации К - средних
    void performKmeansClusteringWithOpenCV();//использование готового алгоритма cv::kmeans
    void showClusteringWithOpenCVImage();

    // QWidget interface
protected:
    void closeEvent(QCloseEvent *event);
};



#endif // MAINWINDOW_H
