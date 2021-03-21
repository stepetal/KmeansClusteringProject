#include "mainwindow.h"




MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), iter_numb(50)
{
    setWindowTitle("K-means clustering");
    createWidgets();
    createActions();
    createLayout();
    createConnections();
}

MainWindow::~MainWindow()
{

}

void MainWindow::closeEvent(QCloseEvent *event)
{
    cv::destroyAllWindows();
}

void MainWindow::createWidgets()
{
    framePathTextEdit = new QTextEdit();
    framePathTextEdit->setMaximumHeight(30);
    framePathTextEdit->setReadOnly(true);
    kValueSpinBox = new QSpinBox();
    kValueSpinBox->setWrapping(true);//значения идут по кругу
    kValueSpinBox->setRange(1,10);
    kValueSpinBox->setAlignment(Qt::AlignCenter);
    kValueSpinBox->setValue(5);
    rgbFeatureRadioButton = new QRadioButton("Цветовая пространство RGB");
    hsvFeatureRadioButton = new QRadioButton("Цветовое пространство HSV");
    hsvFeatureRadioButton->setChecked(true);
    startPushButton = new QPushButton("Старт");
    validateClusteringCheckBox = new QCheckBox("Сравнить с алгоритмом kmeans в OpenCV");
}

void MainWindow::createLayout()
{
    QWidget *mainWindowWidget = new QWidget();
    QGridLayout *mainWindowWidgetLayout = new QGridLayout();
    QLabel *title = new QLabel("Кластеризация с помощью алгоритма K - средних");
    title->setFont(QFont("Times",14,QFont::Bold,true));
    QToolBar *openFrameButtonToolBar = new QToolBar();
    openFrameButtonToolBar->addAction(openFrameAction);

    QGroupBox *openFrameGroupBox = new QGroupBox("Задание фрейма");
    QGridLayout *openFrameGroupBoxLayout = new QGridLayout();
    openFrameGroupBoxLayout->addWidget(framePathTextEdit,0,0,1,1);
    openFrameGroupBoxLayout->addWidget(openFrameButtonToolBar,0,1,1,1);
    openFrameGroupBox->setLayout(openFrameGroupBoxLayout);

    QGroupBox *chooseKValueGroupBox = new QGroupBox("Выбор количества кластеров");
    QHBoxLayout *chooseKValueGroupBoxLayout = new QHBoxLayout();
    chooseKValueGroupBoxLayout->addWidget(kValueSpinBox);
    chooseKValueGroupBox->setLayout(chooseKValueGroupBoxLayout);

    QGroupBox *validateClusteringGroupBox = new QGroupBox("Сравнение с эталоном");
    QHBoxLayout *validateClusteringGroupBoxLayout = new QHBoxLayout();
    validateClusteringGroupBoxLayout->addWidget(validateClusteringCheckBox);
    validateClusteringGroupBox->setLayout(validateClusteringGroupBoxLayout);

    QGroupBox *clusteringFeatureGroupBox = new QGroupBox("Выбор признака кластеризации");
    QGridLayout *clusteringFeatureGroupBoxLayout = new QGridLayout();
    clusteringFeatureGroupBoxLayout->addWidget(rgbFeatureRadioButton,0,0,1,1);
    clusteringFeatureGroupBoxLayout->addWidget(hsvFeatureRadioButton,0,1,1,1);
    clusteringFeatureGroupBox->setLayout(clusteringFeatureGroupBoxLayout);

    mainWindowWidgetLayout->addWidget(title,0,0,1,2,Qt::AlignCenter);
    mainWindowWidgetLayout->addWidget(openFrameGroupBox,1,0,1,2);
    mainWindowWidgetLayout->addWidget(chooseKValueGroupBox,2,0,1,2);
    mainWindowWidgetLayout->addWidget(clusteringFeatureGroupBox,3,0,1,2);
    mainWindowWidgetLayout->addWidget(validateClusteringGroupBox,4,0,1,2);
    mainWindowWidgetLayout->addWidget(startPushButton,5,0,1,2,Qt::AlignCenter);
    mainWindowWidget->setLayout(mainWindowWidgetLayout);

    //установка стилей
    mainWindowWidget->setStyleSheet("background-color: #0e7fc9; color: white; font-size: 12pt");
    kValueSpinBox->setStyleSheet("background-color: white; color: black");
    framePathTextEdit->setStyleSheet("background-color: white; color: black");
    openFrameButtonToolBar->setStyleSheet("font-size: 8pt; color: black;");
    setCentralWidget(mainWindowWidget);
}

void MainWindow::createActions()
{
    openFrameAction = new QAction(QIcon("://open-file-icon.png"),"");
    openFrameAction->setToolTip("Открыть изображение");
}

void MainWindow::createConnections()
{
    connect(startPushButton,&QPushButton::clicked,[&](){
        if(!framePathTextEdit->toPlainText().isEmpty()){
            loadImageForClustering();
            showInputFrame();
            performKmeansClustering();
            showClusteredImage();
            if(validateClusteringCheckBox->isChecked()){
                performKmeansClusteringWithOpenCV();
                showClusteringWithOpenCVImage();
            }
        } else {
            QMessageBox warningBox;
            warningBox.setText("Файл не выбран !");
            warningBox.setStandardButtons(QMessageBox::Ok);
            warningBox.setIcon(QMessageBox::Warning);
            int box_result = warningBox.exec();
            switch (box_result) {
            case QMessageBox::Ok:
                framePathTextEdit->setStyleSheet("background-color:#f7766d; border-color: #f54e42");
                QTimer::singleShot(300,this,[=]{
                    framePathTextEdit->setStyleSheet("background-color:white; border-color: black");
                });
                break;
            default:
                break;
            }
        }
    });
    connect(openFrameAction,&QAction::triggered,[&](){
        QString framePath = QFileDialog::getOpenFileName(this,"Открыть изображение","","Image files: (*.jpg *.png)");
        if(!framePath.isEmpty()){
            framePathTextEdit->setText(framePath);
        }
    });
}

void MainWindow::loadImageForClustering()
{
    frameUnitsList.clear();
    inputFrame = cv::imread(framePathTextEdit->toPlainText().toStdString().c_str(), cv::IMREAD_COLOR);
    //проведем медианную фильтрацию входного изображения
    cv::medianBlur(inputFrame, inputFrame, 3);
    //использование цветового пространства HSV
    if(hsvFeatureRadioButton->isChecked()){
        cv::cvtColor(inputFrame,inputFrame,cv::COLOR_BGR2HSV);
    }
    for (int i = 0; i < inputFrame.cols * inputFrame.rows; i++) {
        frameUnitsList.append(FrameUnit((float)inputFrame.at<cv::Vec3b>(i)[2] / 255, // Red
                                       (float)inputFrame.at<cv::Vec3b>(i)[1] / 255, // Green
                                       (float)inputFrame.at<cv::Vec3b>(i)[0] / 255  // Blue
                                      )
                             );
    }
    if(validateClusteringCheckBox->isChecked()){
        pixelMatrixForOpenCV = cv::Mat::zeros(inputFrame.cols * inputFrame.rows, 3, CV_32F);
        for(int i = 0; i < inputFrame.cols * inputFrame.rows; i++){
            pixelMatrixForOpenCV.at<float>(i, 0) = (float)inputFrame.at<cv::Vec3b>(i)[0] / 255; // Blue
            pixelMatrixForOpenCV.at<float>(i,1) = (float)inputFrame.at<cv::Vec3b>(i)[1] / 255; // Green
            pixelMatrixForOpenCV.at<float>(i, 2) = (float)inputFrame.at<cv::Vec3b>(i)[2] / 255; // Red
        }
    }
}

void MainWindow::showInputFrame()
{
    if (!inputFrame.empty()){
        cv::imshow("Input frame",inputFrame);
        cv::waitKey(300);
    }
}

void MainWindow::showClusteredImage()
{
    if (!resultClusteredMatrix.empty()){
        cv::imshow("Clustered frame",resultClusteredMatrix);
        cv::waitKey(300);
    }
}

void MainWindow::performKmeansClustering()
{
    clusterCenterUnits.clear();
    //в начале следует случайным образом отобрать элементы фрейма - центры кластеров
    for(auto i = 0; i < kValueSpinBox->value();i++){
        clusterCenterUnits.append(frameUnitsList.at(QRandomGenerator::global()->generate() % frameUnitsList.size()));
    }
    //цикл по количеству итераций, заданных в начале программы
    for(auto i = 0; i < iter_numb; i++){
        int cur_cluster_numb {0};
        //задаем расстояния и номера кластеров для элементов согласно удаленности от центра кластера
        foreach (auto clusterCenterUnit, clusterCenterUnits) {
            for(int j = 0; j < frameUnitsList.size(); j++){
                auto cur_distance = clusterCenterUnit.calculateDistance(frameUnitsList[j]);
                if(cur_distance < frameUnitsList[j].minDistance){
                    frameUnitsList[j].minDistance = cur_distance;
                    frameUnitsList[j].clust_numb = cur_cluster_numb;
                }
            }
            cur_cluster_numb++;
        }

        QList<int> unitsInCluster;//число элементов, находящихся в кластере
        //суммарное значение компонент каналов в каждом кластере
        QList<float> sumOfRedChValuesInCluster, sumOfGreenChValuesInCluster, sumOfBlueChValuesInCluster;
        for(int j = 0; j < kValueSpinBox->value();j++){
            unitsInCluster.append(0);
            sumOfBlueChValuesInCluster.append(0);
            sumOfGreenChValuesInCluster.append(0);
            sumOfRedChValuesInCluster.append(0);
        }
        //находим суммарные значения компонент каждого канала в кластере
        for(int j = 0; j < frameUnitsList.size(); j++){
            unitsInCluster[frameUnitsList[j].clust_numb] += 1;
            sumOfBlueChValuesInCluster[frameUnitsList[j].clust_numb] += frameUnitsList[j].blue;
            sumOfGreenChValuesInCluster[frameUnitsList[j].clust_numb] += frameUnitsList[j].green;
            sumOfRedChValuesInCluster[frameUnitsList[j].clust_numb] += frameUnitsList[j].red;
            frameUnitsList[j].minDistance = FLT_MAX;
        }
        //вычисляем новые центры кластеров как отношение суммарного значения канала к количеству элементов
        for(int j = 0; j < clusterCenterUnits.size(); j++){
            clusterCenterUnits[j].blue = sumOfBlueChValuesInCluster[j] / unitsInCluster[j];
            clusterCenterUnits[j].green = sumOfGreenChValuesInCluster[j] / unitsInCluster[j];
            clusterCenterUnits[j].red = sumOfRedChValuesInCluster[j] / unitsInCluster[j];
        }
        //переходим к следующей итерации, в результате которой элементам фрейма будут назначены номера новых центров кластеров
    }
    //подготовка маски для кластеров
    QList<int> colorsList;
    for (int i = 0; i < clusterCenterUnits.count(); i++) {
        colorsList.append(255 / (i + 1));
    }
    resultClusteredMatrix = cv::Mat(inputFrame.rows, inputFrame.cols, CV_32F);
    for (int i = 0; i < inputFrame.cols * inputFrame.rows; i++) {
        resultClusteredMatrix.at<float>(i / inputFrame.cols, i % inputFrame.cols) = (float)(colorsList[frameUnitsList[i].clust_numb]);
    }
    resultClusteredMatrix.convertTo(resultClusteredMatrix, CV_8U);//конвертация в unsigned char
    cv::medianBlur(resultClusteredMatrix, resultClusteredMatrix,3);//медианный фильтр для сглаживания
}

void MainWindow::performKmeansClusteringWithOpenCV()
{
    QList<int> colorsList;
    for (int i = 0; i < kValueSpinBox->value(); i++) {
        colorsList.append(255 / (i + 1));
    }
    cv::Mat bestLabels;//набор номеров кластеров для каждого элемента
    cv::Mat centers;//матрица центров кластеров
    cv::kmeans(pixelMatrixForOpenCV,
               kValueSpinBox->value(),
               bestLabels,
               cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,10,1.0),
               5,
               cv::KMEANS_PP_CENTERS,
               centers);
    resultClusteredMatrixWithOpenCV = cv::Mat(inputFrame.rows, inputFrame.cols, CV_32F);
    for (int i = 0; i < inputFrame.cols * inputFrame.rows; i++) {
        resultClusteredMatrixWithOpenCV.at<float>(i / inputFrame.cols, i % inputFrame.cols) = (float)(colorsList[bestLabels.at<int>(i, 0)]);
    }
    resultClusteredMatrixWithOpenCV.convertTo(resultClusteredMatrixWithOpenCV, CV_8U);//конвертация в unsigned char
    cv::medianBlur(resultClusteredMatrixWithOpenCV, resultClusteredMatrixWithOpenCV, 3);//медианный фильтр для сглаживания
}

void MainWindow::showClusteringWithOpenCVImage()
{
    if (!resultClusteredMatrixWithOpenCV.empty()){
        cv::imshow("cv::kmeans clustered frame",resultClusteredMatrixWithOpenCV);
        cv::waitKey(300);
    }
}


