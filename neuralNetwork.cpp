/*
Afaq Alam 21i-1700 N
Eman Tahir 21i-1718 N
Shizra Burney 21i-2660 N
*/

#include <iostream>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include <cstdlib>
#include <time.h>
#include <random>
#include <fstream>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <cstring>
#include <iomanip>
#include <cmath>

#define TOTAL_LAYERS 3
#define INPUT_NEURONS 4
#define HIDDEN_NEURONS 5
#define OUTPUT_NEURONS 3
#define LEARNING_RATE 0.01
#define inputFilename "Iris.csv"
#define MAX_ROWS 150
#define BATCH_SIZE 10

/*------------------OUTPUT NEURONS (index wise):
                1.  iris setoso
                2.  Iris-versicolor
                3.  Iris-virginica
*/

using namespace std;

struct Neuron
{
public:
    double activation;
    int bias;

    Neuron()
    {
        activation = 0;
        bias = 1;
    }
};

struct Layer
{
public:
    Neuron *neurons;
    int totalNeurons;

    void setNeurons(int n)
    {
        totalNeurons = n;
        neurons = new Neuron[n];
    }
};

struct ThreadData
{
    int currRightNeuron;
    double *weights;
    Layer *layers;
    int currLayer;
    int prevLayer;
};

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void weightsLoop(double randNum, int layer, int leftNodes, int rightNodes)
{
    string fileName = "weights" + to_string(layer) + ".txt";
    ofstream weights(fileName);

    if (!weights.is_open())
    {
        cerr << "\nError opening the file for writing.\n";
    }

    mt19937 gen(random_device{}());

    uniform_real_distribution<double> dis(-2.0f, 2.0f);

    for (int ln = 0; ln < leftNodes; ln++)
    {
        for (int rn = 0; rn < rightNodes; rn++)
        {
            randNum = dis(gen);
            weights << to_string(randNum) + " ";
        }
        weights << endl;
    }

    weights.close();
}

void initNetwork(Layer layers[])
{

    for (int i = 0; i < TOTAL_LAYERS; i++)
    {
        switch (i)
        {
        case 0:
            layers[i].setNeurons(INPUT_NEURONS);
            break;

        case (TOTAL_LAYERS - 1):
            layers[i].setNeurons(OUTPUT_NEURONS);
            break;

        default:
            layers[i].setNeurons(HIDDEN_NEURONS);
        }
    }

    double randNum;

    for (int l = 0; l < TOTAL_LAYERS - 1; l++)
    {
        switch (l)
        {
        case 0:
            weightsLoop(randNum, l, INPUT_NEURONS, HIDDEN_NEURONS);
            break;

        case TOTAL_LAYERS - 2:
            weightsLoop(randNum, l, HIDDEN_NEURONS, OUTPUT_NEURONS);
            // l = TOTAL_LAYERS;
            break;

        default:
            weightsLoop(randNum, l, HIDDEN_NEURONS, HIDDEN_NEURONS);
            break;
        }
    }
}

void sendWeights(int layer, double *&weights, int size, int fw[2])
{
    // close(fw[0]);

    write(fw[1], weights, size * sizeof(double));

    // close(fw[1]);
}

void receiveWeights(int layer, double *&weights, int size, int fw[2])
{
    // close(fw[1]);

    read(fw[0], weights, size * sizeof(double));

    // close(fw[0]);
}

double sigmoid(double x)
{
    double result = 1 / (1 + exp(-x));

    return -2.0f + result * (-2.0f - +2.0f);
}

double derivative(double x)
{
    return x * (1.0 - x);
}

double delta(double loss, double partialDer)
{
    return loss * partialDer;
}

void readWeights(int layer, double *&w)
{
    string fileName = "weights" + to_string(layer) + ".txt";
    ifstream weights(fileName);

    if (!weights.is_open())
    {
        cerr << "\nError opening the file for reading.\n";
    }

    int i = 0;
    double value;

    while (weights >> value)
    {
        w[i] = value;
        i++;
    }

    weights.close();
}

void writeWeights(int layer, double *&w, int size)
{
    string fileName = "weights" + to_string(layer) + ".txt";
    ofstream weights(fileName);

    if (!weights.is_open())
    {
        cerr << "\nError opening the file for writing.\n";
    }

    for (int i = 0; i < size; ++i)
    {
        weights << w[i] << " ";
    }

    weights.close();
}

void softmax(Layer layers[], int currLayer, double prob[OUTPUT_NEURONS])
{
    double sumExp = 0;
    for (int i = 0; i < OUTPUT_NEURONS; i++)
    {
        sumExp += exp(layers[currLayer].neurons[i].activation);
    }
    for (int i = 0; i < OUTPUT_NEURONS; i++)
    {
        prob[i] = exp(layers[currLayer].neurons[i].activation) / sumExp;
    }
}

void calculateCost(double pred[], double req[], double &cost)
{
    cost = 0;
    for (int i = 0; i < OUTPUT_NEURONS; i++)
    {
        cost += (double)pow((pred[i] - req[i]), 2);
        // cout << cost << endl;
    }
}

void *processNeuron(void *args)
{
    ThreadData *data = (ThreadData *)args;

    pthread_mutex_lock(&mutex);

    for (int j = 0; j < data->layers[data->prevLayer].totalNeurons; j++)
    {
        // cout << "Debug: Layer " << data->currLayer << ", Neurons: " << data->layers[data->currLayer].totalNeurons << ", Current Left Neuron: " << j << ", Current Right Neuron: " << data->currRightNeuron << endl;
        data->layers[data->currLayer].neurons[data->currRightNeuron].activation +=
            (data->layers[data->prevLayer].neurons[j].activation * data->weights[j * data->layers[data->currLayer].totalNeurons + data->currRightNeuron]);

        // cout << "\nLeft Neuron: " << data->layers[data->prevLayer].neurons[j].activation << ", Right Neuron: " << data->layers[data->currLayer].neurons[data->currRightNeuron].activation << ", Weight b\\w: " << data->weights[j * data->layers[data->currLayer].totalNeurons + data->currRightNeuron] << endl;
    }

    data->layers[data->currLayer].neurons[data->currRightNeuron].activation +=
        data->layers[data->currLayer].neurons[data->currRightNeuron].bias;

    data->layers[data->currLayer].neurons[data->currRightNeuron].activation =
        sigmoid(data->layers[data->currLayer].neurons[data->currRightNeuron].activation);

    cout << "Neuron " << data->currRightNeuron + 1 << " [" << data->layers[data->currLayer].neurons[data->currRightNeuron].activation << "] " << endl;

    pthread_mutex_unlock(&mutex);

    pthread_exit(NULL);
}

void inputLayer(Layer layers[], int currLayer, double args[INPUT_NEURONS], int feedForward[2])
{
    int leftNeurons = layers[currLayer].totalNeurons;
    int rightNeurons = layers[currLayer + 1].totalNeurons;
    int weightsSize = (leftNeurons * rightNeurons);

    // cout << "\nWeight Size:" << weightsSize << endl;
    double *weights = new double[weightsSize];
    cout << "\nLayer 1 Neurons:\n";

    for (int i = 0; i < INPUT_NEURONS; i++)
    {
        cout << "Neuron " << i + 1 << " [" << args[i] << "]" << endl;
        layers[currLayer].neurons[i].activation = args[i];
    }

    readWeights(currLayer, weights);
    sendWeights(currLayer, weights, weightsSize, feedForward);

    delete[] weights;
}

void hiddenLayer(Layer layers[], int currLayer, int feedForward1[2], int feedForward2[2])
{
    int leftNeurons = layers[currLayer - 1].totalNeurons;
    int rightNeurons = layers[currLayer].totalNeurons;
    int weightsSize = (leftNeurons * rightNeurons);

    double *weights = new double[weightsSize];

    ThreadData d[HIDDEN_NEURONS];

    receiveWeights(currLayer - 1, weights, weightsSize, feedForward1);

    pthread_t tid[HIDDEN_NEURONS];

    cout << "\nLayer 2 Neurons:\n";

    for (int i = 0; i < rightNeurons; i++)
    {
        d[i].weights = weights;
        d[i].currLayer = currLayer;
        d[i].prevLayer = currLayer - 1;
        d[i].layers = layers;
        d[i].currRightNeuron = i;

        pthread_create(&tid[i], NULL, processNeuron, (void *)&d[i]);
    }

    for (int i = 0; i < rightNeurons; i++)
    {
        pthread_join(tid[i], NULL);
    }

    readWeights(currLayer, weights);
    sendWeights(currLayer, weights, weightsSize, feedForward2);

    delete[] weights;
}

void outputLayer(Layer layers[], int currLayer, int feedForward[2], double req[OUTPUT_NEURONS])
{
    int leftNeurons = layers[currLayer - 1].totalNeurons;
    int rightNeurons = layers[currLayer].totalNeurons;
    int weightsSize = (leftNeurons * rightNeurons);

    double *weights = new double[weightsSize];
    double pred[OUTPUT_NEURONS] = {0};
    double cost = 0;
    // double req[OUTPUT_NEURONS] = {0, 0, 0};

    ThreadData d[OUTPUT_NEURONS];

    receiveWeights(currLayer - 1, weights, weightsSize, feedForward);
    fflush(stdin);
    cout << "\nLayer 3 Neurons:\n";

    pthread_t tid[OUTPUT_NEURONS];

    for (int i = 0; i < rightNeurons; i++)
    {
        d[i].weights = weights;
        d[i].currLayer = currLayer;
        d[i].prevLayer = currLayer - 1;
        d[i].layers = layers;
        d[i].currRightNeuron = i;

        pthread_create(&tid[i], NULL, processNeuron, (void *)&d[i]);
    }

    for (int i = 0; i < rightNeurons; i++)
    {
        pthread_join(tid[i], NULL);
    }

    softmax(layers, currLayer, pred);

    cout << "\nPredicted: " << endl;
    for (int i = 0; i < rightNeurons; i++)
    {
        cout << "Neuron " << i + 1 << " [" << pred[i] << "] ";
    }
    cout << endl;

    // req[(int)reqAnswer - 1] = 1;

    calculateCost(pred, req, cost);
    cout << "Cost: " << cost << endl;

    delete[] weights;
}

void propagateBackOutput(Layer layers[], int currLayer)
{
    // double *weights = new double[OUTPUT_NEURONS * HIDDEN_NEURONS];

    // readWeights(currLayer - 1, weights);

    // for (int i = 0; i < OUTPUT_NEURONS; i++)
    // {
    //     for (int j = 0; j < HIDDEN_NEURONS; j++)
    //     {
    //         delta(i, j);
    //     }
    // }
}

void propagateBackHidden(Layer layers[], int currLayer)
{
    // double *weights = new double[HIDDEN_NEURONS * INPUT_NEURONS];

    // readWeights(currLayer - 1, weights);

    // for (int i = 0; i < HIDDEN_NEURONS; i++)
    // {
    //     for (int j = 0; j < INPUT_NEURONS; j++)
    //     {
    //         delta(i, j);
    //     }
    // }
}

void processBatch(double batchData[MAX_ROWS][5], int batchSize, Layer layers[])
{
    key_t key = ftok("neuralnet_v2.cpp", 0);

    int shmid = shmget(key, sizeof(sem_t), 0666 | IPC_CREAT);
    sem_t *procSync = (sem_t *)shmat(shmid, nullptr, 0);

    sem_init(procSync, 1, 0); // Initialize the semaphore in shared memory

    int feedForward1[2];
    int feedForward2[2];
    pipe(feedForward1);
    pipe(feedForward2);

    for (int i = 0; i < batchSize; ++i)
    {
        int currLayer = 0;
        double args[INPUT_NEURONS];
        double reqAnswer[OUTPUT_NEURONS] = {0};
        reqAnswer[(int)batchData[i][4]] = 1;

        for (int j = 0; j < INPUT_NEURONS; j++)
            args[j] = batchData[i][j];

        pid_t fork1 = fork();

        if (fork1 == 0)
        {
            sem_post(procSync); // Signal the next process
            sem_wait(procSync); // Wait for the signal to proceed

            inputLayer(layers, currLayer, args, feedForward1);
            currLayer++;

            pid_t fork2 = fork();

            if (fork2 == 0)
            {
                sem_post(procSync);
                sem_wait(procSync);

                hiddenLayer(layers, currLayer, feedForward1, feedForward2);
                currLayer++;

                pid_t fork3 = fork();

                if (fork3 == 0)
                {
                    sem_post(procSync);
                    sem_wait(procSync);

                    outputLayer(layers, currLayer, feedForward2, reqAnswer);
                    // currLayer++;

                    exit(1);
                }
                else if (fork3 > 0)
                {
                    wait(NULL);
                    sem_post(procSync); // Signal the next process
                    exit(1);
                }
                exit(1);
            }
            else if (fork2 > 0)
            {
                wait(NULL);
                sem_post(procSync); // Signal the next process
                propagateBackOutput(layers, currLayer);
                currLayer--;

                exit(1);
            }
            exit(1);
        }
        else if (fork1 > 0)
        {
            wait(NULL);
            sem_post(procSync); // Signal the next process
            propagateBackHidden(layers, currLayer);
            currLayer--;

            // exit(1);
        }
    }

    // Cleanup: Detach shared memory
    shmdt(procSync);
    shmctl(shmid, IPC_RMID, nullptr);
}

void processData(double inputData[MAX_ROWS][5], int numRows, Layer layers[])
{
    int batchSize = BATCH_SIZE; // You can adjust the batch size

    for (int i = 0; i < numRows; i += batchSize)
    {
        double batchData[batchSize][5];
        for (int j = 0; j < batchSize; ++j)
        {
            if (i + j < numRows)
            {
                for (int k = 0; k < 5; ++k)
                {
                    batchData[j][k] = inputData[i + j][k];
                }
            }
        }
        processBatch(batchData, batchSize, layers);
        cout << "BATCH " << i << endl;
    }
}

void getData(double inputData[MAX_ROWS][5], int &numRows)
{
    ifstream inputFile(inputFilename);

    if (!inputFile.is_open())
    {
        cerr << "Error opening file: " << inputFilename << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    numRows = 0;
    while (getline(inputFile, line) && numRows < MAX_ROWS)
    {
        if (numRows == 0)
        {
            // Skip the header line
            ++numRows;
            continue;
        }

        istringstream ss(line);
        string value;

        // Read the numerical values (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
        for (int i = 0; i <= 5; ++i)
        {
            if (getline(ss, value, ','))
            {
                try
                {
                    if (i > 0 && i < 5)
                    {
                        inputData[numRows - 1][i - 1] = stof(value); // Subtract 1 to adjust for skipping the header
                    }
                    else if (i == 5)
                    {
                        if (value == "iris-setoso")
                        {
                            inputData[numRows - 1][i - 1] = 1;
                        }
                        else if (value == "Iris-versicolor")
                        {
                            inputData[numRows - 1][i - 1] = 2;
                        }
                        else
                        {
                            inputData[numRows - 1][i - 1] = 3;
                        }
                    }
                }
                catch (const std::invalid_argument &e)
                {
                    cerr << "Error converting string to double: " << e.what() << endl;
                    cerr << "Invalid data in line: " << line << endl;
                    exit(EXIT_FAILURE);
                }
            }
            else
            {
                cerr << "Error reading data from line: " << line << endl;
                exit(EXIT_FAILURE);
            }
        }

        ++numRows;
    }

    inputFile.close();
}

int main()
{
    cout.unsetf(ios::floatfield);

    cout << setprecision(5);

    Layer layers[TOTAL_LAYERS];

    int numRows;
    double inputData[MAX_ROWS][5];

    getData(inputData, numRows);

    initNetwork(layers);

    processData(inputData, numRows, layers);

    pthread_mutex_destroy(&mutex);

    return 0;
}