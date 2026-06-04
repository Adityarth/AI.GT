//////////////////////////////////////////////////////////////
// C++ Vehicle Detection and Vehicle Counting using YOLOv8 TensorRT + CUDA + OpenCV
//////////////////////////////////////////////////////////////

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/dnn.hpp>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace cv;
using namespace nvinfer1;

//////////////////////////////////////////////////////////////
// LOGGER
//////////////////////////////////////////////////////////////

class Logger : public ILogger
{
public:

    void log( Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            cout << msg << endl;
        }
    }
} logger;

//////////////////////////////////////////////////////////////
// CLASSES
//////////////////////////////////////////////////////////////

unordered_map<int, string> classNames =
{
    {1, "bicycle"},
    {2, "car"},
    {3, "motorcycle"},
    {4, "bus"},
    {5, "truck"},
    {0, "person"}
};

//////////////////////////////////////////////////////////////
// TRACK STRUCT
//////////////////////////////////////////////////////////////

struct Track
{
    int id = -1;

    int class_id = -1;

    Rect box;

    Point center;

    bool counted = false;
};

//////////////////////////////////////////////////////////////
// IOU
//////////////////////////////////////////////////////////////

float computeIOU(
    const Rect& a,
    const Rect& b)
{
    int x1 = max(a.x, b.x);

    int y1 = max(a.y, b.y);

    int x2 = min( a.x + a.width, b.x + b.width);

    int y2 = min( a.y + a.height, b.y + b.height);

    int intersection = max(0, x2 - x1) * max(0, y2 - y1);

    int unionArea = a.area() + b.area() - intersection;

    if (unionArea <= 0)
        return 0.0f;

    return (float)intersection / unionArea;
}

//////////////////////////////////////////////////////////////
// LOAD ENGINE
//////////////////////////////////////////////////////////////

vector<char> loadEngine( const string& filename)
{
    ifstream file( filename, ios::binary);

    if (!file.good())
    {
        throw runtime_error( "Engine file not found!");
    }

    file.seekg(0, ios::end);

    size_t size = file.tellg();

    file.seekg(0, ios::beg);

    vector<char> buffer(size);

    file.read( buffer.data(), size);

    file.close();

    return buffer;
}

//////////////////////////////////////////////////////////////
// MAIN
//////////////////////////////////////////////////////////////

int main()
{
    //////////////////////////////////////////////////////////
    // LOAD ENGINE
    //////////////////////////////////////////////////////////
    string EngineName;
    cout << "\nEnter Engine Name : ";
    cin >> EngineName;
    cout << "\n!Done" << endl;

    vector<char> engineData = loadEngine(EngineName);

    IRuntime* runtime = createInferRuntime(logger);

    ICudaEngine* engine = runtime->deserializeCudaEngine( engineData.data(), engineData.size());

    IExecutionContext* context = engine->createExecutionContext();

    //////////////////////////////////////////////////////////
    // CUDA STREAM
    //////////////////////////////////////////////////////////

    cudaStream_t stream;

    cudaStreamCreate(&stream);

    //////////////////////////////////////////////////////////
    // MODEL SETTINGS
    //////////////////////////////////////////////////////////

    const int INPUT_W = 640;

    const int INPUT_H = 640;

    const int NUM_CLASSES = 80;

    const int NUM_BOXES = 8400;

    //////////////////////////////////////////////////////////
    // MEMORY SIZE
    //////////////////////////////////////////////////////////

    const int inputSize = 3 * INPUT_W * INPUT_H * sizeof(float);

    const int outputSize = 84 * NUM_BOXES * sizeof(float);

    //////////////////////////////////////////////////////////
    // GPU MEMORY
    //////////////////////////////////////////////////////////

    float* gpuInput = nullptr;

    float* gpuOutput = nullptr;

    cudaMalloc( (void**)&gpuInput, inputSize);

    cudaMalloc( (void**)&gpuOutput, outputSize);

    //////////////////////////////////////////////////////////
    // CPU OUTPUT
    //////////////////////////////////////////////////////////

    float* cpuOutput = nullptr;

    cudaMallocHost( (void**)&cpuOutput, outputSize);

    //////////////////////////////////////////////////////////
    // BINDINGS
    //////////////////////////////////////////////////////////

    int inputIndex = engine->getBindingIndex("images");

    int outputIndex = engine->getBindingIndex("output0");

    void* bindings[2];

    bindings[inputIndex] = gpuInput;

    bindings[outputIndex] = gpuOutput;

    //////////////////////////////////////////////////////////
    // VIDEO
    //////////////////////////////////////////////////////////

    string VideoPath;
    cout << "\nEnter Video Path : ";
    cin >> VideoPath;
    cout << "\n!Done" << endl;

    VideoCapture cap(VideoPath);

    // Webcam:
    // VideoCapture cap(0);

    if (!cap.isOpened())
    {
        cout << "Cannot open video!" << endl;

        return -1;
    }

    //////////////////////////////////////////////////////////
    // VIDEO INFO
    //////////////////////////////////////////////////////////

    int frameWidth = static_cast<int>( cap.get(CAP_PROP_FRAME_WIDTH));

    int frameHeight = static_cast<int>( cap.get(CAP_PROP_FRAME_HEIGHT));

    double videoFPS = cap.get(CAP_PROP_FPS);

    //////////////////////////////////////////////////////////
    // OUTPUT VIDEO
    //////////////////////////////////////////////////////////
    string OutputName;
    cout << "Enter Video Output Name: ";
    cin >> OutputName;
    cout << "\n!Done" << endl;

    VideoWriter writer( OutputName, VideoWriter::fourcc( 'm', 'p', '4', 'v'), videoFPS, Size( frameWidth, frameHeight));

    //////////////////////////////////////////////////////////
    // TRACKING
    //////////////////////////////////////////////////////////

    vector<Track> tracks;

    int nextTrackID = 0;

    //////////////////////////////////////////////////////////
    // COUNTS
    //////////////////////////////////////////////////////////

    int totalCount = 0;

    //////////////////////////////////////////////////////////
    // COUNTING LINE
    //////////////////////////////////////////////////////////

    int lineY =
        frameHeight / 2;

    //////////////////////////////////////////////////////////
    // GPU MATS
    //////////////////////////////////////////////////////////

    cuda::GpuMat gpuFrame;

    cuda::GpuMat gpuResized;

    cuda::GpuMat gpuRGB;

    cuda::GpuMat gpuFloat;

    //////////////////////////////////////////////////////////
    // MAIN LOOP
    //////////////////////////////////////////////////////////

    while (true)
    {
        //////////////////////////////////////////////////////
        // READ FRAME
        //////////////////////////////////////////////////////

        Mat frame;

        cap >> frame;

        if (frame.empty())
            break;

        //////////////////////////////////////////////////////
        // FPS TIMER
        //////////////////////////////////////////////////////

        int64 start = getTickCount();

        //////////////////////////////////////////////////////
        // UPLOAD TO GPU
        //////////////////////////////////////////////////////

        gpuFrame.upload(frame);

        //////////////////////////////////////////////////////
        // GPU RESIZE
        //////////////////////////////////////////////////////

        cuda::resize( gpuFrame, gpuResized, Size( INPUT_W, INPUT_H));

        //////////////////////////////////////////////////////
        // GPU COLOR CONVERSION
        //////////////////////////////////////////////////////

        cuda::cvtColor( gpuResized, gpuRGB, COLOR_BGR2RGB);

        //////////////////////////////////////////////////////
        // GPU FLOAT NORMALIZE
        //////////////////////////////////////////////////////

        gpuRGB.convertTo( gpuFloat, CV_32F, 1.0 / 255.0);

        //////////////////////////////////////////////////////
        // DOWNLOAD TEMP
        //////////////////////////////////////////////////////

        Mat tempCPU;

        gpuFloat.download(tempCPU);

        //////////////////////////////////////////////////////
        // SPLIT CHANNELS
        //////////////////////////////////////////////////////

        vector<Mat> cpuChannels;

        split( tempCPU, cpuChannels);

        //////////////////////////////////////////////////////
        // COPY TO GPU INPUT
        //////////////////////////////////////////////////////

        size_t channelSize = INPUT_W * INPUT_H * sizeof(float);

        cudaMemcpyAsync( gpuInput, cpuChannels[0].ptr<float>(), channelSize, cudaMemcpyHostToDevice, stream);

        cudaMemcpyAsync( gpuInput + INPUT_W * INPUT_H, cpuChannels[1].ptr<float>(), channelSize, cudaMemcpyHostToDevice, stream);

        cudaMemcpyAsync( gpuInput + 2 * INPUT_W * INPUT_H, cpuChannels[2].ptr<float>(), channelSize, cudaMemcpyHostToDevice, stream);

        //////////////////////////////////////////////////////
        // INFERENCE
        //////////////////////////////////////////////////////

        context->enqueueV2( bindings, stream, nullptr);

        //////////////////////////////////////////////////////
        // COPY OUTPUT
        //////////////////////////////////////////////////////

        cudaMemcpyAsync( cpuOutput, gpuOutput, outputSize, cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        //////////////////////////////////////////////////////
        // DETECTIONS
        //////////////////////////////////////////////////////

        vector<Rect> boxes;

        vector<float> confidences;

        vector<int> class_ids;

        //////////////////////////////////////////////////////
        // PARSE OUTPUT
        //////////////////////////////////////////////////////

        for (int i = 0; i < NUM_BOXES; i++)
        {
            float cx = cpuOutput[ 0 * NUM_BOXES + i];

            float cy = cpuOutput[ 1 * NUM_BOXES + i];

            float w = cpuOutput[ 2 * NUM_BOXES + i];

            float h = cpuOutput[ 3 * NUM_BOXES + i];

            //////////////////////////////////////////////////
            // CLASS SCORE
            //////////////////////////////////////////////////

            int class_id = -1;

            float maxScore = 0.0f;

            for (int c = 0; c < NUM_CLASSES; c++)
            {
                float score = cpuOutput[ (4 + c) * NUM_BOXES + i];

                if (score > maxScore)
                {
                    maxScore = score;

                    class_id = c;
                }
            }

            //////////////////////////////////////////////////
            // FILTER
            //////////////////////////////////////////////////

            if (maxScore < 0.4f)
                continue;

            if (classNames.find(class_id) == classNames.end())
            {
                continue;
            }

            //////////////////////////////////////////////////
            // SCALE BOX
            //////////////////////////////////////////////////

            int left = static_cast<int>( (cx - 0.5f * w) * frame.cols / INPUT_W);

            int top = static_cast<int>( (cy - 0.5f * h) * frame.rows / INPUT_H);

            int width = static_cast<int>( w * frame.cols / INPUT_W);

            int height = static_cast<int>( h * frame.rows / INPUT_H);

            Rect box( left, top, width, height);

            boxes.push_back(box);

            confidences.push_back(maxScore);

            class_ids.push_back(class_id);
        }

        //////////////////////////////////////////////////////
        // NMS
        //////////////////////////////////////////////////////

        vector<int> indices;

        dnn::NMSBoxes( boxes, confidences, 0.4f, 0.5f, indices);

        //////////////////////////////////////////////////////
        // TRACKING
        //////////////////////////////////////////////////////

        vector<Track> updatedTracks;

        for (int idx : indices)
        {
            Rect box = boxes[idx];

            Point center( box.x + box.width / 2, box.y + box.height / 2);

            bool matched = false;

            //////////////////////////////////////////////////
            // MATCH EXISTING TRACKS
            //////////////////////////////////////////////////

            for (auto& track : tracks)
            {
                float iou = computeIOU( box, track.box);

                if (iou > 0.3f)
                {
                    track.box = box;

                    track.center = center;

                    track.class_id = class_ids[idx];

                    //////////////////////////////////////////////////
                    // COUNT
                    //////////////////////////////////////////////////

                    if (!track.counted && center.y >= lineY - 5 && center.y <= lineY + 5)
                    {
                        track.counted = true;

                        totalCount++;
                    }

                    updatedTracks.push_back(track);

                    matched = true;

                    break;
                }
            }

            //////////////////////////////////////////////////
            // NEW TRACK
            //////////////////////////////////////////////////

            if (!matched)
            {
                Track track;

                track.id = nextTrackID++;

                track.class_id = class_ids[idx];

                track.box = box;

                track.center = center;

                updatedTracks.push_back(track);
            }
        }

        tracks = updatedTracks;

        //////////////////////////////////////////////////////
        // DRAW LINE
        //////////////////////////////////////////////////////

        line( frame, Point(0, lineY), Point(frame.cols, lineY), Scalar(255, 0, 255), 2);

        //////////////////////////////////////////////////////
        // DRAW TRACKS
        //////////////////////////////////////////////////////

        for (auto& track : tracks)
        {
            rectangle( frame, track.box, Scalar(0, 255, 0), 2);

            circle( frame, track.center, 4, Scalar(0, 0, 255), -1);

            //////////////////////////////////////////////////
            // LABEL
            //////////////////////////////////////////////////

            stringstream ss;

            ss << classNames[ track.class_id] << " ID:" << track.id;

            putText( frame, ss.str(), Point( track.box.x, track.box.y - 10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
        }

        //////////////////////////////////////////////////////
        // FPS
        //////////////////////////////////////////////////////

        double fps = getTickFrequency() / (getTickCount() - start);

        stringstream fpsText;

        fpsText << "FPS: " << fixed << setprecision(1) << fps;

        putText( frame, fpsText.str(), Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 2);

        //////////////////////////////////////////////////////
        // TOTAL COUNT
        //////////////////////////////////////////////////////

        putText(frame, "Total Count: " + to_string(totalCount), Point(20, 80), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 0, 255), 2);

        //////////////////////////////////////////////////////
        // SHOW
        //////////////////////////////////////////////////////

        imshow("YOLOv8 TensorRT",frame);

        //////////////////////////////////////////////////////
        // SAVE OUTPUT
        //////////////////////////////////////////////////////
        writer.write(frame);

        //////////////////////////////////////////////////////
        // EXIT
        //////////////////////////////////////////////////////
        if (waitKey(1) == 27)
            break;
    }
  
    //////////////////////////////////////////////////////////
    // CLEANUP
    //////////////////////////////////////////////////////////
    writer.release();
    cap.release();
    cudaFree(gpuInput);
    cudaFree(gpuOutput);
    cudaFreeHost(cpuOutput);
    cudaStreamDestroy(stream);
    context->destroy();
    engine->destroy();
    runtime->destroy();
    destroyAllWindows();
    return 0;
}
