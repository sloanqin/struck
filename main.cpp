/* 
 * Struck: Structured Output Tracking with Kernels
 * 
 * Code to accompany the paper:
 *   Struck: Structured Output Tracking with Kernels
 *   Sam Hare, Amir Saffari, Philip H. S. Torr
 *   International Conference on Computer Vision (ICCV), 2011
 * 
 * Copyright (C) 2011 Sam Hare, Oxford Brookes University, Oxford, UK
 * 
 * This file is part of Struck.
 * 
 * Struck is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Struck is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Struck.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */
 
#include "Tracker.h"
#include "Config.h"

#include <iostream>
#include <fstream>

#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;
using namespace cv;

static const int kLiveBoxWidth = 80;
static const int kLiveBoxHeight = 80;

void rectangle(Mat& rMat, const FloatRect& rRect, const Scalar& rColour)
{
	IntRect r(rRect);
	rectangle(rMat, Point(r.XMin(), r.YMin()), Point(r.XMax(), r.YMax()), rColour);
}

int main(int argc, char* argv[])
{
	// read config file
	string configPath = "config.txt";
	if (argc > 1)
	{
		configPath = argv[1];
	}
	Config conf(configPath);//作者定义的类Config 读取了所有的配置信息，并且cout输出
	cout << conf << endl;
	
	if (conf.features.size() == 0)
	{
		cout << "error: no features specified in config" << endl;
		return EXIT_FAILURE;
	}
	
	ofstream outFile;//定义一个输出文件流，输出结果
	if (conf.resultsPath != "")
	{
		outFile.open(conf.resultsPath.c_str(), ios::out);
		if (!outFile)
		{
			cout << "error: could not open results file: " << conf.resultsPath << endl;
			return EXIT_FAILURE;
		}
	}
	
	// if no sequence specified then use the camera
	bool useCamera = (conf.sequenceName == "");//根据在config.txt中是否给出视频名称，判断是否使用摄像头
	
	VideoCapture cap;
	
	int startFrame = -1;
	int endFrame = -1;
	FloatRect initBB;//这是一个模板类，
	string imgFormat;
	float scaleW = 1.f;
	float scaleH = 1.f;
	
	if (useCamera)//使用摄像头
	{
		if (!cap.open(0))
		{
			cout << "error: could not start camera capture" << endl;
			return EXIT_FAILURE;
		}
		startFrame = 0;
		endFrame = INT_MAX;
		Mat tmp;
		cap >> tmp;//读入一帧视频
		scaleW = (float)conf.frameWidth/tmp.cols;//config中宽/读入视频的宽，比率
		scaleH = (float)conf.frameHeight/tmp.rows;

		/*分析下这里的拷贝构造，发生了什么*/
		/*该函数，创造了一个矩形，左上角在（120,80）,80*80的矩形*/
		initBB = IntRect(conf.frameWidth/2-kLiveBoxWidth/2, conf.frameHeight/2-kLiveBoxHeight/2, kLiveBoxWidth, kLiveBoxHeight);
		cout << "press 'i' to initialise tracker" << endl;
	}
	else//使用视频
	{
		// parse frames file
		string framesFilePath = conf.sequenceBasePath+"/"+conf.sequenceName+"/"+conf.sequenceName+"_frames.txt";
		ifstream framesFile(framesFilePath.c_str(), ios::in);
		if (!framesFile)
		{
			cout << "error: could not open sequence frames file: " << framesFilePath << endl;
			return EXIT_FAILURE;
		}
		string framesLine;
		getline(framesFile, framesLine);
		sscanf(framesLine.c_str(), "%d,%d", &startFrame, &endFrame);
		if (framesFile.fail() || startFrame == -1 || endFrame == -1)
		{
			cout << "error: could not parse sequence frames file" << endl;
			return EXIT_FAILURE;
		}
		
		imgFormat = conf.sequenceBasePath+"/"+conf.sequenceName+"/imgs/img%05d.png";
		
		// read first frame to get size
		char imgPath[256];
		sprintf(imgPath, imgFormat.c_str(), startFrame);
		Mat tmp = cv::imread(imgPath, 0);
		scaleW = (float)conf.frameWidth/tmp.cols;
		scaleH = (float)conf.frameHeight/tmp.rows;
		
		// read init box from ground truth file
		string gtFilePath = conf.sequenceBasePath+"/"+conf.sequenceName+"/"+conf.sequenceName+"_gt.txt";
		ifstream gtFile(gtFilePath.c_str(), ios::in);
		if (!gtFile)
		{
			cout << "error: could not open sequence gt file: " << gtFilePath << endl;
			return EXIT_FAILURE;
		}
		string gtLine;
		getline(gtFile, gtLine);
		float xmin = -1.f;
		float ymin = -1.f;
		float width = -1.f;
		float height = -1.f;
		sscanf(gtLine.c_str(), "%f,%f,%f,%f", &xmin, &ymin, &width, &height);
		if (gtFile.fail() || xmin < 0.f || ymin < 0.f || width < 0.f || height < 0.f)
		{
			cout << "error: could not parse sequence gt file" << endl;
			return EXIT_FAILURE;
		}
		initBB = FloatRect(xmin*scaleW, ymin*scaleH, width*scaleW, height*scaleH);
	}
	
	
	
	Tracker tracker(conf);//使用conf类，初始化Tracker类
	if (!conf.quietMode)//quietMode模式下，不显示结果，只运算
	{
		namedWindow("result");
	}
	
	Mat result(conf.frameHeight, conf.frameWidth, CV_8UC3);
	bool paused = false;
	bool doInitialise = false;
	srand(conf.seed);
	for (int frameInd = startFrame; frameInd <= endFrame; ++frameInd)
	{
		cout << "frame num is: " << frameInd << endl;//qyy
		Mat frame;
		if (useCamera)
		{
			Mat frameOrig;
			cap >> frameOrig;
			/*分析下这个resize函数的行为，开始的时候frame只是一个空的类，由Mat的默认构造函数创造，data指针为null*/
			/*resize函数，分配了图片的动态内存，然后使data指针指向这片内存*/
			/*当然，如果frame本来data就有指向的内存，resize就不会再分配动态内存了*/
			/*！！！但是这是一个非常低效的行为，每次进入for循环，都要重新为Mat frame分配内存，
			所以建议将frame放在for循环外面声明*/
			/*那么是否会发生内存泄露呢？每次frame data的指针指向的动态内存，在Mat离开作用域后，并没有被显示释放，
			这就要看Mat的默认析构函数是如何操作了*/
			/*当然，opencv的设计者肯定会考虑这个问题的，设计方法可以参考c++ primer的拷贝控制：定义行为像指针的类*/
			/*Mat使用的是引用计数的方式，在c++primer里面讲的比较清楚*/
			resize(frameOrig, frame, Size(conf.frameWidth, conf.frameHeight));
			//imshow("result",frame);//qyy
			//waitKey(0);//qyy
			flip(frame, frame, 1);//作者把视频左右对称翻转了，不知道为什么这么做？
			//imshow("result", frame);//qyy
			//waitKey(0);//qyy
			frame.copyTo(result);
			if (doInitialise)
			{
				if (tracker.IsInitialised())
				{
					tracker.Reset();
				}
				else//这里的逻辑其实有些问题，如果初始化过了，用户要按2次，才会进入这个初始化程序
				{
					tracker.Initialise(frame, initBB);
				}
				doInitialise = false;
			}
			else if (!tracker.IsInitialised())
			{
				rectangle(result, initBB, CV_RGB(255, 255, 255));//没有初始化，就在result上画白色框框
			}
		}
		else
		{			
			char imgPath[256];
			sprintf(imgPath, imgFormat.c_str(), frameInd);
			Mat frameOrig = cv::imread(imgPath, 0);
			if (frameOrig.empty())
			{
				cout << "error: could not read frame: " << imgPath << endl;
				return EXIT_FAILURE;
			}
			resize(frameOrig, frame, Size(conf.frameWidth, conf.frameHeight));
			cvtColor(frame, result, CV_GRAY2RGB);
		
			if (frameInd == startFrame)
			{
				tracker.Initialise(frame, initBB);
			}
		}
		
		if (tracker.IsInitialised())//如果初始化了，就开始跟踪
		{
			tracker.Track(frame);//跟踪程序，把tracker当做一个类来对待，很清晰明了啊，赞一个；算法都在这里面实现
			
			if (!conf.quietMode && conf.debugMode)
			{
				tracker.Debug();//debug模式下，可以开启很多额外的窗口显示
			}
			
			rectangle(result, tracker.GetBB(), CV_RGB(0, 255, 0));//使用绿色框，画出跟踪的效果
			
			if (outFile)//这里是得到的矩形框，存储到txt文本中
			{
				const FloatRect& bb = tracker.GetBB();
				outFile << bb.XMin()/scaleW << "," << bb.YMin()/scaleH << "," << bb.Width()/scaleW << "," << bb.Height()/scaleH << endl;
			}
		}
		
		if (!conf.quietMode)//如果是静默模式的话，根本无法初始化啊
		{
			imshow("result", result);
			int key = waitKey(paused ? 0 : 1);
			if (key != -1)
			{
				if (key == 27 || key == 113) // esc q
				{
					break;
				}
				else if (key == 112) // p
				{
					paused = !paused;
				}
				else if (key == 105 && useCamera)//i
				{
					doInitialise = true;
					cout << "initialised !" << endl;//qyy
				}
			}
			if (conf.debugMode && frameInd == endFrame)
			{
				cout << "\n\nend of sequence, press any key to exit" << endl;
				waitKey();
			}
		}
	}
	
	if (outFile.is_open())
	{
		outFile.close();
	}
	
	return EXIT_SUCCESS;
}
