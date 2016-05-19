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
	Config conf(configPath);//���߶������Config ��ȡ�����е�������Ϣ������cout���
	cout << conf << endl;
	
	if (conf.features.size() == 0)
	{
		cout << "error: no features specified in config" << endl;
		return EXIT_FAILURE;
	}
	
	ofstream outFile;//����һ������ļ�����������
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
	bool useCamera = (conf.sequenceName == "");//������config.txt���Ƿ������Ƶ���ƣ��ж��Ƿ�ʹ������ͷ
	
	VideoCapture cap;
	
	int startFrame = -1;
	int endFrame = -1;
	FloatRect initBB;//����һ��ģ���࣬
	string imgFormat;
	float scaleW = 1.f;
	float scaleH = 1.f;
	
	if (useCamera)//ʹ������ͷ
	{
		if (!cap.open(0))
		{
			cout << "error: could not start camera capture" << endl;
			return EXIT_FAILURE;
		}
		startFrame = 0;
		endFrame = INT_MAX;
		Mat tmp;
		cap >> tmp;//����һ֡��Ƶ
		scaleW = (float)conf.frameWidth/tmp.cols;//config�п�/������Ƶ�Ŀ�����
		scaleH = (float)conf.frameHeight/tmp.rows;

		/*����������Ŀ������죬������ʲô*/
		/*�ú�����������һ�����Σ����Ͻ��ڣ�120,80��,80*80�ľ���*/
		initBB = IntRect(conf.frameWidth/2-kLiveBoxWidth/2, conf.frameHeight/2-kLiveBoxHeight/2, kLiveBoxWidth, kLiveBoxHeight);
		cout << "press 'i' to initialise tracker" << endl;
	}
	else//ʹ����Ƶ
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
	
	
	
	Tracker tracker(conf);//ʹ��conf�࣬��ʼ��Tracker��
	if (!conf.quietMode)//quietModeģʽ�£�����ʾ�����ֻ����
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
			/*���������resize��������Ϊ����ʼ��ʱ��frameֻ��һ���յ��࣬��Mat��Ĭ�Ϲ��캯�����죬dataָ��Ϊnull*/
			/*resize������������ͼƬ�Ķ�̬�ڴ棬Ȼ��ʹdataָ��ָ����Ƭ�ڴ�*/
			/*��Ȼ�����frame����data����ָ����ڴ棬resize�Ͳ����ٷ��䶯̬�ڴ���*/
			/*��������������һ���ǳ���Ч����Ϊ��ÿ�ν���forѭ������Ҫ����ΪMat frame�����ڴ棬
			���Խ��齫frame����forѭ����������*/
			/*��ô�Ƿ�ᷢ���ڴ�й¶�أ�ÿ��frame data��ָ��ָ��Ķ�̬�ڴ棬��Mat�뿪������󣬲�û�б���ʾ�ͷţ�
			���Ҫ��Mat��Ĭ��������������β�����*/
			/*��Ȼ��opencv������߿϶��ῼ���������ģ���Ʒ������Բο�c++ primer�Ŀ������ƣ�������Ϊ��ָ�����*/
			/*Matʹ�õ������ü����ķ�ʽ����c++primer���潲�ıȽ����*/
			resize(frameOrig, frame, Size(conf.frameWidth, conf.frameHeight));
			//imshow("result",frame);//qyy
			//waitKey(0);//qyy
			flip(frame, frame, 1);//���߰���Ƶ���ҶԳƷ�ת�ˣ���֪��Ϊʲô��ô����
			//imshow("result", frame);//qyy
			//waitKey(0);//qyy
			frame.copyTo(result);
			if (doInitialise)
			{
				if (tracker.IsInitialised())
				{
					tracker.Reset();
				}
				else//������߼���ʵ��Щ���⣬�����ʼ�����ˣ��û�Ҫ��2�Σ��Ż���������ʼ������
				{
					tracker.Initialise(frame, initBB);
				}
				doInitialise = false;
			}
			else if (!tracker.IsInitialised())
			{
				rectangle(result, initBB, CV_RGB(255, 255, 255));//û�г�ʼ��������result�ϻ���ɫ���
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
		
		if (tracker.IsInitialised())//�����ʼ���ˣ��Ϳ�ʼ����
		{
			tracker.Track(frame);//���ٳ��򣬰�tracker����һ�������Դ������������˰�����һ�����㷨����������ʵ��
			
			if (!conf.quietMode && conf.debugMode)
			{
				tracker.Debug();//debugģʽ�£����Կ����ܶ����Ĵ�����ʾ
			}
			
			rectangle(result, tracker.GetBB(), CV_RGB(0, 255, 0));//ʹ����ɫ�򣬻������ٵ�Ч��
			
			if (outFile)//�����ǵõ��ľ��ο򣬴洢��txt�ı���
			{
				const FloatRect& bb = tracker.GetBB();
				outFile << bb.XMin()/scaleW << "," << bb.YMin()/scaleH << "," << bb.Width()/scaleW << "," << bb.Height()/scaleH << endl;
			}
		}
		
		if (!conf.quietMode)//����Ǿ�Ĭģʽ�Ļ��������޷���ʼ����
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
