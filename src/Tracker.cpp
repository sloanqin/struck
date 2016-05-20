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
#include "ImageRep.h"
#include "Sampler.h"
#include "Sample.h"
#include "GraphUtils.h"

#include "HaarFeatures.h"
#include "RawFeatures.h"
#include "HistogramFeatures.h"
#include "MultiFeatures.h"

#include "Kernels.h"

#include "LaRank.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <Eigen/Core>

#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace Eigen;

Tracker::Tracker(const Config& conf) :
	m_config(conf),
	m_initialised(false),
	m_pLearner(0),
	m_debugImage(2*conf.searchRadius+1, 2*conf.searchRadius+1, CV_32FC1),
	m_needsIntegralImage(false)
{
	Reset();
}

Tracker::~Tracker()
{
	delete m_pLearner;
	for (int i = 0; i < (int)m_features.size(); ++i)
	{
		delete m_features[i];
		delete m_kernels[i];
	}
}

void Tracker::Reset()
{
	m_initialised = false;
	m_debugImage.setTo(0);
	if (m_pLearner) delete m_pLearner;
	for (int i = 0; i < (int)m_features.size(); ++i)
	{
		delete m_features[i];
		delete m_kernels[i];
	}
	m_features.clear();
	m_kernels.clear();
	
	m_needsIntegralImage = false;
	m_needsIntegralHist = false;
	
	int numFeatures = m_config.features.size();
	vector<int> featureCounts;
	for (int i = 0; i < numFeatures; ++i)
	{
		switch (m_config.features[i].feature)
		{
		case Config::kFeatureTypeHaar:
			m_features.push_back(new HaarFeatures(m_config));
			m_needsIntegralImage = true;
			break;			
		case Config::kFeatureTypeRaw:
			m_features.push_back(new RawFeatures(m_config));
			break;
		case Config::kFeatureTypeHistogram:
			m_features.push_back(new HistogramFeatures(m_config));
			m_needsIntegralHist = true;
			break;
		}
		featureCounts.push_back(m_features.back()->GetCount());
		
		switch (m_config.features[i].kernel)
		{
		case Config::kKernelTypeLinear:
			m_kernels.push_back(new LinearKernel());
			break;
		case Config::kKernelTypeGaussian:
			m_kernels.push_back(new GaussianKernel(m_config.features[i].params[0]));
			break;
		case Config::kKernelTypeIntersection:
			m_kernels.push_back(new IntersectionKernel());
			break;
		case Config::kKernelTypeChi2:
			m_kernels.push_back(new Chi2Kernel());
			break;
		}
	}
	
	if (numFeatures > 1)
	{
		MultiFeatures* f = new MultiFeatures(m_features);
		m_features.push_back(f);
		
		MultiKernel* k = new MultiKernel(m_kernels, featureCounts);
		m_kernels.push_back(k);		
	}
	
	m_pLearner = new LaRank(m_config, *m_features.back(), *m_kernels.back());
}
	

void Tracker::Initialise(const cv::Mat& frame, FloatRect bb)
{
	m_bb = IntRect(bb);//这里创造了一个临时int矩形变量，然后使用合成拷贝函数给了m_bb
	ImageRep image(frame, m_needsIntegralImage, m_needsIntegralHist);
	for (int i = 0; i < 1; ++i)//?这里用个for循环，循环1次干嘛？
	{
		UpdateLearner(image);
	}
	m_initialised = true;
}

void Tracker::Track(const cv::Mat& frame)
{
	assert(m_initialised);
	//计算积分图，若选择haar特征，m_needsIntegralImage=true，m_needsIntegralHist=false
	//计算积分图是为了方便计算haar特征
	ImageRep image(frame, m_needsIntegralImage, m_needsIntegralHist);
	//在上一帧矩形框的searchRadius范围内，采样n个矩形框，存在vector中
	vector<FloatRect> rects = Sampler::PixelSamples(m_bb, m_config.searchRadius);
	
	vector<FloatRect> keptRects;//存储筛选后的框
	keptRects.reserve(rects.size());
	for (int i = 0; i < (int)rects.size(); ++i)
	{
		if (!rects[i].IsInside(image.GetRect())) continue;//超出图像范围的框，被舍弃掉
		keptRects.push_back(rects[i]);
	}
	
	//使用image和矩形框，创建一个多样本类，送给后面分类
	//这里的多样本类，指的是类标签中有正样本和负样本
	MultiSample sample(image, keptRects);
	
	vector<double> scores;
	//这一句很耗时，对样本进行分类，分类输出的结果存储在vector scores中
	//所以这一句的实现，是作者算法的核心思想
	m_pLearner->Eval(sample, scores);//将sample里面的每个样本计算score存在scores这个vector中
	
	double bestScore = -DBL_MAX;
	int bestInd = -1;
	for (int i = 0; i < (int)keptRects.size(); ++i)
	{		
		if (scores[i] > bestScore)
		{
			bestScore = scores[i];//找到置信度最高的样本，记录分数和编号
			bestInd = i;
		}
	}
	
	UpdateDebugImage(keptRects, m_bb, scores);//在上一帧的m_bb周围，根据score的大小，画不同颜色的点
	
	if (bestInd != -1)
	{
		m_bb = keptRects[bestInd];
		UpdateLearner(image);//这一句也比较耗时，更新分类器
#if VERBOSE		
		cout << "track score: " << bestScore << endl;
#endif
	}
}

void Tracker::UpdateDebugImage(const vector<FloatRect>& samples, const FloatRect& centre, const vector<double>& scores)
{
	double mn = VectorXd::Map(&scores[0], scores.size()).minCoeff();//使用Eigen的Map静态方法，创建了一个临时对象
	double mx = VectorXd::Map(&scores[0], scores.size()).maxCoeff();//使用该临时对象的成员函数，得到了最大值和最小值
	m_debugImage.setTo(0);//把图片像素点全部设置为0
	for (int i = 0; i < (int)samples.size(); ++i)
	{
		int x = (int)(samples[i].XMin() - centre.XMin());//qyy，计算左上角之间的距离，为什么不计算矩形中心点的距离？
		int y = (int)(samples[i].YMin() - centre.YMin());
		//根据score分数的高低，在image上显示不同的颜色，问题是这个颜色归一化到0-1了，应该是跟float的特性相符吧？？
		m_debugImage.at<float>(m_config.searchRadius+y, m_config.searchRadius+x) = (float)((scores[i]-mn)/(mx-mn));
	}
}

void Tracker::Debug()
{
	imshow("tracker", m_debugImage);//显示出刚才在UpdateDebugImage里更新的m_debugImage
	m_pLearner->Debug();
}

void Tracker::UpdateLearner(const ImageRep& image)
{
	// note these return the centre sample at index 0
	vector<FloatRect> rects = Sampler::RadialSamples(m_bb, 2*m_config.searchRadius, 5, 16);
	//vector<FloatRect> rects = Sampler::PixelSamples(m_bb, 2*m_config.searchRadius, true);
	
	vector<FloatRect> keptRects;
	keptRects.push_back(rects[0]); // the true sample，第0个rect是真值
	for (int i = 1; i < (int)rects.size(); ++i)
	{
		if (!rects[i].IsInside(image.GetRect())) continue;//超出图像范围的框，被舍弃掉
		keptRects.push_back(rects[i]);
	}
		
#if VERBOSE		
	cout << keptRects.size() << " samples" << endl;
#endif
		
	MultiSample sample(image, keptRects);
	m_pLearner->Update(sample, 0);//更新larank分类器
}
