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
	m_bb = IntRect(bb);//���ﴴ����һ����ʱint���α�����Ȼ��ʹ�úϳɿ�����������m_bb
	ImageRep image(frame, m_needsIntegralImage, m_needsIntegralHist);
	for (int i = 0; i < 1; ++i)//?�����ø�forѭ����ѭ��1�θ��
	{
		UpdateLearner(image);
	}
	m_initialised = true;
}

void Tracker::Track(const cv::Mat& frame)
{
	assert(m_initialised);
	//�������ͼ����ѡ��haar������m_needsIntegralImage=true��m_needsIntegralHist=false
	//�������ͼ��Ϊ�˷������haar����
	ImageRep image(frame, m_needsIntegralImage, m_needsIntegralHist);
	//����һ֡���ο��searchRadius��Χ�ڣ�����n�����ο򣬴���vector��
	vector<FloatRect> rects = Sampler::PixelSamples(m_bb, m_config.searchRadius);
	
	vector<FloatRect> keptRects;//�洢ɸѡ��Ŀ�
	keptRects.reserve(rects.size());
	for (int i = 0; i < (int)rects.size(); ++i)
	{
		if (!rects[i].IsInside(image.GetRect())) continue;//����ͼ��Χ�Ŀ򣬱�������
		keptRects.push_back(rects[i]);
	}
	
	//ʹ��image�;��ο򣬴���һ���������࣬�͸��������
	//����Ķ������ָ࣬�������ǩ�����������͸�����
	MultiSample sample(image, keptRects);
	
	vector<double> scores;
	//��һ��ܺ�ʱ�����������з��࣬��������Ľ���洢��vector scores��
	//������һ���ʵ�֣��������㷨�ĺ���˼��
	m_pLearner->Eval(sample, scores);//��sample�����ÿ����������score����scores���vector��
	
	double bestScore = -DBL_MAX;
	int bestInd = -1;
	for (int i = 0; i < (int)keptRects.size(); ++i)
	{		
		if (scores[i] > bestScore)
		{
			bestScore = scores[i];//�ҵ����Ŷ���ߵ���������¼�����ͱ��
			bestInd = i;
		}
	}
	
	UpdateDebugImage(keptRects, m_bb, scores);//����һ֡��m_bb��Χ������score�Ĵ�С������ͬ��ɫ�ĵ�
	
	if (bestInd != -1)
	{
		m_bb = keptRects[bestInd];
		UpdateLearner(image);//��һ��Ҳ�ȽϺ�ʱ�����·�����
#if VERBOSE		
		cout << "track score: " << bestScore << endl;
#endif
	}
}

void Tracker::UpdateDebugImage(const vector<FloatRect>& samples, const FloatRect& centre, const vector<double>& scores)
{
	double mn = VectorXd::Map(&scores[0], scores.size()).minCoeff();//ʹ��Eigen��Map��̬������������һ����ʱ����
	double mx = VectorXd::Map(&scores[0], scores.size()).maxCoeff();//ʹ�ø���ʱ����ĳ�Ա�������õ������ֵ����Сֵ
	m_debugImage.setTo(0);//��ͼƬ���ص�ȫ������Ϊ0
	for (int i = 0; i < (int)samples.size(); ++i)
	{
		int x = (int)(samples[i].XMin() - centre.XMin());//qyy���������Ͻ�֮��ľ��룬Ϊʲô������������ĵ�ľ��룿
		int y = (int)(samples[i].YMin() - centre.YMin());
		//����score�����ĸߵͣ���image����ʾ��ͬ����ɫ�������������ɫ��һ����0-1�ˣ�Ӧ���Ǹ�float����������ɣ���
		m_debugImage.at<float>(m_config.searchRadius+y, m_config.searchRadius+x) = (float)((scores[i]-mn)/(mx-mn));
	}
}

void Tracker::Debug()
{
	imshow("tracker", m_debugImage);//��ʾ���ղ���UpdateDebugImage����µ�m_debugImage
	m_pLearner->Debug();
}

void Tracker::UpdateLearner(const ImageRep& image)
{
	// note these return the centre sample at index 0
	vector<FloatRect> rects = Sampler::RadialSamples(m_bb, 2*m_config.searchRadius, 5, 16);
	//vector<FloatRect> rects = Sampler::PixelSamples(m_bb, 2*m_config.searchRadius, true);
	
	vector<FloatRect> keptRects;
	keptRects.push_back(rects[0]); // the true sample����0��rect����ֵ
	for (int i = 1; i < (int)rects.size(); ++i)
	{
		if (!rects[i].IsInside(image.GetRect())) continue;//����ͼ��Χ�Ŀ򣬱�������
		keptRects.push_back(rects[i]);
	}
		
#if VERBOSE		
	cout << keptRects.size() << " samples" << endl;
#endif
		
	MultiSample sample(image, keptRects);
	m_pLearner->Update(sample, 0);//����larank������
}
