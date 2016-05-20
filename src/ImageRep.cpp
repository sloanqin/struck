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

#include "ImageRep.h"

#include <cassert>

#include <opencv/highgui.h>

using namespace std;
using namespace cv;

static const int kNumBins = 16;

ImageRep::ImageRep(const Mat& image, bool computeIntegral, bool computeIntegralHist, bool colour) :
	m_channels(colour ? 3 : 1),//����colour��true��false��ѡ��3ͨ������1ͨ��
	m_rect(0, 0, image.cols, image.rows)
{	
	for (int i = 0; i < m_channels; ++i)
	{
		m_images.push_back(Mat(image.rows, image.cols, CV_8UC1));//����һ��Mat���������imageͬ��С
		if (computeIntegral) m_integralImages.push_back(Mat(image.rows+1, image.cols+1, CV_32SC1));//��������ͼMat
		if (computeIntegralHist)
		{
			for (int j = 0; j < kNumBins; ++j)
			{
				m_integralHistImages.push_back(Mat(image.rows+1, image.cols+1, CV_32SC1));
			}
		}
	}
		
	if (colour)
	{
		assert(image.channels() == 3);
		split(image, m_images);//��Ӧ���ǽ�image��3��ͨ�����ѵ�m_images[0] m_images[1] m_images[2]
	}
	else
	{
		assert(image.channels() == 1 || image.channels() == 3);
		if (image.channels() == 3)
		{
			cvtColor(image, m_images[0], CV_RGB2GRAY);//�������image����m_images��
		}
		else if (image.channels() == 1)
		{
			image.copyTo(m_images[0]);
		}
	}
	
	if (computeIntegral)
	{
		for (int i = 0; i < m_channels; ++i)
		{
			//equalizeHist(m_images[i], m_images[i]);
			//�������ͼ��ʹ�û���ͼ���Ժܷ�������haar����
			//�ο�blog��http://blog.csdn.net/sloanqin/article/details/50530246
			integral(m_images[i], m_integralImages[i]);
		}
	}
	
	if (computeIntegralHist)
	{
		Mat tmp(image.rows, image.cols, CV_8UC1);
		tmp.setTo(0);
		for (int j = 0; j < kNumBins; ++j)
		{
			for (int y = 0; y < image.rows; ++y)
			{
				const uchar* src = m_images[0].ptr(y);
				uchar* dst = tmp.ptr(y);
				for (int x = 0; x < image.cols; ++x)
				{
					int bin = (int)(((float)*src/256)*kNumBins);
					*dst = (bin == j) ? 1 : 0;
					++src;
					++dst;
				}
			}
			
			integral(tmp, m_integralHistImages[j]);			
		}
	}
}

int ImageRep::Sum(const IntRect& rRect, int channel) const
{
	//����ʹ��assert��������飬�Ǻܺõ�ϰ�ߣ�ֵ��ѧϰ
	assert(rRect.XMin() >= 0 && rRect.YMin() >= 0 && rRect.XMax() <= m_images[0].cols && rRect.YMax() <= m_images[0].rows);
	return m_integralImages[channel].at<int>(rRect.YMin(), rRect.XMin()) +
			m_integralImages[channel].at<int>(rRect.YMax(), rRect.XMax()) -
			m_integralImages[channel].at<int>(rRect.YMax(), rRect.XMin()) -
			m_integralImages[channel].at<int>(rRect.YMin(), rRect.XMax());//���ؾ��ο��ڵ����غ�
}

void ImageRep::Hist(const IntRect& rRect, Eigen::VectorXd& h) const
{
	assert(rRect.XMin() >= 0 && rRect.YMin() >= 0 && rRect.XMax() <= m_images[0].cols && rRect.YMax() <= m_images[0].rows);
	int norm = rRect.Area();
	for (int i = 0; i < kNumBins; ++i)
	{
		int sum = m_integralHistImages[i].at<int>(rRect.YMin(), rRect.XMin()) +
			m_integralHistImages[i].at<int>(rRect.YMax(), rRect.XMax()) -
			m_integralHistImages[i].at<int>(rRect.YMax(), rRect.XMin()) -
			m_integralHistImages[i].at<int>(rRect.YMin(), rRect.XMax());
		h[i] = (float)sum/norm;
	}
}
