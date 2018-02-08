#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;

#define V_PROJECT 1
#define H_PROJECT 2

typedef struct
{
	int begin;
	int end;

}char_range_t;


//获取文本的投影用于分割字符(垂直，水平)
int GetTextProjection(Mat &src, vector<int>& pos, int mode)
{
	if (mode == V_PROJECT)
	{
		for (int i = 0; i < src.rows; i++)
		{
			uchar* p = src.ptr<uchar>(i);
			for (int j = 0; j < src.cols; j++)
			{
				if (p[j] == 0)
				{
					pos[j]++;
				}
			}
		}
	}
	else if (mode == H_PROJECT)
	{
		for (int i = 0; i < src.cols; i++)
		{

			for (int j = 0; j < src.rows; j++)
			{
				if (src.at<uchar>(j, i) == 0)
				{
					pos[j]++;
				}
			}
		}
	}
	return 0;
}

//获取每个分割字符的范围，min_thresh：波峰的最小幅度，min_range：两个波峰的最小间隔
int GetPeekRange(vector<int> &vertical_pos, vector<char_range_t> &peek_range, int min_thresh = 2, int min_range = 10)
{
	int begin = 0;
	int end = 0;
	for (int i = 0; i < vertical_pos.size(); i++)
	{
		if (vertical_pos[i] > min_thresh && begin == 0)
		{
			begin = i;
		}
		else if (vertical_pos[i] > min_thresh && begin != 0)
		{
			continue;
		}
		else if (vertical_pos[i] < min_thresh && begin != 0)
		{
			end = i;
			if (end - begin >= min_range)
			{
				char_range_t tmp;
				tmp.begin = begin;
				tmp.end = end;
				peek_range.push_back(tmp);

			}
			begin = 0;
			end = 0;
		}
		else if (vertical_pos[i] < min_thresh || begin == 0)
		{
			continue;
		}
		else
		{
			//printf("raise error!\n");
		}
	}
	return 0;
}


inline void save_cut(const Mat& img, int id)
{
	char name[128] = { 0 };
	sprintf(name, "./save_cut/%d.jpg", id);
	imwrite(name, img);
}

//切割字符
int CutChar(Mat &img, const vector<char_range_t>& v_peek_range, const vector<char_range_t>& h_peek_range, vector<Mat>& chars_set)
{
	static int count = 0;
	Mat show_img = img.clone();
	cvtColor(show_img, show_img, CV_GRAY2BGR);
	for (int i = 0; i < v_peek_range.size(); i++)
	{
		Rect r(v_peek_range[i].begin, 0, v_peek_range[i].end - v_peek_range[i].begin, img.rows);
		rectangle(show_img, r, Scalar(255, 0, 0), 2);
		Mat single_char = img(r).clone();
		chars_set.push_back(single_char);
		save_cut(single_char, count);
		count++;
	}

	imshow("cut", show_img);
	waitKey();

	return 0;
}

Mat cut_one_line(const Mat& src,int begin,int end)
{
	//printf("end:%d  begin:%d\n", end, begin);
	Mat line = src(Rect(0,begin,src.cols,end-begin)).clone();
	return line;
}

vector<Mat> CutSingleChar(Mat& img)
{
	//resize(img, img, Size(), 1.5, 1.5, INTER_LANCZOS4);
	threshold(img, img, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	imshow("binary", img);
	vector<int> horizion_pos(img.rows, 0);
	vector<char_range_t> h_peek_range;
	GetTextProjection(img, horizion_pos, H_PROJECT);
	GetPeekRange(horizion_pos, h_peek_range, 10, 10);
	
	/*将每一文本行切割*/
	vector<Mat> lines_set;
	for (int i = 0; i < h_peek_range.size(); i++)
	{
		Mat line = cut_one_line(img, h_peek_range[i].begin, h_peek_range[i].end);
		lines_set.push_back(line);		
	}

	vector<Mat> chars_set;
	for (int i = 0; i < lines_set.size(); i++)
	{
		//printf("test %d\n", i);
		Mat line = lines_set[i];
		vector<int> vertical_pos(line.cols, 0);
		vector<char_range_t> v_peek_range;
		GetTextProjection(line, vertical_pos, V_PROJECT);
		GetPeekRange(vertical_pos, v_peek_range);
		CutChar(line, v_peek_range, h_peek_range, chars_set);
	}
	return chars_set;
}

int main()
{
	Mat img = imread("12.png", 0);
	resize(img, img,Size(),2,2);
	vector<Mat> chars_set = CutSingleChar(img);

	for (int i = 0; i < chars_set.size(); i++)
	{
		/*字符识别*/
	}

	waitKey();
	return 0;
}
