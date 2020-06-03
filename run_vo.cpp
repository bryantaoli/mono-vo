// -------------- test the visual odometry -------------
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "config.h"
#include "visual_odometry.h"

int main ( int argc, char** argv )
{
	string inputDocument = "./srcMono/default.yaml";
	myslam::Config::setParameterFile(inputDocument);

    myslam::VisualOdometry::Ptr vo ( new myslam::VisualOdometry );//构造函数，但是vo是一个指针

    string dataset_dir = myslam::Config::get<string> ( "dataset_dir" );//运行完dataset_dir变量会被赋予“dataset_dir”这个值
    cout<<"dataset: "<<dataset_dir<<endl;
    ifstream fin ( dataset_dir+"/associate.txt" );
    if ( !fin )
    {
        cout<<"please generate the associate file called associate.txt!"<<endl;
        return 1;
    }

    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while ( !fin.eof() )
    {
        string rgb_time, rgb_file, depth_time, depth_file;
		//fin就是associate.txt
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_times.push_back ( atof ( rgb_time.c_str() ) );
        depth_times.push_back ( atof ( depth_time.c_str() ) );
        rgb_files.push_back ( dataset_dir+"/"+rgb_file );
        depth_files.push_back ( dataset_dir+"/"+depth_file );

        if ( fin.good() == false )
            break;
    }

    myslam::Camera::Ptr camera ( new myslam::Camera );

    

    cout<<"read total "<<rgb_files.size() <<" entries"<<endl;
    for ( int i=0; i<rgb_files.size(); i++ )
    {
        cout<<"****** loop "<<i<<" ******"<<endl;
        Mat color = cv::imread ( rgb_files[i] );
        //Mat depth = cv::imread ( depth_files[i], -1 );
		if (color.data == nullptr) {
			break;
		}
            
        myslam::Frame::Ptr pFrame = myslam::Frame::createFrame();
        pFrame->camera_ = camera;
        pFrame->color_ = color;
        //pFrame->depth_ = depth;
        pFrame->time_stamp_ = rgb_times[i];

        vo->addFrame ( pFrame );

        if ( vo->state_ == myslam::VisualOdometry::LOST )
            break;
        SE3 Twc = pFrame->T_c_w_.inverse();

        
    }

    return 0;
}
