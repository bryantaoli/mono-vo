/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>

#include "config.h"
#include "visual_odometry.h"

namespace myslam
{

VisualOdometry::VisualOdometry() :
    state_ ( NOT_INITIALIZED ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 ), matcher_flann_ ( new cv::flann::LshIndexParams ( 5,10,2 ) )
{
    num_of_features_    = Config::get<int> ( "number_of_features" );
    scale_factor_       = Config::get<double> ( "scale_factor" );
    level_pyramid_      = Config::get<int> ( "level_pyramid" );
    match_ratio_        = Config::get<float> ( "match_ratio" );
    max_num_lost_       = Config::get<float> ( "max_num_lost" );
    min_inliers_        = Config::get<int> ( "min_inliers" );
    key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
    map_point_erase_ratio_ = Config::get<double> ( "map_point_erase_ratio" );
    orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
}

VisualOdometry::~VisualOdometry()
{

}

bool VisualOdometry::addFrame ( Frame::Ptr frame )
{
    switch ( state_ )
    {
	case NOT_INITIALIZED:
	{
		state_ = INITIALIZING;
		ref_ = frame;//��ǰ֡�Ͳο�֡һ��
							 // extract features from first frame and add them into map
		extractKeyPoints();
		computeDescriptors();
		keypoints_ref_ = keypoints_curr_;
		descriptors_ref_ = descriptors_curr_;
		
		break;
	}
    case INITIALIZING:
    {
        
        curr_ = frame;//��ǰ֡�Ͳο�֡һ��
        // extract features from first frame and add them into map
		// ��ȡ����curr_�Ĺؼ��㣬�������curr_��������
		std::vector<cv::KeyPoint> keypoints_1 = keypoints_ref_;
		Mat descriptors_1 = descriptors_ref_;
        extractKeyPoints();
        computeDescriptors();
		std::vector<cv::KeyPoint> keypoints_2 = keypoints_curr_;
		Mat descriptors_2 = descriptors_curr_;
		cv::Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
		vector<DMatch> initialmatch, goodmatches;
		matcher->match(descriptors_1, descriptors_2, initialmatch);
		matcher->match(descriptors_1, descriptors_2, initialmatch);
		double min_dist = 10000, max_dist = 0;
		for (int i = 0; i < descriptors_1.rows; i++) {
			double dist = initialmatch[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}
		for (int i = 0; i < descriptors_1.rows; i++) {
			if (initialmatch[i].distance <= max(2 * min_dist, 30.0)) {
				goodmatches.push_back(initialmatch[i]);
			}
		}


		Mat R12_2d2d, t12_2d2d, r12_2d2d;

		pose_estimation_2d2d(keypoints_1, keypoints_2, goodmatches, R12_2d2d, t12_2d2d);
		cv::Rodrigues(R12_2d2d, r12_2d2d);
		vector<Point3f> pts3d_1_2;//���ǻ������  ��������1Ϊ���ϵ��camϵ�����꣨��Ϊ����ϵ��
		triangulation(keypoints_1, keypoints_2, goodmatches, R12_2d2d, t12_2d2d, pts3d_1_2);
		//��������һ������Ȼ����pts_2d_pic1_pic2�����������ȣ�������������pts_2d_pic1_pic2��double�������Կռ任ʱ��ķ���������
		vector<Point2d> pts_2d_1_2;//ͼ2�������㰴  ͼ1��ͼ2֮���ƥ��˳�����
		for (DMatch m : goodmatches) {
			pts_2d_1_2.push_back(keypoints_2[m.trainIdx].pt);
			//pts_2d_1_2.push_back(keypoints_1[m.queryIdx].pt);
		}
		Mat r12_3d2d, t12_3d2d;
		Mat K = (Mat_<double>(3, 3) << ref_->camera_->fx_, 0, ref_->camera_->cx_, 0, ref_->camera_->fy_, ref_->camera_->cy_, 0, 0, 1);
		cv::solvePnP(pts3d_1_2, pts_2d_1_2, K, Mat(), r12_3d2d, t12_3d2d, false, cv::SOLVEPNP_EPNP);
		cout << "r12_3d2d:" << endl << r12_3d2d << endl;		
		cout << "t12_3d2d:" << endl << t12_3d2d << endl << endl;
		//�����ʼ���Ļ��߲�����,���ǻ����ɵ���ά����������¸�ֵ
		vector<MapPoint::Map_point_litao> pts3d_2_1;
		pts3d_2_1.clear();
		for (size_t i = 0; i < pts3d_1_2.size(); i++) {
			MapPoint::Map_point_litao aa;
			aa.TrainId = goodmatches[i].trainIdx;
			aa.pts3d = pts3d_1_2[i]; 
			pts3d_2_1.push_back(aa);
		}
		double delta_t1 = t12_3d2d.at<double>(0) - t12_2d2d.at<double>(0); 
		double delta_t2 = t12_3d2d.at<double>(1) - t12_2d2d.at<double>(1); 
		double delta_t3 = t12_3d2d.at<double>(2) - t12_2d2d.at<double>(2);


		//����Լ����ν����pnp�����࣬��Ϊ�ɹ�
		if ((delta_t1 *delta_t1 + delta_t2 * delta_t2 + delta_t3 * delta_t3) < 0.05 ) {
			//addKeyFrame();      // the first frame is a key-frame
			if (map_->keyframes_.empty())
			{
				// first key-frame, add all 3d points into map
				for (size_t i = 0; i<keypoints_curr_.size(); i++)
				{
					/*double d = curr_->findDepth ( keypoints_curr_[i] );
					if ( d < 0 )
					continue;*/
					Vector3d p_world = ref_->camera_->pixel2world(
						Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), curr_->T_c_w_, d
						);
					Vector3d n = p_world - ref_->getCamCenter();
					n.normalize();
					MapPoint::Ptr map_point = MapPoint::createMapPoint(
						p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
						);
					map_->insertMapPoint(map_point);
				}
			}

			map_->insertKeyFrame(ref_);
			ref_ = curr_;

			state_ = OK;
		}
        break;
    }
    case OK:
    {
        curr_ = frame;
        curr_->T_c_w_ = ref_->T_c_w_;
        extractKeyPoints();
        computeDescriptors();
        featureMatching();
        poseEstimationPnP();
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            curr_->T_c_w_ = T_c_w_estimated_;
            optimizeMap();
            num_lost_ = 0;
            if ( checkKeyFrame() == true ) // is a key-frame
            {
                addKeyFrame();
            }
        }
        else // bad estimation due to various reasons
        {
            num_lost_++;
            if ( num_lost_ > max_num_lost_ )
            {
                state_ = LOST;
            }
            return false;
        }
        break;
    }
    case LOST:
    {
        cout<<"vo has lost."<<endl;
        break;
    }
    }

    return true;
}

void VisualOdometry::pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1, std::vector<KeyPoint> keypoints_2, std::vector<DMatch> matches, Mat & R, Mat & t)
{
	// ����ڲ�,TUM Freiburg2
	//-- ��ƥ���ת��Ϊvector<Point2f>����ʽ
	vector<Point2f> points1;
	vector<Point2f> points2;
	for (int i = 0; i < (int)matches.size(); i++)
	{
		points1.push_back(keypoints_1[matches[i].queryIdx].pt);
		points2.push_back(keypoints_2[matches[i].trainIdx].pt);
	}
	//-- �����������
	Mat fundamental_matrix;
	fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
	//cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

	//-- ���㱾�ʾ���
	Point2d principal_point(ref_->camera_->cx_, ref_->camera_->cy_);	//�������, TUM dataset�궨ֵ
	double focal_length = (ref_->camera_->fx_ + ref_->camera_->fy_) / 2.0;			//�������, TUM dataset�궨ֵ
	Mat essential_matrix;
	essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
	//cout << "essential_matrix is " << endl << essential_matrix << endl;

	//-- ���㵥Ӧ����
	Mat homography_matrix;
	homography_matrix = findHomography(points1, points2, RANSAC, 3);
	//cout << "homography_matrix is " << endl << homography_matrix << endl;

	//-- �ӱ��ʾ����лָ���ת��ƽ����Ϣ.
	recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
	Mat r;
	cv::Rodrigues(R, r);
}

void VisualOdometry::triangulation(const vector<KeyPoint>& keypoint_1, const vector<KeyPoint>& keypoint_2, const std::vector<DMatch>& matches, const Mat & R, const Mat & t, vector<Point3f>& points)
{
	Mat T1 = (Mat_<float>(3, 4) <<
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0);
	Mat T2 = (Mat_<float>(3, 4) <<
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
		);

	Mat K = (Mat_<double>(3, 3) << ref_->camera_->fx_, 0, ref_->camera_->cx_, 0, ref_->camera_->fy_, ref_->camera_->cy_, 0, 0, 1);
	vector<Point2f> pts_1, pts_2;
	//vector<Point3d> ps_3d_12;
	for (DMatch m : matches)
	{
		// ����������ת�����������
		pts_1.push_back(Camera::pixel2cam(keypoint_1[m.queryIdx].pt, K));
		pts_2.push_back(Camera::pixel2cam(keypoint_2[m.trainIdx].pt, K));

		Mat pts_4d;
		cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

		// ת���ɷ��������
		for (int i = 0; i < pts_4d.cols; i++)
		{
			Mat x = pts_4d.col(i);
			x /= x.at<double>(3, 0); // ��һ��
			Point3f p(
				x.at<double>(0, 0),
				x.at<double>(1, 0),
				x.at<double>(2, 0)
				);
			points.push_back(p);
		}
	}
}

void VisualOdometry::extractKeyPoints()
{
	//curr_��һ��public������������߲���Ҫ�������������Ƽ��Ժ��Լ�д����Ҳ��ôд
    
    orb_->detect ( curr_->color_, keypoints_curr_ );
}

void VisualOdometry::computeDescriptors()
{
    
    orb_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_ );
}



void VisualOdometry::featureMatching()
{
    
    vector<cv::DMatch> matches;
    // select the candidates in map 
    Mat desp_map;
    vector<MapPoint::Ptr> candidate;  //candidat ��ѡ
    for ( auto& allpoints: map_->map_points_ )
    {
        MapPoint::Ptr& p = allpoints.second;
        // check if p in curr frame image 
        if ( curr_->isInFrame(p->pos_) )
        {
            // add to candidate 
            p->visible_times_++;
            candidate.push_back( p );
            desp_map.push_back( p->descriptor_ );
        }
    }
    
    matcher_flann_.match ( desp_map, descriptors_curr_, matches );
    // select the best matches
    float min_dis = std::min_element (
                        matches.begin(), matches.end(),
                        [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
    {
        return m1.distance < m2.distance;
    } )->distance;

    match_3dpts_.clear();
    match_2dkp_index_.clear();
    for ( cv::DMatch& m : matches )
    {
        if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
        {
            match_3dpts_.push_back( candidate[m.queryIdx] );
            match_2dkp_index_.push_back( m.trainIdx );
        }
    }
    cout<<"good matches: "<<match_3dpts_.size() <<endl;
}

void VisualOdometry::poseEstimationPnP()
{
    // construct the 3d 2d observations
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;

    for ( int index:match_2dkp_index_ )
    {
        pts2d.push_back ( keypoints_curr_[index].pt );
    }
    for ( MapPoint::Ptr pt:match_3dpts_ )
    {
        pts3d.push_back( pt->getPositionCV() );
    }

    Mat K = ( cv::Mat_<double> ( 3,3 ) <<
              ref_->camera_->fx_, 0, ref_->camera_->cx_,
              0, ref_->camera_->fy_, ref_->camera_->cy_,
              0,0,1
            );
    Mat rvec, tvec, inliers;
    cv::solvePnPRansac ( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
    num_inliers_ = inliers.rows;
    cout<<"pnp inliers: "<<num_inliers_<<endl;
    T_c_w_estimated_ = SE3 (
                           SO3 ( rvec.at<double> ( 0,0 ), rvec.at<double> ( 1,0 ), rvec.at<double> ( 2,0 ) ),
                           Vector3d ( tvec.at<double> ( 0,0 ), tvec.at<double> ( 1,0 ), tvec.at<double> ( 2,0 ) )
                       );

    // using bundle adjustment to optimize the pose
    
    
    cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;
}

bool VisualOdometry::checkEstimatedPose()
{
    // check if the estimated pose is good
    if ( num_inliers_ < min_inliers_ )
    {
        cout<<"reject because inlier is too small: "<<num_inliers_<<endl;
        return false;
    }
    // if the motion is too large, it is probably wrong
    SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    if ( d.norm() > 5.0 )
    {
        cout<<"reject because motion is too large: "<<d.norm() <<endl;
        return false;
    }
    return true;
}

bool VisualOdometry::checkKeyFrame()
{
    SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}

void VisualOdometry::addKeyFrame()
{
    if ( map_->keyframes_.empty() )
    {
        // first key-frame, add all 3d points into map
        for ( size_t i=0; i<keypoints_curr_.size(); i++ )
        {
            /*double d = curr_->findDepth ( keypoints_curr_[i] );
            if ( d < 0 ) 
                continue;*/
            Vector3d p_world = ref_->camera_->pixel2world (
                Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ), curr_->T_c_w_, d
            );
            Vector3d n = p_world - ref_->getCamCenter();
            n.normalize();
            MapPoint::Ptr map_point = MapPoint::createMapPoint(
                p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
            );
            map_->insertMapPoint( map_point );
        }
    }
    
    map_->insertKeyFrame ( curr_ );
    ref_ = curr_;
}

void VisualOdometry::addMapPoints()
{
    // add the new map points into map
    vector<bool> matched(keypoints_curr_.size(), false); 
    for ( int index:match_2dkp_index_ )
        matched[index] = true;
    for ( int i=0; i<keypoints_curr_.size(); i++ )
    {
        if ( matched[i] == true )   
            continue;
        double d = ref_->findDepth ( keypoints_curr_[i] );
        if ( d<0 )  
            continue;
        Vector3d p_world = ref_->camera_->pixel2world (
            Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ), 
            curr_->T_c_w_, d
        );
        Vector3d n = p_world - ref_->getCamCenter();
        n.normalize();
        MapPoint::Ptr map_point = MapPoint::createMapPoint(
            p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
        );
        map_->insertMapPoint( map_point );
    }
}

void VisualOdometry::optimizeMap()
{
    // remove the hardly seen and no visible points 
    for ( auto iter = map_->map_points_.begin(); iter != map_->map_points_.end(); )
    {
        if ( !curr_->isInFrame(iter->second->pos_) )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        float match_ratio = float(iter->second->matched_times_)/iter->second->visible_times_;
        if ( match_ratio < map_point_erase_ratio_ )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        
        double angle = getViewAngle( curr_, iter->second );
        if ( angle > 3.14159265358979323846/6. )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        if ( iter->second->good_ == false )
        {
            // TODO try triangulate this map point 
        }
        iter++;
    }
    
    if ( match_2dkp_index_.size()<100 )
        addMapPoints();
    if ( map_->map_points_.size() > 1000 )  
    {
        // TODO map is too large, remove some one 
        map_point_erase_ratio_ += 0.05;
    }
    else 
        map_point_erase_ratio_ = 0.1;
    cout<<"map points: "<<map_->map_points_.size()<<endl;
}

double VisualOdometry::getViewAngle ( Frame::Ptr frame, MapPoint::Ptr point )
{
    Vector3d n = point->pos_ - frame->getCamCenter();
    n.normalize();
    return acos( n.transpose()*point->norm_ );
}


}
